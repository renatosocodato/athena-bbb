"""Scaffold-separated binary classification with an explicit descriptor contract."""
import importlib.metadata
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

DESCRIPTORS = ('MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds')
SCHEMA_VERSION = 2


def _molecule(value):
    """Parse a SMILES with atom mapping annotations removed."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Nonempty SMILES strings required')
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty molecule')
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            atom.SetAtomMapNum(0)
    return mol


def _canonical_molecule(value):
    mol = _molecule(value)
    return mol, Chem.MolToSmiles(mol)


def prepare(frame):
    """Canonicalize structures; reject invalid molecules and conflicting labels.

    Atom maps are removed before computing molecule and scaffold identity.
    Murcko ring scaffolds define groups. Acyclic structures use canonical SMILES
    and can remain chemically similar across groups.
    """
    if not {'smiles', 'label'}.issubset(frame.columns):
        raise ValueError('Columns smiles and label required')
    if frame[['smiles', 'label']].isna().any().any() or not frame.label.isin([0, 1]).all():
        raise ValueError('Binary labels 0/1 and SMILES required')
    rows = []
    for r in frame.itertuples():
        mol, canonical = _canonical_molecule(r.smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or canonical
        rows.append(dict(smiles=canonical, label=int(r.label), scaffold=scaffold))
    clean = pd.DataFrame(rows)
    if clean.empty:
        raise ValueError('Empty training data')
    if (clean.groupby('smiles').label.nunique() > 1).any():
        raise ValueError('Conflicting labels for same structure')
    return clean.drop_duplicates('smiles').reset_index(drop=True)


def features(smiles):
    """Six ordered RDKit descriptors; inputs must be an iterable of molecules."""
    if isinstance(smiles, str):
        raise ValueError('Supply a sequence of SMILES, not one string')
    rows = []
    for value in smiles:
        molecule = _molecule(value)
        rows.append([getattr(Descriptors, name)(molecule) for name in DESCRIPTORS])
    if not rows:
        raise ValueError('At least one molecule required')
    result = np.asarray(rows, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite descriptors')
    return result


def _metrics(labels, probabilities):
    return {'balanced_accuracy': float(balanced_accuracy_score(labels, probabilities >= .5)),
            'brier_score': float(brier_score_loss(labels, probabilities)),
            'roc_auc': float(roc_auc_score(labels, probabilities))}


def train(frame, seed=2026, test_size=.25):
    """Fit once on a fixed scaffold holdout. Never searches seeds using test scores.

    The returned estimator retains its complete split/corpus and descriptor
    metadata. This travels with joblib serialization and is checked at inference.
    The test partition is not used to refit the returned model.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError('Integer seed required')
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise ValueError('test_size must be strictly between zero and one')
    data = prepare(frame)
    if data.scaffold.nunique() < 2:
        raise ValueError('At least two scaffold groups required')
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(seed)).split(data, groups=data.scaffold))
    if data.iloc[train_idx].label.nunique() != 2 or data.iloc[test_idx].label.nunique() != 2:
        raise ValueError('Both classes required in fixed train and test partitions; supply adequate data')
    model = RandomForestClassifier(n_estimators=100, random_state=int(seed), class_weight='balanced', n_jobs=1)
    model.fit(features(data.iloc[train_idx].smiles), data.iloc[train_idx].label)
    assignment = data.copy()
    assignment['split'] = 'train'
    assignment.loc[test_idx, 'split'] = 'test'
    model.athena_metadata_ = {
        'schema_version': SCHEMA_VERSION, 'descriptors': list(DESCRIPTORS),
        'seed': int(seed), 'test_size': test_size,
        'versions': {name: importlib.metadata.version(name) for name in ('rdkit', 'scikit-learn', 'numpy')},
        'corpus': assignment.to_dict(orient='records'),
        'descriptor_min': features(data.iloc[train_idx].smiles).min(axis=0).tolist(),
        'descriptor_max': features(data.iloc[train_idx].smiles).max(axis=0).tolist(),
    }
    prob = predict(model, data.iloc[test_idx].smiles)
    metrics = _metrics(data.iloc[test_idx].label, prob)
    metrics.update(n_train=len(train_idx), n_test=len(test_idx), seed=int(seed),
                   n_train_scaffolds=data.iloc[train_idx].scaffold.nunique(),
                   n_test_scaffolds=data.iloc[test_idx].scaffold.nunique(),
                   interpretation='Held-out scaffold classification only; Brier score is a diagnostic, not evidence of calibrated human BBB probabilities')
    return model, metrics, assignment


def model_metadata(model):
    metadata = getattr(model, 'athena_metadata_', None)
    if not metadata or metadata.get('schema_version') != SCHEMA_VERSION or metadata.get('descriptors') != list(DESCRIPTORS):
        raise ValueError('Unsupported model or descriptor schema. Train a new model with this version.')
    for name, version in metadata.get('versions', {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Model runtime version differs for {name}; use the recorded environment')
    if list(model.classes_) != [0, 1]:
        raise ValueError('Model must contain binary classes 0 and 1')
    return metadata


def predict(model, smiles):
    """Return class-1 scores; these are not calibrated clinical probabilities."""
    model_metadata(model)
    return model.predict_proba(features(smiles))[:, 1]


def applicability(model, smiles):
    """Flag values outside fitted descriptor ranges; this is a coarse diagnostic."""
    metadata = model_metadata(model)
    x = features(smiles)
    return ((x < np.asarray(metadata['descriptor_min'])) | (x > np.asarray(metadata['descriptor_max']))).any(axis=1)


def evaluate_external(model, training_frame, benchmark):
    """Evaluate a disjoint benchmark without refitting.

    training_frame may be None: the serialized model carries the complete
    development corpus (including holdout). If supplied, it must agree with
    that recorded corpus; callers cannot pass an unrelated table to hide overlap.
    """
    external = prepare(benchmark)
    training = prepare(training_frame) if training_frame is not None else None
    if training is not None and (set(training.smiles) & set(external.smiles) or set(training.scaffold) & set(external.scaffold)):
        raise ValueError('External benchmark overlaps training structures or scaffolds')
    metadata = model_metadata(model)
    recorded = pd.DataFrame(metadata['corpus'])
    if training is not None and set(zip(training.smiles, training.label)) != set(zip(recorded.smiles, recorded.label)):
        raise ValueError('Supplied training table differs from recorded development corpus')
    if set(recorded.smiles) & set(external.smiles) or set(recorded.scaffold) & set(external.scaffold):
        raise ValueError('External benchmark overlaps recorded development structures or scaffolds')
    if external.label.nunique() != 2:
        raise ValueError('Both classes required for external ROC AUC')
    probabilities = predict(model, external.smiles)
    return {'n_external': len(external), **_metrics(external.label, probabilities),
            'class_counts': {str(label): int(external.label.eq(label).sum()) for label in (0, 1)},
            'groups_by_class': {str(label): int(external.loc[external.label.eq(label), 'scaffold'].nunique())
                                for label in (0, 1)},
            'outside_training_descriptor_range': int(applicability(model, external.smiles).sum())}
