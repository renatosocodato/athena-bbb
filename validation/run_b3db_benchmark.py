"""Frozen B3DB software benchmark; downloads public CC0 data, never private inputs.

Run after installation: python validation/run_b3db_benchmark.py --output NEW_DIR
Protocol is fixed in BENCHMARK_PROTOCOL.json; no hyperparameter/seed search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from athena_bbb import core

COMMIT = '75dab1cc607bb7a03f3de5c576cffd223e767844'
FILES = {
    'B3DB_classification.tsv': '47a160aea1551423ead2aab70b301fb1646aea03cf151fd614eb37c78fa55b82',
    'B3DB_classification_external.tsv': '84940d866f7cb53b361c38e787143c20b65641b57040006e020609f9d6f242cf',
}


def acquire(directory, name):
    path = directory / name
    url = f'https://raw.githubusercontent.com/theochem/B3DB/{COMMIT}/B3DB/{name}'
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != FILES[name]:
        raise ValueError('Dataset checksum mismatch: ' + name)
    return pd.read_csv(path, sep='\t')


def curate(frame):
    rows = []
    invalid = 0
    for row in frame[['SMILES', 'BBB+/BBB-']].itertuples(index=False, name=None):
        smiles, label = row
        if label not in ('BBB+', 'BBB-'):
            invalid += 1
            continue
        try:
            _, canonical = core._canonical_molecule(smiles)
        except ValueError:
            invalid += 1
            continue
        rows.append({'smiles': canonical, 'label': int(label == 'BBB+')})
    data = pd.DataFrame(rows, columns=['smiles', 'label'])
    conflicts = data.groupby('smiles').label.nunique()
    conflict_names = set(conflicts[conflicts > 1].index)
    conflict_rows = int(data.smiles.isin(conflict_names).sum())
    data = data.loc[~data.smiles.isin(conflict_names)]
    before = len(data)
    data = core.prepare(data)
    return data, {'raw_rows': len(frame), 'invalid_rows': invalid, 'conflicting_rows': conflict_rows,
                  'canonical_duplicate_rows': before-len(data), 'retained': len(data)}


def metrics(y, p):
    both = len(np.unique(y)) == 2
    return {'balanced_accuracy': float(balanced_accuracy_score(y, p >= .5)) if both else None,
            'roc_auc': float(roc_auc_score(y, p)) if both else None,
            'brier_score': float(brier_score_loss(y, p))}


def evaluate(frame, models):
    if frame.empty:
        return {'n': 0, 'note': 'No disjoint external molecules retained', 'models': {}}
    x = core.features(frame.smiles); y = frame.label.to_numpy()
    probabilities = {name: model.predict_proba(x)[:, 1] for name, model in models.items()}
    result = {'n': len(frame), 'class_counts': {str(k): int(v) for k, v in frame.label.value_counts().items()},
              'scaffolds': frame.scaffold.nunique(), 'models': {name: metrics(y, p) for name, p in probabilities.items()}}
    groups = frame.scaffold.to_numpy(); unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(8841)
    bootstrap = {name: {metric: [] for metric in ('balanced_accuracy','roc_auc','brier_score')} for name in models}
    differences = {name: {metric: [] for metric in ('balanced_accuracy','roc_auc','brier_score')} for name in models if name != 'athena_rf'}
    for _ in range(2000):
        sample = np.concatenate([indices[i] for i in rng.integers(len(indices),size=len(indices))])
        values = {name: metrics(y[sample], p[sample]) for name, p in probabilities.items()}
        for name in models:
            for metric, value in values[name].items():
                if value is not None:
                    bootstrap[name][metric].append(value)
                    if name != 'athena_rf':
                        differences[name][metric].append(values['athena_rf'][metric]-value)
    def intervals(items):
        return {metric: {'low':float(np.quantile(values,.025)), 'high':float(np.quantile(values,.975)),
                         'valid_draws':len(values)} if values else {'low':None,'high':None,'valid_draws':0}
                for metric,values in items.items()}
    for name in models:
        result['models'][name]['conditional_scaffold_bootstrap_95'] = intervals(bootstrap[name])
    result['rf_minus_baseline_paired_intervals'] = {name: intervals(items) for name,items in differences.items()}
    result['bootstrap_draws_without_both_classes'] = 2000-len(bootstrap['athena_rf']['roc_auc'])
    influences = {}
    for name,p in probabilities.items():
        values=[]
        for index in indices:
            keep=np.ones(len(y),dtype=bool);keep[index]=False
            if keep.any() and len(np.unique(y[keep]))==2:
                values.append(float(balanced_accuracy_score(y[keep],p[keep]>=.5)))
        influences[name]={'min_balanced_accuracy':min(values) if values else None,
                          'max_balanced_accuracy':max(values) if values else None,
                          'valid_omissions':len(values)}
    result['leave_one_test_scaffold_out'] = influences
    result['uncertainty_scope'] = 'Conditional on fixed fitted models and observed scaffold groups; no training refit or cohort/species sampling uncertainty included'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output directory must be new')
    args.output.mkdir(parents=True)
    dev,dev_report=curate(acquire(args.output,'B3DB_classification.tsv'))
    external,external_report=curate(acquire(args.output,'B3DB_classification_external.tsv'))
    model,internal_metrics,assignment=core.train(dev[['smiles','label']],seed=2026,test_size=.25)
    train=assignment.loc[assignment.split.eq('train')]
    holdout=assignment.loc[assignment.split.eq('test')]
    x=core.features(train.smiles);y=train.label
    prior=DummyClassifier(strategy='prior').fit(x,y)
    logistic=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=2026)).fit(x,y)
    models={'athena_rf':model,'training_prior':prior,'scaled_logistic':logistic}
    overlap_structure=external.smiles.isin(set(assignment.smiles))
    overlap_scaffold=external.scaffold.isin(set(assignment.scaffold))
    disjoint=external.loc[~(overlap_structure|overlap_scaffold)].reset_index(drop=True)
    summary={'protocol':'BENCHMARK_PROTOCOL.json, frozen before outcomes','upstream_commit':COMMIT,'upstream_license':'CC0-1.0',
             'development_curation':dev_report,'external_curation':external_report,
             'split':{'seed':2026,'test_size':.25,'train':len(train),'holdout':len(holdout),'grouping':'Murcko rings; exact canonical structure for acyclic molecules'},
             'external_exclusion':{'canonical_overlap':int(overlap_structure.sum()),'scaffold_overlap':int(overlap_scaffold.sum()),
                                   'union_removed':int((overlap_structure|overlap_scaffold).sum()),'retained':len(disjoint)},
             'internal_holdout':evaluate(holdout,models),'strict_external':evaluate(disjoint,models),
             'interpretation_limit':'Classification on heterogeneous historical B3DB labels, not validated human BBB probabilities, treatment effects or prospective validation',
             'versions':core.model_metadata(model)['versions']}
    if not disjoint.empty:
        summary['strict_external']['outside_training_descriptor_range']=int(core.applicability(model,disjoint.smiles).sum())
        if disjoint.label.nunique()==2:
            summary['strict_external']['core_evaluate_external']=core.evaluate_external(model,None,disjoint[['smiles','label']])
    assignment.to_csv(args.output/'development_split.csv',index=False)
    disjoint.to_csv(args.output/'external_disjoint.csv',index=False)
    (args.output/'benchmark_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:summary[key] for key in ['development_curation','external_exclusion','split']},indent=2))


if __name__=='__main__':main()
