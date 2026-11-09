"""Retrospective cross-corpus BBB test with a fixed development model.

Run with the installed ATHENA BBB package and the original B3DB runner on PYTHONPATH.
Files are downloaded into the private evidence folder and verified before use.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from athena_bbb import core
from run_b3db_benchmark import acquire, curate, evaluate

COMMIT = "c35e6366391f74605b625fdcfdd126b7f1ac3a3f"
FILES = {
    "train": "eedfdce2f76210747d1424dba4f5b93addabeffce0fa673773fe90818ee4508f",
    "valid": "a9b79d5affe4c4e5cbe962894c96c17b804e8a7c770973f46faeb3a3fc09c3d1",
    "test": "aac236cb9ac46e9fa96cb8b6bc6f857e6cde9699e85be7494aca87e1679f3fcf",
}


def fetch(directory, split):
    path = directory / f"neutox_smiles_{split}.csv"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/xuejunhe/NeuTox-2.0/{COMMIT}/BBB/random0/smiles/{split}.csv"
        with urlopen(url, timeout=30) as response:
            path.write_bytes(response.read())
    if sha256(path.read_bytes()).hexdigest() != FILES[split]:
        raise ValueError(f"Source hash mismatch for {split}")
    return pd.read_csv(path)


def parent_keys(smiles):
    mol = rdMolStandardize.ChargeParent(Chem.MolFromSmiles(smiles))
    # Standard InChIKey connectivity ignores stereo and canonicalizes some tautomers.
    key = Chem.MolToInchiKey(mol).split("-")[0]
    if not key:
        raise ValueError("Cannot verify parent connectivity")
    return key, MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def attach_keys(table, exclude_unverified):
    keys = []
    for smiles in table.smiles:
        try:
            keys.append(parent_keys(smiles))
        except (RuntimeError, ValueError):
            if not exclude_unverified:
                raise
            keys.append((None, None))
    result = table.copy()
    result["parent"], result["parent_scaffold"] = zip(*keys)
    return result


def counts(table):
    return {
        "molecules": len(table),
        "positive": int(table.label.eq(1).sum()),
        "negative": int(table.label.eq(0).sum()),
        "groups": int(table.scaffold.nunique()),
        "negative_groups": int(table.loc[table.label.eq(0), "scaffold"].nunique()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    destination = args.output / "neutox_results.json"
    if destination.exists():
        raise FileExistsError(destination)
    RDLogger.DisableLog("rdApp.warning")
    dev, _ = curate(acquire(args.b3db, "B3DB_classification.tsv"))
    old_external, _ = curate(acquire(args.b3db, "B3DB_classification_external.tsv"))
    raw = pd.concat([fetch(args.output, split) for split in FILES], ignore_index=True)
    ext, curation = curate(raw.assign(**{"BBB+/BBB-": raw.active.map({1: "BBB+", 0: "BBB-"})}))
    known = attach_keys(pd.concat([dev, old_external]).drop_duplicates("smiles"), False)
    ext = attach_keys(ext, True)
    unverified = ext.parent.isna()
    overlap = (
        ext.smiles.isin(known.smiles) | ext.scaffold.isin(known.scaffold)
        | ext.parent.isin(known.parent)
        | (ext.parent_scaffold.ne("") & ext.parent_scaffold.isin(known.parent_scaffold))
    )
    disjoint = ext.loc[~(overlap | unverified)].copy()
    conflicting = disjoint.groupby("parent").label.transform("nunique").gt(1)
    heldout = disjoint.loc[~conflicting].sort_values("smiles").drop_duplicates("parent").copy()
    # Keep groups at least as coarse as standardized parent ring scaffolds.
    heldout["scaffold"] = heldout.parent_scaffold.where(heldout.parent_scaffold.ne(""), heldout.parent)
    print(json.dumps({"curation": curation, "excluded_overlap": int(overlap.sum()),
                      "excluded_unverified": int(unverified.sum()), "after_filter": counts(heldout)}), flush=True)

    model, _, assignment = core.train(dev[["smiles", "label"]], seed=2026, test_size=.25)
    train = assignment.loc[assignment.split.eq("train")]
    x = core.features(train.smiles)
    models = {
        "athena_rf": model,
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced",
                                                                         random_state=2026)).fit(x, train.label),
        "prior": DummyClassifier(strategy="prior").fit(x, train.label),
    }
    report = {
        "source_commit": COMMIT, "sha256": FILES, "curation": curation,
        "excluded_overlap": int(overlap.sum()), "excluded_unverified": int(unverified.sum()),
        "parent_conflict_rows": int(conflicting.sum()), "retained": counts(heldout),
        "source_boundary": "Public retrospective cross-corpus test. The source files do not identify measurement provenance per row.",
        "metrics": evaluate(heldout, models),
    }
    with destination.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False, default=lambda x: x.item())


if __name__ == "__main__":
    main()
