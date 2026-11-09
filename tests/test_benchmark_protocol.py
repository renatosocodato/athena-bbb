"""No-download checks on the public benchmark's explicit curation contract."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

script = Path(__file__).resolve().parents[1] / 'validation/run_b3db_benchmark.py'
spec = importlib.util.spec_from_file_location('athena_b3db_protocol',script)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_curation_removes_all_conflicts_and_collapses_canonical_duplicates():
    raw = pd.DataFrame({'SMILES':['CCO','[CH3:1][CH2:2][OH:3]','CCN','[NH2:1][CH2:2][CH3:3]','CCCl','ClCC',''],
                        'BBB+/BBB-':['BBB+','BBB-','BBB+','BBB+','BBB-','BBB-','BBB+']})
    clean, report = benchmark.curate(raw)
    assert report == {'raw_rows':7,'invalid_rows':1,'conflicting_rows':2,'canonical_duplicate_rows':2,'retained':2}
    assert set(clean.smiles)=={'CCN','CCCl'}
    assert dict(zip(clean.smiles,clean.label))=={'CCN':1,'CCCl':0}


def test_one_class_does_not_produce_balanced_or_auc_claim():
    result = benchmark.metrics(np.array([1,1,1]),np.array([.8,.9,.7]))
    assert result['roc_auc'] is None and result['balanced_accuracy'] is None
    assert np.isclose(result['brier_score'],(.04+.01+.09)/3)
