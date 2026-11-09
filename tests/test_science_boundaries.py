import numpy as np
import pandas as pd
import pytest
from athena_bbb.core import DESCRIPTORS, SCHEMA_VERSION, prepare, features, train, predict


MAPPED_ETHANOL = '[CH3:1][CH2:2][OH:3]'
MAPPED_ANILINE = '[NH2:7][c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1'

def test_canonical_duplicates_removed():
    x=prepare(pd.DataFrame({'smiles':['CCO','OCC',MAPPED_ETHANOL,'CCN'],'label':[1,1,1,0]}))
    assert len(x)==2

@pytest.mark.parametrize('duplicate', ['OCC', MAPPED_ETHANOL])
def test_conflicts_rejected(duplicate):
    with pytest.raises(ValueError, match='Conflicting labels'):
        prepare(pd.DataFrame({'smiles': ['CCO', duplicate], 'label': [0, 1]}))

def test_invalid_smiles_rejected():
    with pytest.raises(ValueError):features(['not_a_molecule'])

def test_descriptor_reference():
    x=features(['CCO']);assert x.shape==(1,6)
    assert abs(x[0,0]-46.069)<.01;assert x[0,3]==1

def test_scaffold_holdout_and_prediction_reproducible():
    smiles=['C'*i for i in range(1,21)]+['C'*i+'O' for i in range(1,21)]
    x=pd.DataFrame({'smiles':smiles,'label':[i%2 for i in range(40)]})
    model,metrics,assignment=train(x)
    assert set(assignment.loc[assignment.split.eq('train'),'scaffold']).isdisjoint(set(assignment.loc[assignment.split.eq('test'),'scaffold']))
    duplicate = x.loc[x.smiles.eq('CCO')].assign(smiles=MAPPED_ETHANOL)
    _, again, assignment2 = train(pd.concat([x, duplicate], ignore_index=True))
    assert metrics == again
    pd.testing.assert_frame_equal(assignment,assignment2)
    p=predict(model,['CCO']);assert 0<=p[0]<=1
    np.testing.assert_array_equal(p, predict(model, [MAPPED_ETHANOL]))


def test_external_overlap_rejected_before_prediction():
    from athena_bbb.core import evaluate_external
    training=pd.DataFrame({'smiles':['CCO','CCN'],'label':[1,0]})
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_external(None,training,training)

@pytest.mark.parametrize('aniline', ['c1ccccc1N', MAPPED_ANILINE])
def test_distinct_compound_with_same_ring_scaffold_rejected(aniline):
    from athena_bbb.core import evaluate_external
    training=pd.DataFrame({'smiles':['c1ccccc1O','CCN'],'label':[1,0]})
    external=pd.DataFrame({'smiles':[aniline,'CCCCCl'],'label':[1,0]})
    with pytest.raises(ValueError,match='overlaps'):
        evaluate_external(None,training,external)

@pytest.mark.parametrize('smiles', ['', ' ', 123, None])
def test_empty_and_nonstring_molecules_rejected(smiles):
    with pytest.raises(ValueError):features([smiles])


def test_empty_or_scalar_descriptor_input_rejected():
    with pytest.raises(ValueError):features([])
    with pytest.raises(ValueError):features('CCO')


def test_external_evaluation_uses_recorded_corpus_and_survives_serialization(tmp_path):
    import joblib
    from athena_bbb.core import evaluate_external, model_metadata
    data=pd.DataFrame({'smiles':['C'*i for i in range(1,21)]+['C'*i+'O' for i in range(1,21)],'label':[i%2 for i in range(40)]})
    model,_,_=train(data)
    path=tmp_path/'model.joblib';joblib.dump(model,path);restored=joblib.load(path)
    assert model_metadata(restored)['corpus']==model_metadata(model)['corpus']
    external=pd.DataFrame({'smiles':['c1ccccc1','C1CCCCC1'],'label':[0,1]})
    result=evaluate_external(restored,None,external)
    assert result['n_external']==2 and 0<=result['brier_score']<=1
    assert result['class_counts'] == {'0': 1, '1': 1}
    assert result['groups_by_class'] == {'0': 1, '1': 1}
    with pytest.raises(ValueError,match='overlaps'):evaluate_external(restored,None,data)
    mapped_overlap = pd.DataFrame({'smiles': [MAPPED_ETHANOL, 'C1CCCCC1'], 'label': [0, 1]})
    with pytest.raises(ValueError, match='overlaps recorded development'):
        evaluate_external(restored, None, mapped_overlap)
    mapped_training = data.copy()
    mapped_training.loc[mapped_training.smiles.eq('CCO'), 'smiles'] = MAPPED_ETHANOL
    assert evaluate_external(restored, mapped_training, external) == result
    unrelated=pd.DataFrame({'smiles':['C1CCCC1','C1CCC1'],'label':[0,1]})
    with pytest.raises(ValueError,match='differs'):evaluate_external(restored,unrelated,external)


def test_foreign_estimator_rejected_at_prediction():
    with pytest.raises(ValueError,match='schema'):predict(object(),['CCO'])


def test_model_runtime_version_mismatch_rejected():
    from athena_bbb.core import model_metadata
    class Model:
        classes_=[0,1]
        athena_metadata_={'schema_version':SCHEMA_VERSION,'descriptors':list(DESCRIPTORS),'versions':{'rdkit':'0.invalid'}}
    with pytest.raises(ValueError,match='runtime version differs'):model_metadata(Model())


def test_old_model_schema_requires_retraining():
    class OldModel:
        classes_ = [0, 1]
        athena_metadata_ = {'schema_version': 1, 'descriptors': list(DESCRIPTORS)}

    with pytest.raises(ValueError, match='Train a new model'):
        predict(OldModel(), ['CCO'])
