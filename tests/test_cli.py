import subprocess
import sys
from pathlib import Path
import pytest

def test_example_and_no_overwrite(tmp_path):
    out=tmp_path/'example'
    command=[sys.executable,'-m','athena_bbb','example','--output',str(out)]
    subprocess.run(command,check=True,capture_output=True)
    assert (out/'result.json').is_file()
    second=subprocess.run(command,capture_output=True)
    assert second.returncode!=0


def test_predict_and_external_cli_roundtrip(tmp_path):
    import json
    import pandas as pd
    base=[sys.executable,'-m','athena_bbb']
    trained=tmp_path/'trained'
    subprocess.run([*base,'example','--output',str(trained)],check=True,capture_output=True)
    inputs=tmp_path/'predict.csv';pd.DataFrame({'smiles':['CCO','c1ccccc1']}).to_csv(inputs,index=False)
    output=tmp_path/'predictions'
    subprocess.run([*base,'predict','--model',str(trained/'model.joblib'),'--input',str(inputs),'--output',str(output)],check=True,capture_output=True)
    assert len(pd.read_csv(output/'predictions.csv'))==2
    external=tmp_path/'benchmark.csv';pd.DataFrame({'smiles':['c1ccccc1','C1CCCCC1'],'label':[0,1]}).to_csv(external,index=False)
    evaluated=tmp_path/'evaluation'
    subprocess.run([*base,'evaluate','--model',str(trained/'model.joblib'),'--input',str(external),'--output',str(evaluated)],check=True,capture_output=True)
    assert json.loads((evaluated/'result.json').read_text())['n_external']==2

    external.write_text('smiles,label\n[CH3:1][CH2:2][OH:3],0\nC1CCCCC1,1\n')
    rejected = tmp_path / 'overlap'
    result = subprocess.run([*base, 'evaluate', '--model', str(trained / 'model.joblib'),
                             '--input', str(external), '--output', str(rejected)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'overlaps recorded development' in result.stderr
    assert not rejected.exists()


def test_invalid_input_does_not_leave_partial_output(tmp_path):
    inp=tmp_path/'bad.csv';inp.write_text('smiles,label\n,1\n')
    output=tmp_path/'failed'
    result=subprocess.run([sys.executable,'-m','athena_bbb','run','--input',str(inp),'--output',str(output)],capture_output=True)
    assert result.returncode!=0 and not output.exists()
