"""Train, predict and evaluate explicit chemical inputs without remote services."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import joblib
import pandas as pd
from . import core



def main(argv=None):
    parser = argparse.ArgumentParser(description='Chemical scaffold-separated BBB classification')
    sub = parser.add_subparsers(dest='command', required=True)
    example = sub.add_parser('example', help='Synthetic exercise; its labels have no pharmacological meaning')
    example.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run', help='Train once with a scaffold holdout')
    run.add_argument('--input', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seed', type=int, default=2026)
    run.add_argument('--test-size', type=float, default=.25)
    for name in ['predict', 'evaluate']:
        command = sub.add_parser(name, help='Use a trusted model.joblib produced by this package')
        command.add_argument('--model', type=Path, required=True)
        command.add_argument('--input', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        parser.error('Output directory already exists; choose a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.athena-bbb-', dir=output.parent) as staging:
            args.output = Path(staging)
            result = analyze(args)
            (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            # Exclusive mkdir refuses even an empty directory created concurrently.
            output.mkdir(exist_ok=False)
            for item in args.output.iterdir():
                shutil.move(str(item), output / item.name)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f'Input error: {exc}\n')
    return 0


def analyze(args):
    if args.command in ['predict', 'evaluate']:
        # joblib uses pickle: the caller must trust the provenance of this file.
        model = joblib.load(args.model)
        frame = pd.read_csv(args.input)
        if args.command == 'evaluate':
            return {'data_kind': 'user_supplied_external', **core.evaluate_external(model, None, frame)}
        if 'smiles' not in frame:
            raise ValueError('Column smiles required')
        result = pd.DataFrame({'smiles': frame.smiles, 'class_1_score': core.predict(model, frame.smiles),
                               'outside_training_descriptor_range': core.applicability(model, frame.smiles)})
        result.to_csv(args.output / 'predictions.csv', index=False)
        return {'data_kind': 'user_supplied', 'n_predictions': len(result),
                'interpretation': 'Class scores are not calibrated human BBB probabilities; range flags are coarse applicability diagnostics'}
    if args.command == 'example':
        smiles = ['C','CC','CCC','CCCC','CCCCC','CCCCCC','CO','CCO','CCCO','CCCCO','CN','CCN','CCCN','CCCCN','CCl','CCCl','CCCCl','CCCCCl','CF','CCF','CCCF','CCCCF','CBr','CCBr','CCCBr','CCCCBr','C=O','CC=O','CCC=O','CCCC=O','C#N','CC#N','CCC#N','CCCC#N','CS','CCS','CCCS','CCCCS','CC(=O)O','CCC(=O)O']
        frame = pd.DataFrame({'smiles': smiles, 'label': [i % 2 for i in range(len(smiles))]})
    else:
        frame = pd.read_csv(args.input)
    model, metrics, assignment = core.train(frame, seed=getattr(args, 'seed', 2026), test_size=getattr(args, 'test_size', .25))
    joblib.dump(model, args.output / 'model.joblib')
    assignment.to_csv(args.output / 'split.csv', index=False)
    (args.output / 'model_metadata.json').write_text(json.dumps(core.model_metadata(model), indent=2) + '\n')
    return {'data_kind': 'synthetic' if args.command == 'example' else 'user_supplied', **metrics,
            'model_format': 'joblib; load only trusted models', 'synthetic_label_warning': args.command == 'example'}
