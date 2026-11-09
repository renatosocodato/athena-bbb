# athena-bbb

Chemical scaffold-separated binary classification for BBB-oriented research datasets. Author: Renato Socodato. Version 0.1.0rc4, prepared for review.

## Install and exercise

```sh
python -m pip install '.[test]'
athena-bbb example --output synthetic_training
python -m pytest tests -q
```

## Explicit data, prediction and external evaluation

```sh
athena-bbb run --input training.csv --seed 2026 --test-size 0.25 --output trained
athena-bbb predict --model trained/model.joblib --input molecules.csv --output predictions
athena-bbb evaluate --model trained/model.joblib --input reserved_benchmark.csv --output external_evaluation
```

Output directories must be new. Inputs are validated in a temporary staging directory before committing outputs, so invalid analysis inputs do not leave a partial result directory. No network, account, private directory or token is required by these commands.

The Python API is in `athena_bbb.core`: `prepare`, `features`, `train`, `predict`, `applicability`, `evaluate_external`, and `model_metadata`. Training and evaluation discard SMILES atom maps before computing canonical structure and scaffold identity. The model stores the full development corpus, split, schema and versions. Inference checks that representation and external evaluation rejects chemical overlap. Train a new model with this version if you have a schema 1 model. Read INPUTS.md for chemical independence, score interpretation, applicability and model-file security.


## Scientific boundary

The synthetic example establishes executable behavior only. It does not validate BBB performance. The returned estimator uses its fixed training partition and is not refit on the internal holdout. The external benchmark is reserved separately and never used to refit. Brier score is a diagnostic, not a calibration procedure; class scores do not establish human CNS exposure, therapeutic efficacy or clinical utility. This version reports point estimates and coarse descriptor-range diagnostics. Real assay harmonization and independent performance uncertainty remain separate scientific tasks.

## Empirical demonstration

An optional, frozen-protocol B3DB benchmark with public CC0 data compares the fitted classifier to training-prior and logistic baselines. Its scaffold holdout has 2,114 molecules; strict filtering leaves only 42 external molecules, including one negative, so this is not robust external BBB validation. See [protocol, results and limits](validation/README.md). The ordinary CLI example remains deliberately synthetic.

Python 3.11 or newer is required. Pillow >=13 is explicit because the RDKit dependency otherwise selects a version with known image-decoder vulnerabilities on Python 3.10. No clinical safety claim follows from dependency screening.

## Scope of this release

The package distributes the documented public API and command-line workflows. The historical grant workflows, project data and unvalidated experimental stages are outside this release.
