# Independent implementation and scientific checks

A frozen-protocol public B3DB benchmark used a scaffold-separated internal holdout of 2,114 molecules: RF AUC 0.850 (scaffold bootstrap 0.801–0.899), balanced accuracy 0.754 (0.708–0.804). Strict external-overlap filtering retained only 42 molecules, including one negative. External class imbalance makes the apparent external score fragile; bootstrap intervals conditional on both classes do not resolve that limitation. Baselines, exclusions and class counts are retained in validation/.

This is a benchmark of the supplied representation and fitting procedure, not robust external BBB validation, clinical calibration or evidence of human CNS exposure. Python >=3.11 and Pillow >=13 avoid the vulnerable image dependency selected by RDKit on Python 3.10.

These are internal computational checks against independent implementations, public inputs and real engines. No external peer review is claimed. Full execution details are in VALIDATION.md.

## Revision after independent candidate review

Remove atom-map annotations before chemical identity and overlap checks. Model schema 2 requires retraining; frozen B3DB results reproduced. Exclude the historical benchmark workflow. Hosted execution records for the exact distributions are tracked separately.
