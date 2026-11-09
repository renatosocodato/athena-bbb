# 0.1.0rc5

- Report class and scaffold support for external evaluation. Add a pinned retrospective cross-corpus test with NeuTox.

# 0.1.0rc4

- Remove atom-map annotations before chemical identity and overlap checks. Model schema 2 requires retraining; frozen B3DB results reproduced. Exclude the historical benchmark workflow.

# 0.1.0rc4

- Unified molecule identity across curation, training and external evaluation by removing SMILES atom maps before canonicalization and scaffold extraction. Schema 2 requires retraining older models because their split may contain mapped and unmapped forms of the same molecule.

# 0.1.0rc3

- Added prediction and external evaluation CLI commands using explicit CSV files.
- Model serialization now carries ordered descriptor/schema, runtime versions and the complete development corpus/split. Prediction rejects incompatible models and runtime versions.
- External evaluation checks the recorded corpus, preventing an unrelated supplied training table from hiding overlap; both structures and scaffold groups are checked.
- Added coarse training descriptor-range flags, clearly separated from validated applicability-domain claims.
- Rejected empty/nonstring molecules and empty descriptor batches; added roundtrip, overlap, schema and failure-output tests.
- Failed analysis inputs no longer leave partial result directories; narrowed optional legacy dependencies.

# 0.1.0rc1

Initial review candidate: portable API, explicit inputs, synthetic example and selected original workflow. No private datasets redistributed.

## 0.1.0rc3

Added reproducible public B3DB demonstration with pinned inputs, fixed scaffold holdout, prior/logistic baselines, paired scaffold-bootstrap intervals and explicit external evidence limits. Added no-download curation tests. No model tuning or weakening of overlap checks. No raw data or fitted benchmark model redistributed.
