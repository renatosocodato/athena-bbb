# Input and model contracts

`run` accepts CSV columns `smiles,label`. Labels must be binary 0/1. ATHENA removes SMILES atom map annotations before computing canonical structure, scaffold and descriptors with RDKit. For example, `CCO` and `[CH3:1][CH2:2][OH:3]` identify the same molecule. Canonical identity retains isotopes and stereochemistry and uses the supplied salt and tautomer representation. Empty or invalid molecules and conflicting labels for canonical duplicates are rejected. Canonical duplicates with the same label are collapsed before splitting.

One `GroupShuffleSplit` uses Murcko ring scaffolds, with canonical molecules as groups for acyclic structures. Acyclic analogues can therefore occupy different partitions: this split prevents exact-structure leakage and shared ring scaffolds, not all similarity leakage. Train and test must both contain both classes; the program refuses inadequate fixed partitions rather than searching seeds to improve scores. Freeze seed, partition fraction, chemistry preprocessing and intended benchmark before inspecting outcomes.

The model uses six ordered RDKit descriptors: MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds. The returned RandomForest estimator is fit only on the training partition. It carries schema, software versions, complete development corpus/split and training descriptor ranges. `model_metadata.json` exposes those fields alongside `model.joblib`. Prediction rejects mismatched recorded library versions or unsupported descriptor schemas. Use the recorded environment to load a model reproducibly. Schema 2 records atom map independent identities. Train a new model from the development data to replace a schema 1 model because its original split may contain the same molecule in both partitions.

`predict` accepts CSV column `smiles`. It emits `class_1_score` and a flag for molecules outside the per-descriptor ranges seen in training. Class scores are not calibrated clinical probabilities. Range flags are coarse diagnostics and do not establish that an unflagged molecule is within a validated applicability domain.

`evaluate` accepts an independently reserved benchmark CSV `smiles,label`. Both classes are required. Evaluation uses the same atom map independent identity rule as training. Canonical structures and scaffold groups must be disjoint from the complete recorded development corpus, including its internal holdout. Supplying an unrelated training table to the Python API cannot bypass this check: a supplied table must match the model's recorded corpus. No refitting takes place. Balanced accuracy, ROC AUC and Brier score describe this benchmark and have no automatic human BBB or clinical meaning. The ordinary evaluate command supplies point estimates. The optional B3DB validation runner additionally supplies conditional scaffold-bootstrap intervals; these are not calibrated clinical probability intervals.

The demonstration labels alternate arbitrarily across a synthetic chemistry list. Its scores only demonstrate execution; they are not a BBB benchmark. Real dataset labels, assay definitions, species, exposure, rights and external independence must be documented separately.

Joblib uses pickle. Only load model files whose provenance you trust. Never publish model metadata or splits from private chemistry unless those data are intended for disclosure: by design the model stores its development molecules for leakage checking.

External evaluation reports molecule and group counts by class alongside
its metrics. Check the number of independent groups contributing to each
class when interpreting a small or imbalanced panel.
