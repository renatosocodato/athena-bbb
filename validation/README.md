# B3DB empirical software demonstration

The benchmark protocol was fixed before computing model outcomes. The dataset is public, and its upstream README and LICENSE designate its materials/data as CC0-1.0. No raw dataset or fitted benchmark model is redistributed here. The optional runner downloads two pinned upstream files and checks their SHA-256 before use.

```sh
python -m pip install .
python validation/run_b3db_benchmark.py --output new_b3db_run
```

The output directory includes downloaded public data, the explicit development split and a JSON result. Do not use this command as a recipe to search seeds or remove hard test molecules. Runtime is CPU-only. The fixed benchmark is not part of the ordinary no-network unit tests.

Sources:

- [B3DB repository, fixed tree 75dab1cc607bb7a03f3de5c576cffd223e767844](https://github.com/theochem/B3DB/tree/75dab1cc607bb7a03f3de5c576cffd223e767844)
- [Upstream CC0 licence](https://github.com/theochem/B3DB/blob/75dab1cc607bb7a03f3de5c576cffd223e767844/LICENSE)
- Meng, F., Xi, Y., Huang, J. & Ayers, P. W. A curated diverse molecular database of blood-brain barrier permeability with chemical descriptors. *Scientific Data* 8, 289 (2021). [doi:10.1038/s41597-021-01069-5](https://doi.org/10.1038/s41597-021-01069-5).

## Observed on 8 November 2026

The 0.1.0rc4 rerun used the same pinned TSVs and frozen analysis protocol with atom maps removed before identity checks. The development split and retained external records matched the original files byte for byte. Curation counts, exclusions, metrics and bootstrap intervals all matched the stored aggregate.

The development file had 7,807 rows; two invalid molecules were excluded by the predeclared rule, leaving 7,805. One fixed scaffold split produced 5,691 training and 2,114 holdout molecules. The holdout contains 610 groups, 1,268 BBB+ and 846 BBB- labels.

| Model | Holdout balanced accuracy | Holdout ROC AUC | Holdout Brier score |
|---|---:|---:|---:|
| ATHENA random forest | 0.754 (0.708–0.804) | 0.850 (0.801–0.899) | 0.151 (0.125–0.171) |
| Scaled logistic baseline | 0.719 (0.659–0.786) | 0.794 (0.728–0.856) | 0.176 (0.148–0.199) |
| Training-prior baseline | 0.500 | 0.500 | 0.242 (0.226–0.263) |

Parentheses show 95% percentile intervals from 2,000 shared scaffold-cluster bootstrap draws conditional on the fixed fitted models and observed test scaffolds. These intervals do not include model-training uncertainty or differences between assays, species and source cohorts. Acyclic analogue similarity is not eliminated by the canonical-group fallback.

The paired random-forest-minus-logistic difference in balanced accuracy is 0.035 (0.001–0.076); the ROC-AUC difference is 0.056 (0.022–0.106). These are observed estimates for this split, not evidence of universal superiority or novelty of the random-forest method. Leave-one-test-scaffold-out balanced accuracy for the forest ranges from 0.743 to 0.776.

## External boundary, retained rather than optimized away

The upstream external file contains 175 molecules. The strict overlap rule removed 133 molecules sharing a canonical structure or group with the complete development corpus, including its internal holdout. **Only 42 molecules remain: 41 BBB+ and one BBB-.**

The forest's descriptive external balanced accuracy is 0.902, ROC AUC 0.951 and Brier score 0.104; logistic balanced accuracy is 0.915, ROC AUC 0.902 and Brier score 0.134. The paired intervals do not resolve a clear external advantage over logistic. More importantly, there is only one negative example. Of 2,000 scaffold resamples, 725 omit one of the two classes. Intervals that retain both classes repeatedly reuse that same negative molecule and do not estimate uncertainty across a population of new negative compounds. Removing its scaffold makes balanced accuracy and AUC unavailable. This is a boundary on external evidence, not a reason to weaken the overlap filter.

The complete numerical record, including class counts, bootstrap valid-draw counts and sensitivity outputs, is in `B3DB_RESULT_20261108.json`. All 42 retained external molecules fell inside the fitted per-descriptor ranges; that coarse range flag does not turn this small subset into a validated applicability domain.

Two interpretations remain compatible with the observations: the descriptors can discriminate these historical labels beyond a constant prior, while the apparent external result can also reflect an unusually selected/easy subset and the single available negative. A more fundamental alternative is that pooled BBB labels partly encode source/assay definitions rather than a universal binary permeability state. An independently reserved, source/assay-disjoint set with adequate negative representation would distinguish these readings more effectively. That next evaluation was not conducted here.

This demonstration verifies a documented empirical workflow and reports its performance and limits. It does not establish calibrated human BBB probabilities, clinical utility, therapeutic effects or prospective validation on data that did not exist when the software was developed.

## Retrospective cross-corpus test on NeuTox 2.0

The fixed ATHENA model and baselines were fitted on the original B3DB
development partition. A separate benchmark combined the three pinned
NeuTox BBB SMILES files in one corpus, removed invalid entries, contradictory
labels and canonical duplicates, then excluded structures or scaffolds
overlapping either B3DB file. The audit also excluded matches in standardized
parent connectivity and parent ring scaffolds. Remaining contradictory
parent groups were removed and one representative was retained per parent.

Of 8,239 canonical molecules after initial curation, 8,130 shared at least
one exclusion key with B3DB. The stricter final panel contained **96 molecules
across 81 groups**, including **49 negatives across 36 groups**. The random
forest achieved ROC AUC 0.839 (0.721–0.923) and balanced accuracy 0.717
(0.606–0.808). The logistic baseline gave 0.769 and 0.678. All 2,000
scaffold-cluster resamples contained both classes. Intervals condition on
these observed groups and the fixed fitted models.

This panel adds negative-group support relative to the original B3DB
external subset. The source files lack per-row experimental provenance.
Shared assay or study origin can therefore remain even after the chemical
filters. The result is retrospective and uses the source's original binary
labels.

With the package installed, create a writable B3DB input directory and run
`python validation/neutox_audit.py --b3db INPUT_DIR --output NEW_OUTPUT`.
The runner fetches and verifies the pinned input files as needed. The
package includes aggregate results in `NEUTOX_RESULTS.json`.

Sources: [NeuTox 2.0 paper](https://doi.org/10.1016/j.envint.2024.109244) and
[pinned source files](https://github.com/xuejunhe/NeuTox-2.0/tree/c35e6366391f74605b625fdcfdd126b7f1ac3a3f/BBB/random0/smiles).
