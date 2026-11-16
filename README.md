# Expert-based student risk monitoring: reference implementation and analysis scripts

Code accompanying the manuscript "Expert-Based Student Risk Monitoring in a School Management Platform:
Concurrent Validation of Multi-Role Ratings against Homeroom Teacher Judgement"
(International Journal of Modern Education and Computer Science, manuscript PAPER024920, under review).

## Contents

| File | Purpose |
|---|---|
| `risk_engine.py` | Reference implementation of the aggregation engine (Algorithm 1), with worked examples. Run `python risk_engine.py` to test it. |
| `final_analysis.py` | Recomputes the priority levels from the raw dimension scores and produces every statistic in Section 4 (`results.json` and CSV tables). |
| `verify_numbers.py` | Compares the numbers printed in the manuscript with `results.json`; stops if any differs. |
| `make_figures.py` | Draws Fig. 1 to Fig. 8 at 300 dpi from `results.json`. |

## How to reproduce

1. Prepare one CSV row per student with the dimension scores `D1` to `D5` (integers 1 to 5, blank if missing),
   the homeroom rating `H`, and the platform records listed in Table 2 of the paper.
2. `python final_analysis.py DATA.csv OUTDIR`
3. `python verify_numbers.py OUTDIR/results.json`
4. `python make_figures.py OUTDIR FIGDIR`

Bootstrap and random-weight analyses use a fixed seed (20261018) and 2,000 resamples.
Software: Python 3.13, pandas, NumPy, SciPy, scikit-learn, Matplotlib.

## Data

The student-level dataset is not included because it contains attendance and household records of children.
A de-identified extract is available from the corresponding authors on reasonable request, subject to the
approval of the ethics committee that approved the study (Khon Kaen University, reference HE683594).
