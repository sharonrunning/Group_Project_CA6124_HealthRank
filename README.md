# HealthRank: A Personalized Health-Aware Meal Recommender

CA6124 Group Project · Group 15 · Wang Danyun, Hao Ruiying, Luo Yurou

> Live demo: https://healthrank-ca6124.streamlit.app/

## Project objective

Food recommenders that only optimise preference tend to reproduce the healthiness of what users already choose: on
MealRec+H, both a popularity baseline and a standard personalised model (BPR-MF) recommend meals slightly *less*
healthy than users' own history. HealthRank recommends complete meals (appetizer + main dish + dessert) that stay
personalised while nudging each user towards healthier options. It is a wellness nudge for existing users, not
dietary or medical advice.

The model, **Health-aware BPR**, adapts BPR-MF (Rendle et al., 2009) by adding a weighted health term to the ranking
score, `s(u, m) = p_u · q_m + λ · health(m)`, and selects λ on the validation set under an accuracy budget.
Main result on the test set (mean of 3 seeds): meal health of the top-10 rises by **10.6%** over BPR-MF
(0.332 → 0.367), recommendations become healthier than users' own history for 1,085 of 1,114 users, and NDCG@10
does not change significantly (0.0355 → 0.0358, p = 0.64).

## System architecture

```
 MealRec+H raw files (data/raw/)
        │  src/data_preprocessing.py   validation checks, unified CSVs, combined health score
        ▼
 data/processed/<dataset>/     train / val / test, meal_health, meal_course, course_info, stats
        │  src/dataset.py  load_data()   one shared loader for every component
        ▼
 ┌──────────────────────────────┐        ┌────────────────────────────────────────────┐
 │ Models  (src/models.py)      │        │ Evaluation  (src/evaluate.py,              │
 │  Popularity                  │──────▶ │   src/health_metrics.py)                   │
 │  BPR-MF (λ = 0)              │        │  run_experiments.py · ablation.py ·        │
 │  Health-aware BPR (λ = 0.15) │        │  case_study.py · visualize_results.py      │
 └──────────────┬───────────────┘        └────────────────────────────────────────────┘
                │ src/train.py → models/checkpoints/health_aware_bpr.pt
                ▼
 src/inference.py  load_recommender() · recommend_for_user()
        │
        ▼
 src/integration.py + app.py  →  Streamlit demo (Meals for you · Model insights)
```

| Component | Files | Owner |
|---|---|---|
| Data pipeline and shared loader | `src/data_preprocessing.py`, `src/dataset.py`, `DATA_FORMAT.md`, `CA6124_memberB_data_exploration.ipynb` | Hao Ruiying (B) |
| Models, training and λ selection, inference | `src/models.py`, `src/train.py`, `src/inference.py`, `CA6124_memberA_models.ipynb`, `models/checkpoints/`, `load_recommender()`, `recommend_for_user(recommender, user_id, k=10)`, `models/checkpoints/health_aware_bpr.pt` | Wang Danyun (A) |
| Evaluation, experiments, ablation, figures | `src/evaluate.py`, `src/health_metrics.py`, `src/run_experiments.py`, `src/ablation.py`, `src/case_study.py`, `src/visualize_results.py` | Luo Yurou (C) |
| Demo application | `src/integration.py`, `app.py`, `.streamlit/config.toml` | Hao Ruiying (B) |

## Installation

Python 3.10 or later. All commands are run from the repository root.

```bash
python -m venv .venv && source .venv/bin/activate      # optional
python -m pip install -r requirements.txt
```

Everything runs on a CPU; a GPU is used automatically if PyTorch finds one.

## Reproducing the report

Every number, table and figure in Sections 5–6 of the report is produced by the steps below (about 10 minutes on a
4-core CPU in total). The repository already contains the processed data, the trained checkpoint and all outputs, so
each step can also be run on its own.

| Step | Command | Time | Output |
|---|---|---|---|
| 1. Prepare data | `python src/data_preprocessing.py` | seconds | `data/processed/MealRec+H/` |
| 2. Tune λ and train the final model | `python src/train.py` | ~2 min | `results/lambda_selection_validation.csv`, `models/checkpoints/health_aware_bpr_retrained.pt` |
| 3. Main experiments (3 models × 3 seeds) | `python src/run_experiments.py` | ~1.5 min | `results/evaluation/main_results.csv`, `significance.csv`, `metrics_by_seed.csv` |
| 4. Ablation and λ sweep | `python src/ablation.py` | ~3.5 min | `results/evaluation/ablation_*.csv` |
| 5. User groups, case study, efficiency | `python src/case_study.py` | ~20 s | `results/evaluation/user_groups.csv`, `case_study.csv`, `efficiency.json` |
| 6. Figures and tables | `python src/visualize_results.py` | seconds | `figures/results_*.png`, `results/evaluation/tables.md`, `main_results.tex` |

Expected key numbers (MealRec+H test set, K = 10, mean of seeds 42 / 123 / 2026):

| Model | Recall@10 | NDCG@10 | AvgHealth@10 | HealthGain@10 |
|---|---|---|---|---|
| Popularity | 0.0227 | 0.0141 | 0.331 | −0.014 |
| BPR-MF | 0.0500 | 0.0355 | 0.332 | −0.013 |
| Health-aware BPR (λ = 0.15) | 0.0510 | 0.0358 | 0.367 | +0.022 |

## Dataset preparation

- Source: **MealRec+** — Li, M., Li, L., Tao, X., & Huang, J. X. (2024). *MealRec+: A Meal Recommendation Dataset with Meal-Course Affiliation for Personalization and Healthiness.* SIGIR '24. https://github.com/WUT-IDEA/MealRecPlus
- We use **MealRec+H** by default (1,575 users, 3,817 meals, 46,767 user–meal interactions, official 8:1:1 split).
- Raw files are already in `data/raw/`. Processed files are already in `data/processed/`.

Regenerate processed data (takes a few seconds; stops with an error if any critical check fails):

```bash
python src/data_preprocessing.py                  # MealRec+H
python src/data_preprocessing.py --dataset MealRec+L
```

Course names for the demo are already extracted to `data/processed/MealRec+H/course_info.csv` (7,280 courses).
To regenerate it, download `MealRec+/MealRec+H/meta_data/course.zip` (~195 MB, Git LFS) from the MealRec+ GitHub
page into `data/raw/MealRec+H/meta_data/` and re-run the script; only 4 small columns are read and the zip is not committed.

File formats and the shared model/evaluation interface: see **[DATA_FORMAT.md](DATA_FORMAT.md)**.

Step-by-step walkthrough with the statistics table and figures for the report: `CA6124_memberB_data_exploration.ipynb`
(outputs: `results/dataset_stats_table.csv`, `figures/dataset_overview.png`, `figures/fsa_vs_who.png`).

Loading the data in Python:

```python
from src.dataset import load_data
data = load_data("MealRec+H")
print(data.n_users, data.n_meals, len(data.train))
```

## Training and tuning the models

Models are defined in `src/models.py` (Member A): `Popularity`, and `BPRMF` / `train_bpr()` for both BPR-MF
(`lambda_health = 0`) and Health-aware BPR. Training settings: 64 factors, 20 epochs, Adam (lr 0.001), batch size
2,048, L2 regularisation 0.0001, one uniformly sampled negative per positive (`results/experiment_config.json`).

```bash
python src/train.py                         # select λ on validation, then train and save the final model
python src/train.py --lambda-health 0.15    # skip the search and train with a given λ
python src/train.py --out models/checkpoints/health_aware_bpr.pt   # overwrite the checkpoint used by the demo
```

**λ selection rule** (validation set, seed 42): the largest λ in {0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2} whose
NDCG@10 is at most 3% below BPR-MF and whose top-10 lists are healthier on average than users' own training meals.
This selects **λ = 0.15** (validation NDCG@10 −2.1%, AvgHealth@10 0.367); λ = 0.2 is rejected (NDCG −3.5%).

`src/train.py` is the command-line version of `CA6124_memberA_models.ipynb` (originally run in Google Colab). It
reproduces the notebook's validation table and the saved checkpoint exactly: retraining with seed 42 gives the same
top-10 lists for all 1,575 users as `models/checkpoints/health_aware_bpr.pt`. By default it writes
`health_aware_bpr_retrained.pt` (not committed) so that the demo's checkpoint is never overwritten by accident.

## Evaluation and experiments

All scripts follow the protocol in [DATA_FORMAT.md](DATA_FORMAT.md), the same as A's final notebook: candidates = the
3,810 meals seen in training minus each user's train meals, ground truth restricted to those candidates, hyper-parameters chosen on `val`, `test` reported once.

```bash
python src/run_experiments.py      # Popularity, BPR-MF, Health-aware BPR x 3 seeds (+ A's demo checkpoint)
python src/ablation.py             # health-component ablation + lambda sweep
python src/case_study.py           # results by user group, one-user case study, efficiency
python src/visualize_results.py    # figures/results_*.png, results/evaluation/tables.md, main_results.tex
python src/evaluate.py --recs results/health_bpr_test_top20.csv --split test   # score any recommendation file
```

| Script | What it does |
|---|---|
| `src/evaluate.py` | Precision / Recall / NDCG / HitRate / MRR @K; reads and writes the agreed `user_id, meal_id, rank, score` files |
| `src/health_metrics.py` | AvgHealth, AvgFSA, AvgWHO, HealthGain (vs. the user's own history), HealthyRatio, HealthyExposure (MealRec+ ranking-exposure gap); Coverage, Novelty, ILD |
| `src/run_experiments.py` | trains every model for each seed with A's `train_bpr`, evaluates val + test, paired Wilcoxon tests |
| `src/ablation.py` | health term removed at training time / at ranking time / both; lambda sweep, retrained vs. post-hoc re-ranking |
| `src/case_study.py` | results by tercile of users' history health and history size, one-user case study, training and ranking time |
| `src/visualize_results.py` | report figures and Markdown / LaTeX tables |

Outputs: `results/evaluation/` (`main_results.csv`, `metrics_by_seed.csv`, `significance.csv`, `ablation_*.csv`,
`user_groups.csv`, `case_study.csv`, `efficiency.json`, `tables.md`), `figures/results_*.png`, and top-20 test lists
`results/<model>_test_top20.csv`. `evaluate.py` reproduces the numbers in A's notebook exactly (seed 42, λ = 0.15:
Recall@10 0.0491, NDCG@10 0.0321, AvgHealth@10 0.3677). Popularity differs slightly (NDCG@10 0.0141 vs. 0.0140)
because ties are broken by the lower meal_id here.

## Running the demonstration

Online: https://healthrank-ca6124.streamlit.app/ . Locally:

```bash
python -m pip install -r requirements.txt
python src/data_preprocessing.py      # only needed if data/processed/ is missing
python -m streamlit run app.py
```

A browser tab opens at http://localhost:8501. The app has two pages in a side navigation; the sidebar also holds the example-user picker (anonymised users from the dataset), the number of meals and a "Try a random user" button.

- **Meals for you** (end-user view): a grid of meal cards (3 per row on desktop, 1 on a phone). Each card shows the main dish, the appetizer and dessert on separate lines, a nutrition-score bar (0–1, explained under *About the nutrition score*) and one plain-language reason. *View meal* opens a details dialog with each dish's rating, review count and FSA/WHO scores, plus all reasons. The trained model weight (λ = 0.15, selected on the validation set) is used; no model numbers are shown.
- **Model insights** (for the teaching team): the user's history size and average healthiness, each recommendation's health_score, FSA, WHO and model score written as preference + health bonus (the model rescales health_score to [0, 1] internally before multiplying by λ), data-grounded reasons, ✅ for meals among the user's held-out test interactions, a history-vs-recommendation health chart and the full table. *Technical settings* lets you include users without test data.

**Model hook-up.** `src/integration.py` loads the Health-aware BPR once at start-up via
`src/inference.py` (`load_recommender()` → `recommend_for_user(recommender, user_id, k)`), using the checkpoint
`models/checkpoints/health_aware_bpr.pt`. Because the model score is `preference + λ × health`, the demo shows each
meal's preference part and health bonus separately. If the model file exists but fails to load (e.g. missing checkpoint
or torch), the page shows the real error; the Popularity placeholder is used only when no model file exists at all.

Course names come from `course_info.csv` (see Dataset preparation); if it is missing, courses show as `Course #id`.

## Repository contents

```
app.py, src/integration.py            Streamlit demo
src/                                  data pipeline, models, training, inference, evaluation
data/raw/, data/processed/            original MealRec+ files and unified CSVs
models/checkpoints/                   trained Health-aware BPR used by the demo
.ipynb                                data exploration (B) and model development (A) notebooks
results/, figures/                    experiment outputs and report figures
DATA_FORMAT.md                        shared data format, model output format and evaluation protocol
requirements.txt                      Python dependencies
```
