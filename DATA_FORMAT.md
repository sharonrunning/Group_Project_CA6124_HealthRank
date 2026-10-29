# Data Format (agreed interface for A / B / C)

Dataset: **MealRec+H** (Li et al., SIGIR '24, https://github.com/WUT-IDEA/MealRecPlus).
Regenerate: `python src/data_preprocessing.py` → `data/processed/MealRec+H/`

All IDs are **0-based indices** from the official dataset (not raw Allrecipes IDs).
`meal_id` = a meal (bundle of 3 courses); `course_id` = a single dish. Do not mix them.

## Files

| File | Columns | Notes |
|---|---|---|
| `train.csv` / `val.csv` / `test.csv` | `user_id, meal_id` | Official 8:1:1 split (`val` = official "tune"). 37,413 / 4,677 / 4,677 rows |
| `meal_health.csv` | `meal_id, fsa, who, fsa_norm, who_norm, health_score` | **`health_score` ∈ [0,1], higher = healthier** (see below). Raw `fsa`: lower = healthier; raw `who`: higher = healthier |
| `user_health_train.csv` | `user_id, user_mean_fsa, user_mean_who, user_mean_health_score, n_train` | Mean health of each user's **train** meals only (no test leakage) |
| `meal_course.csv` | `meal_id, course_id, category_id, category` | Every meal = appetizer + main_dish + dessert |
| `course_health.csv` | `course_id, fsa, who` | Course-level health |
| `stats.json` | — | Dataset statistics for the report |

## health_score (project-defined — A and C please confirm)

```
health_score = 0.5 * (1 - minmax(FSA)) + 0.5 * minmax(WHO)
```

- This combined score is **our own definition**, not part of MealRec+. FSA is flipped so both parts point the same way (higher = healthier); min-max is over all 3,817 meals.
- The raw `fsa` and `who` columns are kept, so C can also report FSA and WHO separately (recommended, since those are the dataset's official metrics).
- Weights live in `W_FSA, W_WHO` at the top of `src/data_preprocessing.py`.

## Loading (A and C should use this, so everyone reads the same data)

```python
from src.dataset import load_data
data = load_data("MealRec+H")
data.train_matrix      # CSR (1575 x 3817)
data.train_items[u]    # meals to EXCLUDE when ranking for user u
data.test_items[u]     # ground truth for user u (1,114 test users)
data.health            # health_score per meal_id (higher = healthier)
```

## Model output (A → B and C)

Member A's inference interface (`src/inference.py`):

```python
from src.inference import load_recommender, recommend_for_user
recommender = load_recommender()                          # call ONCE (loads models/checkpoints/health_aware_bpr.pt)
df = recommend_for_user(recommender, user_id=25, k=10)    # DataFrame: meal_id, score, health_score, fsa, who
```

Seen train meals and meals absent from train are excluded. The demo (`app.py` via `src/integration.py`) uses this interface.

Batch results for evaluation: CSV `user_id, meal_id, rank, score` (rank starts at 1), saved to `results/<model>_test_top<K>.csv`.

## Evaluation protocol (C)

- Candidates = all meals that appear in `train.csv` (3,810 of 3,817), minus each user's own train meals.
- Ground truth = the user's `test.csv` meals that are candidates; test users left with none are not scored.
- Hyper-parameters (incl. health weight) tuned on `val`; `test` used once for final numbers.
- Report K ∈ {10, 20}; accuracy (Precision/Recall/NDCG@K) + mean `health_score`@K.
