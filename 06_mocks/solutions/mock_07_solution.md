# Mock 07 — Solar upsell classification: solution

Story: predict `has_solar` (11% of meters) from attributes and the 2023 daily consumption
pattern, then hand sales a "top 20 meters to target" list. The notebook reports an accuracy
and an "AUC" and ships the list. Almost every number in the Results cell is wrong or
meaningless.

Note on reproducibility: the mock omits `random_state` in `train_test_split`, so **your
numbers will differ from the ones quoted here on every run**. That is planted problem 5.
The executed run in the repo shows accuracy 0.811, "AUC" 0.827, "AUC (holdout)" 0.500, while the
Results text still says "~93% accuracy": a hardcoded claim from an earlier run that no longer matches
the number printed two lines above it. Noticing that mismatch is itself an attention-to-detail point.

## Planted problems

| # | Where | Kind | Why it matters | How to detect | Fix | Interviewer follow-up |
|---|---|---|---|---|---|---|
| 1 | Results: "identifies solar meters with ~93% accuracy" | suspicious result | Positives are 11%. Predicting "no" for everyone gives 0.889. Accuracy tells you nothing here; in the executed run the model is *below* the trivial baseline (0.811). | `meters.has_solar.mean()`; `DummyClassifier(strategy="most_frequent")` | Compare to baseline; use AUC / precision-recall / `classification_report` | "What accuracy would a model that never predicts solar get?" |
| 2 | Attributes: `df["region_solar_rate"] = df.groupby("region")["has_solar"].transform("mean")` | bug (target leakage) | The feature is the target mean by region computed on **all** rows including the test set. Each test row's own label is inside its feature. With 5 regions the leak is mild, but the pattern is the classic target-encoding leak. | Ask "which columns were computed using `has_solar`?"; check correlation of the feature with the target inside the test set | Drop it, or compute per training fold only (e.g. `TargetEncoder` inside a pipeline) | "If a region had one meter, what would this feature equal?" |
| 3 | Consumption features: `.sort_values(...).drop_duplicates(["meter_id", "season"])` **before** `pivot` | bug | Dedup keeps the *first day* of each season per meter, so the "summer/winter ratio" is the ratio of two single daily readings (noise). The head shows ratios like 1.007 and 0.36 for ordinary meters. | Inspect the intermediate `rd` shape (drops from ~107k to 900 rows); look at `seasonal.head()` spread; compare `seasonal.groupby(has_solar).sw_ratio.describe()` | `groupby(["meter_id","season"]).kwh.mean().unstack()` | "How many rows should each meter contribute to a seasonal mean?" |
| 4 | Attributes: `OneHotEncoder().fit_transform(df[...])` on the full frame, then `StandardScaler` inside the pipeline | research choice | Encoder is fitted on all data outside the CV loop; with new categories in production the code crashes (`handle_unknown` not set). Splitting preprocessing between "outside" and "inside" the pipeline is how leaks creep in. | Notice `enc.fit_transform` runs before `train_test_split` | `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` inside the `Pipeline` | "What happens when a new tariff appears next month?" |
| 5 | Train/test: `train_test_split(X, y, test_size=0.3)` | research choice | No `stratify=y`: with 34 positives the test set may hold 5 or 15 of them. No `random_state`: every rerun gives different headline numbers, so the reported result cannot be reproduced. | Rerun the cell; count `y_test.sum()` | `stratify=y, random_state=0`; better, `StratifiedKFold` + `cross_val_predict` | "Run it again. Do you get the same number?" |
| 6 | Model: `accuracy_score(y_test, pipe.predict(X_test))` at the default 0.5 threshold | research choice | With an 11% positive rate and weak features, the model almost never crosses 0.5 for the positive class, so recall for "solar" is ~0 and the accuracy is just the base rate. | `classification_report`; `confusion_matrix`; `pipe.predict(X_test).sum()` | Report precision/recall; pick threshold from the PR curve to match sales capacity; `class_weight="balanced"` | "How many of the 20 you would call actually have solar?" |
| 7a | `auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])` printed as "AUC" and reused in Results | suspicious result | It is the **training** AUC. The name hides it and the Results cell reports it as the model's performance. | Read the cell: `y_train`, `X_train` | Evaluate on the holdout / out-of-fold | "Which dataset did that AUC come from?" |
| 7b | `roc_auc_score(y_test, pipe.predict(X_test))` printed as "AUC (holdout)" | bug | Hard 0/1 labels passed instead of probabilities. AUC from labels is just a one-point ROC; here 0.500 because no positive was predicted. The author ignored the 0.500. | The value 0.500 exactly is a red flag; check the second argument | `pipe.predict_proba(X_test)[:, 1]` | "Why does AUC on labels give exactly 0.5 here?" |
| 8 | Who to target: `pipe.predict_proba(X)[:, 0]` | bug | Column 0 is P(class 0) = P(no solar). The "top 20" is the 20 meters **least** likely to have solar. Confirm: the list is all large-consumption SMEs and only 1 of 20 is flagged solar. | Check `pipe.classes_`; sanity check the top list against `has_solar` | `[:, 1]`, or `[:, list(pipe.classes_).index(1)]` | "Which column corresponds to which class?" |
| 8b | Same cell: probabilities computed on `X` (all rows, incl. training) | research choice | Training rows get optimistic scores and are mixed with test rows in one ranking. | — | Rank on out-of-fold probabilities (`cross_val_predict`) | "Is the ranking of a training row comparable with a test row?" |
| 9 | `df["tariff"].fillna(df["tariff"].mode()[0])` | research choice (minor leak) | Mode is computed on the full sample before splitting. Harmless here, but the habit is wrong. | Runs before the split | `SimpleImputer(strategy="most_frequent")` inside the pipeline | "Where should imputation statistics come from?" |
| 10 | `fillna(0)` then `np.log(x + 1)` for `annual_kwh_estimate` | bug | Six missing meters become `log(1) = 0` while real values are 7.5–10.5: an artificial cluster far outside the data, which the scaler then treats as 5+ sigma outliers. | `describe()` shows min 0 in `annual_kwh_estimate` and `log_kwh`; `X.log_kwh.value_counts().head()` | Leave NaN, impute median inside the pipeline (or add a missing-indicator) | "What does the model learn from a value of exactly 0?" |
| 11 | One-hot columns `region_london`, `region_wales`, ... alongside `region_London` | bug (data cleaning) | Six lowercased rows become four extra dummy columns with 1–3 rows each: pure noise and a sign the categories were never inspected. | `meters.region.value_counts()`; the printed column list has 9 region columns for 5 regions | `str.title()` (or `.str.strip().str.lower()`) before encoding | "How many regions are there?" |
| 12 | No `class_weight`, no threshold tuning anywhere | research choice | For a rare class the default loss and threshold optimise the wrong thing. | — | `class_weight="balanced"`; choose threshold from PR curve | "Sales can call 30 meters. Which threshold do you use?" |

## Honest result (from `mock_07_fixed.ipynb`, `random_state=0`)

| | Broken notebook (one run) | Fixed notebook |
|---|---|---|
| Baseline accuracy (always "no") | not computed | 0.889 |
| Holdout accuracy | 0.811 (below baseline) | 1.000 |
| Holdout AUC from probabilities | not computed (labels gave 0.500) | 1.000 |
| Out-of-fold AUC, all 300 meters | — | 0.994 |
| Solar meters in the top-20 list | 1 of 20 | 20 of 20 |
| Summer/winter ratio, solar vs non-solar | noise (single-day ratio) | 0.41 vs 0.52 with almost no overlap |

The fixed model is *too* good, and that is worth saying aloud too: an AUC of 1.0 should
make you ask whether the seasonal ratio is essentially how `has_solar` was defined. In this
synthetic data it is (the generator subtracts solar output from summer consumption). In a
real dataset you would check whether the feature could only be computed *because* the
label was known, and you would exclude already-flagged meters from the campaign list.

## Scoring

Finding 1, 3, 7a/7b, 8 and 11 is a solid pass (they are visible from the outputs alone).
Finding 2, 5, 6, 10 as well means you read the code rather than the numbers. 4, 8b, 9, 12
are bonus methodology points.
