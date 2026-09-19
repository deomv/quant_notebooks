"""Exercises for 04_sklearn/02_preprocessing_pipelines.ipynb"""
import numpy as np
import pandas as pd

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
"""

QUESTIONS = [
    dict(
        prompt="Fit a `StandardScaler` on `X_train` only. Assign its learned mean (`mean_`, array) to `mean` and its scale (`scale_`, array) to `scale`.",
        setup="X_train = np.array([[1.0], [2.0], [3.0]])\nX_test = np.array([[4.0], [5.0]])",
        solution="scaler = StandardScaler().fit(X_train)\nmean = scaler.mean_\nscale = scaler.scale_",
        hint="StandardScaler().fit(X_train); the attributes end with an underscore.",
        answer_var=["mean", "scale"],
        tol=1e-6,
    ),
    dict(
        prompt="Transform `X_test` with a scaler fitted on **`X_train` only** (not on train+test). Assign the transformed array to `answer`.",
        setup="X_train = np.array([[1.0], [2.0], [3.0]])\nX_test = np.array([[4.0], [5.0]])",
        solution="scaler = StandardScaler().fit(X_train)\nanswer = scaler.transform(X_test)",
        hint="fit on X_train, then .transform(X_test). fit_transform on the concatenation would leak.",
        tol=1e-6,
    ),
    dict(
        prompt="Compute the same transformed `X_test` **by hand**: (x − mean of X_train) / population std of X_train (ddof=0). Assign the array to `answer`.",
        setup="X_train = np.array([[1.0], [2.0], [3.0]])\nX_test = np.array([[4.0], [5.0]])",
        solution="answer = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)",
        hint="np.std defaults to ddof=0, which is what StandardScaler uses.",
        tol=1e-6,
    ),
    dict(
        prompt="Build a `Pipeline` with steps named 'scale' (StandardScaler) and 'ridge' (Ridge(alpha=1.0)), fit it on `X`, `y`, and assign the ridge coefficient array (from `named_steps`) to `answer`.",
        setup="X = pd.DataFrame({\"x1\": [1, 2, 3, 4, 5, 6], \"x2\": [6, 5, 4, 3, 2, 1]})\ny = pd.Series([3, 5, 7, 9, 11, 13])",
        solution="pipe = Pipeline([(\"scale\", StandardScaler()), (\"ridge\", Ridge(alpha=1.0))]).fit(X, y)\nanswer = pipe.named_steps[\"ridge\"].coef_",
        hint="Pipeline([('scale', ...), ('ridge', ...)]); after fit, pipe.named_steps['ridge'].coef_.",
        tol=1e-6,
    ),
    dict(
        prompt="Use a `ColumnTransformer` that scales `kwh` with StandardScaler and one-hot encodes `region` with `OneHotEncoder()` (default settings). Fit it on `X` and assign the transformed array's **shape** (tuple) to `answer`.",
        setup="X = pd.DataFrame({\"kwh\": [10.0, 20.0, 30.0, 40.0], \"region\": [\"N\", \"S\", \"N\", \"E\"]})",
        solution="ct = ColumnTransformer([(\"num\", StandardScaler(), [\"kwh\"]), (\"cat\", OneHotEncoder(), [\"region\"])])\nanswer = tuple(ct.fit_transform(X).shape)",
        hint="ColumnTransformer([('num', StandardScaler(), ['kwh']), ('cat', OneHotEncoder(), ['region'])]). Three regions → 3 one-hot columns + 1 scaled = 4.",
        check_fn=lambda got, exp: (tuple(got) == tuple(exp), "" if tuple(got) == tuple(exp) else f"expected shape {tuple(exp)}"),
    ),
    dict(
        prompt="Same ColumnTransformer as before, fitted on `X`. Assign the list of output feature names (`get_feature_names_out()`) to `answer`.",
        setup="X = pd.DataFrame({\"kwh\": [10.0, 20.0, 30.0, 40.0], \"region\": [\"N\", \"S\", \"N\", \"E\"]})",
        solution="ct = ColumnTransformer([(\"num\", StandardScaler(), [\"kwh\"]), (\"cat\", OneHotEncoder(), [\"region\"])]).fit(X)\nanswer = list(ct.get_feature_names_out())",
        hint="After fit, ct.get_feature_names_out() gives names like 'num__kwh' and 'cat__region_E'.",
    ),
    dict(
        prompt="`X_new` contains a region ('W') never seen in `X`. Fit a `OneHotEncoder` on `X[['region']]` so that unseen categories become all-zeros instead of raising, and assign the transformed dense array for `X_new` to `answer`.",
        setup="X = pd.DataFrame({\"region\": [\"N\", \"S\", \"N\", \"E\"]})\nX_new = pd.DataFrame({\"region\": [\"S\", \"W\"]})",
        solution="enc = OneHotEncoder(handle_unknown=\"ignore\", sparse_output=False).fit(X[[\"region\"]])\nanswer = enc.transform(X_new[[\"region\"]])",
        hint="OneHotEncoder(handle_unknown='ignore', sparse_output=False).",
        tol=1e-9,
    ),
    dict(
        prompt="Fill the missing value in `X` with the **median** using `SimpleImputer`, and assign the imputed array to `answer`.",
        setup="X = np.array([[1.0], [np.nan], [3.0], [10.0]])",
        solution="answer = SimpleImputer(strategy=\"median\").fit_transform(X)",
        hint="SimpleImputer(strategy='median').fit_transform(X). Median of 1, 3, 10 is 3.",
        tol=1e-9,
    ),
    dict(
        prompt="Fit `LogisticRegression` on `X`, `y` and assign the predicted probability of class 1 for `X_new` (a 1-D array) to `answer`. Remember which column of `predict_proba` is class 1.",
        setup="X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])\ny = np.array([0, 0, 0, 1, 1, 1])\nX_new = np.array([[2.5], [4.5]])",
        solution="clf = LogisticRegression().fit(X, y)\nanswer = clf.predict_proba(X_new)[:, 1]",
        hint="predict_proba returns two columns; class 1 is column index 1.",
        tol=1e-6,
    ),
    dict(
        prompt="Using `p` (probabilities of class 1) and `y_true`, compute two things: the AUC with sklearn (`auc`, float) and the number of rows flagged as positive at threshold 0.3 (`n_flagged`, int).",
        setup="p = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.2])\ny_true = np.array([0, 0, 1, 1, 1, 0])",
        solution="auc = float(roc_auc_score(y_true, p))\nn_flagged = int((p >= 0.3).sum())",
        hint="roc_auc_score(y_true, p) uses probabilities, not hard labels. (p >= 0.3).sum() counts flags.",
        answer_var=["auc", "n_flagged"],
        tol=1e-6,
    ),
    dict(
        prompt="Real data. Predict whether a meter is an SME from `annual_kwh_estimate`, `region`, `tariff`, using the given train/test split and pipeline `pre`. Fit a `Pipeline` of `pre` then `LogisticRegression(max_iter=1000)` and assign the test AUC (float, rounded to 3 decimals) to `answer`.",
        setup="from sklearn.model_selection import train_test_split\nmeters = pd.read_csv(\"../data/meters.csv\")\nmeters[\"region\"] = meters[\"region\"].str.title()\nXm = meters[[\"annual_kwh_estimate\", \"region\", \"tariff\"]]\nym = (meters[\"customer_type\"] == \"sme\").astype(int)\nXm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0, stratify=ym)\npre = ColumnTransformer([\n    (\"num\", make_pipeline(SimpleImputer(strategy=\"median\"), StandardScaler()), [\"annual_kwh_estimate\"]),\n    (\"cat\", make_pipeline(SimpleImputer(strategy=\"most_frequent\"), OneHotEncoder(handle_unknown=\"ignore\")), [\"region\", \"tariff\"]),\n])",
        solution="clf = Pipeline([(\"pre\", pre), (\"logit\", LogisticRegression(max_iter=1000))]).fit(Xm_tr, ym_tr)\np_te = clf.predict_proba(Xm_te)[:, 1]\nanswer = round(float(roc_auc_score(ym_te, p_te)), 3)",
        hint="Pipeline([('pre', pre), ('logit', LogisticRegression(max_iter=1000))]).fit(Xm_tr, ym_tr); predict_proba(Xm_te)[:, 1]; roc_auc_score.",
        tol=1e-3,
        note="AUC is 1.0: annual kWh separates SMEs perfectly in this synthetic data. Say aloud that a perfect score is suspicious and check why.",
    ),
]
