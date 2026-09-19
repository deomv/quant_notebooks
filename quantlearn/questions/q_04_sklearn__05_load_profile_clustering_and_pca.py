"""Exercises for 04_sklearn/05_load_profile_clustering_and_pca.ipynb"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

COMMON_SETUP = """
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import silhouette_score
"""


def _same_clustering(got, exp):
    try:
        g = np.asarray(got).ravel()
        e = np.asarray(exp).ravel()
    except Exception:
        return False, "expected an array of cluster labels"
    if g.shape != e.shape:
        return False, f"expected {e.shape[0]} labels, got {g.shape[0]}"
    ok = adjusted_rand_score(e, g) > 0.999
    return ok, "" if ok else "the grouping differs from the reference (label numbers may differ, the grouping must not)"


QUESTIONS = [
    dict(
        prompt="Cluster the six 2-D points `P` into 2 groups with `KMeans(n_clusters=2, random_state=0, n_init=10)`. Assign the label array to `answer` (label numbering does not matter, the grouping does).",
        setup="P = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)",
        solution="answer = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(P)",
        hint="KMeans(...).fit_predict(P)",
        check_fn=_same_clustering,
    ),
    dict(
        prompt="Fit the same KMeans on `P` and assign the cluster centres, **sorted by their first coordinate**, as a 2×2 array to `answer`.",
        setup="P = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)",
        solution="km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(P)\nc = km.cluster_centers_\nanswer = c[np.argsort(c[:, 0])]",
        hint="km.cluster_centers_ then sort rows by column 0 with np.argsort.",
        tol=1e-6,
    ),
    dict(
        prompt="Recompute the centre of the cluster containing point `[0, 0]` by hand: the mean of the points that share its label. Assign the 1-D array to `answer`.",
        setup="P = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)\nlabels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(P)",
        solution="answer = P[labels == labels[0]].mean(axis=0)",
        hint="Mask P with labels == labels[0]; mean over axis 0.",
        tol=1e-9,
    ),
    dict(
        prompt="Compute the inertia by hand: sum over points of squared distance to their own cluster centre. Assign the float to `answer` (it should equal `km.inertia_`).",
        setup="P = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], dtype=float)\nkm = KMeans(n_clusters=2, random_state=0, n_init=10).fit(P)",
        solution="centres = km.cluster_centers_[km.labels_]\nanswer = float(((P - centres) ** 2).sum())",
        hint="km.cluster_centers_[km.labels_] gives each point's centre; subtract, square, sum everything.",
        tol=1e-6,
    ),
    dict(
        prompt="`D` holds five 4-period daily profiles; two are big versions of the others. Normalise each row to sum to 1 (shape, not size). Assign the normalised array to `answer`.",
        setup="D = np.array([[1, 1, 4, 2], [10, 10, 40, 20], [3, 1, 1, 3], [30, 10, 10, 30], [2, 2, 2, 2]], dtype=float)",
        solution="answer = D / D.sum(axis=1, keepdims=True)",
        hint="Divide by the row sums; keepdims=True (or reshape) so broadcasting works row-wise.",
        tol=1e-9,
    ),
    dict(
        prompt="Cluster the row-normalised `D` into 2 groups with `KMeans(n_clusters=2, random_state=0, n_init=10)`. Assign the labels to `answer`. The two 'big' days should now join their small twins.",
        setup="D = np.array([[1, 1, 4, 2], [10, 10, 40, 20], [3, 1, 1, 3], [30, 10, 10, 30], [2, 2, 2, 2]], dtype=float)\nDn = D / D.sum(axis=1, keepdims=True)",
        solution="answer = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(Dn)",
        hint="fit_predict on Dn, not on D.",
        check_fn=_same_clustering,
    ),
    dict(
        prompt="For k in [2, 3, 4], fit `KMeans(n_clusters=k, random_state=0, n_init=10)` on `Q` and collect the inertia. Assign a Series indexed by k with the inertias to `answer`.",
        setup="rng = np.random.default_rng(0)\nQ = np.vstack([rng.normal(0, 0.3, (10, 2)), rng.normal(5, 0.3, (10, 2)), rng.normal([0, 5], 0.3, (10, 2))])",
        solution="vals = {}\nfor k in [2, 3, 4]:\n    vals[k] = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Q).inertia_\nanswer = pd.Series(vals)",
        hint="Loop over k; .fit(Q).inertia_; pd.Series(dict).",
        tol=1e-4,
    ),
    dict(
        prompt="Compute the silhouette score of the k=3 clustering of `Q` (same KMeans settings). Assign the float to `answer`.",
        setup="rng = np.random.default_rng(0)\nQ = np.vstack([rng.normal(0, 0.3, (10, 2)), rng.normal(5, 0.3, (10, 2)), rng.normal([0, 5], 0.3, (10, 2))])",
        solution="labels = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(Q)\nanswer = float(silhouette_score(Q, labels))",
        hint="silhouette_score(Q, labels).",
        tol=1e-6,
    ),
    dict(
        prompt="Fit `PCA(n_components=2)` on the 4×2 matrix `M` and assign `explained_variance_ratio_` to `answer`.",
        setup="M = np.array([[1, 2], [2, 4], [3, 6.5], [4, 8]], dtype=float)",
        solution="answer = PCA(n_components=2).fit(M).explained_variance_ratio_",
        hint="PCA(n_components=2).fit(M).explained_variance_ratio_",
        tol=1e-6,
    ),
    dict(
        prompt="Project `M` onto its first principal component and reconstruct it back with `PCA(n_components=1)`. Assign the reconstruction error (sum of squared differences between `M` and the reconstruction, float) to `answer`.",
        setup="M = np.array([[1, 2], [2, 4], [3, 6.5], [4, 8]], dtype=float)",
        solution="pca = PCA(n_components=1).fit(M)\nrecon = pca.inverse_transform(pca.transform(M))\nanswer = float(((M - recon) ** 2).sum())",
        hint="pca.inverse_transform(pca.transform(M)) is the reconstruction.",
        tol=1e-6,
    ),
    dict(
        prompt="`GroupKFold(n_splits=3)` on 6 rows with `groups`. Assign a list of the **test group names** per fold (each fold as a sorted list of the distinct group values in its test set) to `answer`.",
        setup="X = np.arange(6).reshape(-1, 1)\ngroups = np.array([\"A\", \"A\", \"B\", \"B\", \"C\", \"C\"])",
        solution="answer = [sorted(set(groups[te])) for _, te in GroupKFold(n_splits=3).split(X, groups=groups)]",
        hint="GroupKFold(...).split(X, groups=groups) yields (train_idx, test_idx); groups[test_idx] tells you which group is held out.",
        check_fn=lambda got, exp: (sorted(map(list, got)) == sorted(map(list, exp)), "" if sorted(map(list, got)) == sorted(map(list, exp)) else "each fold should hold out exactly one whole group"),
    ),
    dict(
        prompt="Real data. `X_shape` holds row-normalised meter-day profiles (48 columns). Standardise the columns with `StandardScaler`, fit `PCA(n_components=3)` and assign the **cumulative** explained variance ratio of the 3 components (float, rounded to 3 decimals) to `answer`.",
        setup="raw = pd.read_csv(\"../data/meter_halfhourly_2023.csv.gz\").drop_duplicates()\nraw[\"date\"] = pd.to_datetime(raw[\"settlement_date\"])\nraw = raw[raw[\"meter_id\"] != \"M100007\"]\nper_day = raw.groupby([\"meter_id\", \"date\"])[\"kwh\"].agg(n=\"size\", nunique=\"nunique\")\nkeep = per_day[(per_day[\"n\"] == 48) & (per_day[\"nunique\"] > 1)].index\nraw = raw.set_index([\"meter_id\", \"date\"]).loc[keep].reset_index()\nX_raw = raw.pivot_table(index=[\"meter_id\", \"date\"], columns=\"settlement_period\", values=\"kwh\")\ngross = X_raw.abs().sum(axis=1)\nX_shape = X_raw[gross > 1].div(gross[gross > 1], axis=0)",
        solution="Z = StandardScaler().fit_transform(X_shape)\npca = PCA(n_components=3).fit(Z)\nanswer = round(float(pca.explained_variance_ratio_.sum()), 3)",
        hint="StandardScaler().fit_transform(X_shape) then PCA(n_components=3).fit(...).explained_variance_ratio_.sum().",
        tol=2e-3,
    ),
]
