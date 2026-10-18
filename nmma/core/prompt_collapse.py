"""
Prompt-collapse (PC) probability of a BNS merger from the remnant classifiers of
Puecher & Dietrich (arXiv:2408.10678, Zenodo record 12545674).

Classifier A (collapse vs NS remnant) takes ``Mtot``, ``LambdaTilde``, ``ChiEff`` and
``Mratio_fixed`` = m1/m2 >= 1 (note: NMMA's ``q`` = m2/m1 <= 1 is the inverse).

The released pickles were written with scikit-learn 1.4.0 and do not load with
newer versions (sklearn pickles are tied to the sklearn version, an sklearn limitation,
not an NMMA one). :class:`NumpyGradientBoostingClassifier` therefore reads the same
models from a plain ``classifier<X>_numpy.npz`` (no pickle, no scikit-learn), written
once with a scikit-learn 1.4.0 environment by
``bns_vs_nsbh_eos/dev/pc_classifier/nmma_integration/export_classifier_numpy.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Range spanned by the training data (dataset_classA.json); outside it the
# classifier extrapolates.
TRAIN_MTOT = (2.40, 3.44)
TRAIN_Q_GE1_MAX = 2.06
TRAIN_CHI_EFF = (-0.27, 0.41)

COLLAPSE_LABEL = 1
FEATURES = ["Mtot", "LambdaTilde", "ChiEff", "Mratio_fixed"]


NUMPY_KEYS = [
    "classes", "learning_rate", "init_raw", "tree_class", "tree_start",
    "left", "right", "feature", "threshold", "value", "mean", "scale",
]  # fmt: skip


class NumpyGradientBoostingClassifier:
    """
    A fitted ``sklearn.ensemble.GradientBoostingClassifier`` plus its ``StandardScaler``
    as plain arrays, with ``transform``, ``predict_proba`` and ``classes_``.

    Trees of all boosting stages are concatenated; tree ``t`` owns nodes
    ``tree_start[t]:tree_start[t+1]`` (children indices relative to its first node, -1 for
    a leaf) and adds ``learning_rate * value[leaf]`` to raw score column ``tree_class[t]``.
    ``init_raw`` is the initial raw score (one column for binary, one per class otherwise).
    Binary: ``P(classes[1]) = expit(raw)``; multiclass: ``softmax(raw)``. As in
    scikit-learn, a sample goes left when ``float32(x[feature]) <= threshold``.
    """

    def __init__(self, **arrays):
        missing = [k for k in NUMPY_KEYS if k not in arrays]
        if missing:
            raise ValueError(f"classifier file is missing keys: {missing}")
        self.classes_ = np.asarray(arrays["classes"])
        self.learning_rate = float(arrays["learning_rate"])
        self.init_raw = np.atleast_1d(np.asarray(arrays["init_raw"], dtype=float))
        self.tree_class = np.asarray(arrays["tree_class"], dtype=int)
        self.tree_start = np.asarray(arrays["tree_start"], dtype=int)
        self.left = np.asarray(arrays["left"], dtype=int)
        self.right = np.asarray(arrays["right"], dtype=int)
        self.feature = np.asarray(arrays["feature"], dtype=int)
        self.threshold = np.asarray(arrays["threshold"], dtype=float)
        self.value = np.asarray(arrays["value"], dtype=float)
        self.mean = np.asarray(arrays["mean"], dtype=float)
        self.scale = np.asarray(arrays["scale"], dtype=float)

    @classmethod
    def from_file(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k: data[k] for k in data.files})

    def transform(self, x):
        """Standardise (DataFrame or array) like the released ``StandardScaler``."""
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def _tree_values(self, x32, tree):
        sl = slice(self.tree_start[tree], self.tree_start[tree + 1])
        left, right = self.left[sl], self.right[sl]
        feature, threshold, value = self.feature[sl], self.threshold[sl], self.value[sl]
        node = np.zeros(len(x32), dtype=int)
        while True:
            internal = left[node] >= 0
            if not internal.any():
                return value[node]
            go_left = x32[np.arange(len(x32)), feature[node]] <= threshold[node]
            node = np.where(internal, np.where(go_left, left[node], right[node]), node)

    def predict_proba(self, x):
        x32 = np.asarray(x, dtype=np.float32)
        raw = np.tile(self.init_raw, (len(x32), 1))
        for tree, column in enumerate(self.tree_class):
            raw[:, column] += self.learning_rate * self._tree_values(x32, tree)
        if len(self.init_raw) == 1:
            p1 = 1.0 / (1.0 + np.exp(-raw[:, 0]))
            return np.column_stack([1.0 - p1, p1])
        raw -= raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)


class PromptCollapseClassifier:
    """
    Wrap a fitted binary classifier and its scaler.

    Parameters
    ----------
    model : object
        Has ``predict_proba`` and ``classes_``; class 1 is prompt collapse.
    scaler : object
        Has ``transform``, applied to a DataFrame with columns :data:`FEATURES`.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    @classmethod
    def from_directory(cls, directory, classifier="A"):
        """Load ``classifier<X>_numpy.npz`` if present (no scikit-learn needed), else the
        Zenodo ``classifier<X>_model.pkl`` and ``scaler_classifier<X>.pkl`` (these need
        scikit-learn 1.4.0)."""
        import joblib

        directory = Path(directory)
        numpy_file = directory / f"classifier{classifier}_numpy.npz"
        if numpy_file.is_file():  # preferred: no pickle, no scikit-learn needed
            model = NumpyGradientBoostingClassifier.from_file(numpy_file)
            return cls(model, model)
        model_file = directory / f"classifier{classifier}_model.pkl"
        scaler_file = directory / f"scaler_classifier{classifier}.pkl"
        for path in (model_file, scaler_file):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path.name} not found in {directory} (nor "
                    f"classifier{classifier}_numpy.npz); get the Zenodo files from "
                    "https://zenodo.org/records/12545674"
                )
        try:
            return cls(joblib.load(model_file), joblib.load(scaler_file))
        except (AttributeError, ModuleNotFoundError, ImportError) as err:
            raise RuntimeError(
                "Could not unpickle the prompt-collapse classifier; the Zenodo "
                "files were written with scikit-learn 1.4.0 and need that version "
                f"({err})"
            ) from err

    def probability(self, total_mass, lambda_tilde, chi_eff, mass_ratio):
        """
        P(prompt collapse).

        Parameters
        ----------
        total_mass, lambda_tilde, chi_eff : array_like
        mass_ratio : array_like
            m1/m2 >= 1, as the classifier was trained.
        """
        columns = [
            np.asarray(x, dtype=float) for x in (total_mass, lambda_tilde, chi_eff, mass_ratio)
        ]
        shape = np.broadcast(*columns).shape  # () for scalar inputs: callers rely on it
        x = pd.DataFrame(
            {
                name: np.broadcast_to(col, shape).ravel()
                for name, col in zip(FEATURES, columns)
            }
        )
        finite = np.isfinite(x.to_numpy()).all(axis=1)
        prob = np.full(len(x), np.nan)
        if finite.any():
            scaled = self.scaler.transform(x[finite])
            column = list(self.model.classes_).index(COLLAPSE_LABEL)
            prob[finite] = self.model.predict_proba(scaled)[:, column]
        return prob.reshape(shape)

    @staticmethod
    def outside_training_range(total_mass, chi_eff, mass_ratio):
        """True where the input lies outside the classifier's training range."""
        total_mass, chi_eff, mass_ratio = (
            np.asarray(x, dtype=float) for x in (total_mass, chi_eff, mass_ratio)
        )
        return (
            (total_mass < TRAIN_MTOT[0])
            | (total_mass > TRAIN_MTOT[1])
            | (mass_ratio > TRAIN_Q_GE1_MAX)
            | (chi_eff < TRAIN_CHI_EFF[0])
            | (chi_eff > TRAIN_CHI_EFF[1])
        )
