"""Central configuration: constants, enumerations and model registry.

Keeping these in one place avoids magic numbers scattered across the codebase
and makes the application behaviour explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations for user-selectable options. Using ``str`` enums keeps the
# values friendly to display in the Streamlit UI while remaining type-safe.
# ---------------------------------------------------------------------------
class MissingValueStrategy(str, Enum):
    DROP_ROWS = "Drop rows with any missing value"
    MEAN_MODE = "Mean (numeric) / Mode (categorical)"
    MEDIAN_MODE = "Median (numeric) / Mode (categorical)"


class OutlierDetection(str, Enum):
    IQR = "IQR (Interquartile Range)"
    ZSCORE = "Z-score"


class OutlierTreatment(str, Enum):
    NONE = "Do nothing"
    REMOVE_ROWS = "Remove rows"
    CAP = "Cap (clip to bounds)"
    WINSORIZE = "Winsorize (5th/95th percentile)"


class ScalingMethod(str, Enum):
    STANDARD = "Standardization (Z-score)"
    MINMAX = "Normalization (Min-Max)"
    ROBUST = "Robust Scaling (median/IQR)"


class EncodingMethod(str, Enum):
    ONE_HOT = "One-Hot Encoding"
    LABEL = "Label Encoding"


class FeatureSelectionMethod(str, Enum):
    NONE = "None (keep all features)"
    VARIANCE = "Variance Threshold"
    CORRELATION = "Correlation-based removal"
    SELECT_K_BEST = "SelectKBest (ANOVA F-test)"
    MUTUAL_INFO = "Mutual Information"
    RFE = "RFE (Recursive Feature Elimination)"


class ProblemType(str, Enum):
    CLASSIFICATION = "Classification"


# ---------------------------------------------------------------------------
# Data / default constants
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2
MIN_ROWS_FOR_MODEL: int = 20
MIN_FEATURES_FOR_MODEL: int = 2
HIGH_CARDINALITY_THRESHOLD: int = 15
MAX_CLASSES: int = 20
MAX_ID_UNIQUE_RATIO: float = 0.5
MIN_CLASS_RATIO_WARNING: float = 0.3
OUTLIER_Z_THRESHOLD: float = 3.0
IQR_MULTIPLIER: float = 1.5
SMALL_DATASET_ROWS: int = 100

DEFAULT_K_SELECT = 5
DEFAULT_VARIANCE_THRESHOLD = 0.01
DEFAULT_CORR_THRESHOLD = 0.9


@dataclass
class PipelineConfig:
    """Immutable bundle of the user's preprocessing choices.

    This config object is what lets us rebuild the exact same transformation
    pipeline every run (and, combined with a fitted pipeline, lets a user
    download a single artefact that fully reproduces the trained model).
    """

    missing_strategy: MissingValueStrategy = MissingValueStrategy.MEAN_MODE
    detect_outliers: bool = True
    outlier_detection: OutlierDetection = OutlierDetection.IQR
    outlier_treatment: OutlierTreatment = OutlierTreatment.NONE
    scale: bool = True
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    encoding_method: EncodingMethod = EncodingMethod.ONE_HOT
    feature_selection: FeatureSelectionMethod = FeatureSelectionMethod.NONE
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    correlation_threshold: float = DEFAULT_CORR_THRESHOLD
    k_best: int = DEFAULT_K_SELECT
    n_features_rfe: int = DEFAULT_K_SELECT
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = RANDOM_STATE


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_NAMES: tuple[str, ...] = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Naive Bayes",
)


def build_model(name: str):
    """Return an unfitted classifier instance for the given display name.

    Parameters
    ----------
    name:
        One of :data:`MODEL_NAMES`.

    Returns
    -------
    A scikit-learn classifier with a sensible default hyper-parameterisation.
    The ``random_state`` is fixed for every model that supports it so results
    are reproducible.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "K-Nearest Neighbors": lambda: KNeighborsClassifier(n_jobs=-1),
        "Support Vector Machine": lambda: SVC(probability=True, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB,
    }
    if name not in factories:
        raise ValueError(f"Unknown model name: {name!r}. Choose one of {MODEL_NAMES}.")
    return factories[name]()
