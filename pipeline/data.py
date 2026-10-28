"""Dataset keys shared by loading, MATLAB input conversion and plotting.

Field names and spatial-coordinate choices follow the KBC dataset configuration.
IDKC always applies its own min-max preprocessing; KBC's PCA/WL is not used.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from scipy.io import loadmat, savemat


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    x_key: str = "data"
    y_key: str = "class"
    pos_key: str | None = None


SPECS = (
    DatasetSpec("spiral", "spiral.mat"),
    DatasetSpec("AC", "AC.mat"),
    DatasetSpec("4C", "4C.mat"),
    DatasetSpec("RingG", "RingG.mat"),
    DatasetSpec("complex9", "complex9.mat"),
    DatasetSpec("USPS", "USPS.mat"),
    DatasetSpec("STL-10", "STL-10.mat"),
    DatasetSpec("Cifar-10", "Cifar-10.mat"),
    DatasetSpec("ImageNet-10", "ImageNet-10.mat"),
    DatasetSpec("ImageNet-Dogs", "ImageNet-Dogs.mat"),
    DatasetSpec("MNIST", "mnist.mat"),
    DatasetSpec("COIL20", "COIL20.mat", "X", "Y"),
    DatasetSpec("w1Gaussians", "w1Gaussians.mat"),
    DatasetSpec("w100Gaussians", "w100Gaussians.mat"),
    DatasetSpec("sparse_3_dense_3_dense_3", "sparse_3_dense_3_dense_3.mat"),
    DatasetSpec("sparse_8_dense_1_dense_1", "sparse_8_dense_1_dense_1.mat"),
    DatasetSpec("one_gaussian_10_one_line_5_2", "one_gaussian_10_one_line_5_2.mat"),
    DatasetSpec("tutorial", "processed_tutorial.pkl", "expression_scaled", "ground_truth", "locations"),
    DatasetSpec("tonsil", "processed_tonsil.pkl", "expression_scaled", "ground_truth", "locations"),
    DatasetSpec("airway", "processed_airway.pkl", "expression_scaled", "ground_truth", "locations"),
    DatasetSpec("crohn", "processed_crohn.pkl", "expression_scaled", "ground_truth", "locations"),
    DatasetSpec("dlpfc_151507", "151507_final.pkl", "expression_scaled", "ground_truth", "locations"),
    DatasetSpec("non_spherical", "non_spherical.mat"),
    DatasetSpec("non_spherical_gap", "non_spherical_gap.mat"),
    DatasetSpec("non_spherical_gap_0_5", "non_spherical_gap_0_5.mat"),
    DatasetSpec("non_spherical_gap_0_8", "non_spherical_gap_0_8.mat"),
)
BY_NAME = {spec.name: spec for spec in SPECS}


@dataclass
class Dataset:
    spec: DatasetSpec
    source: Path
    features: np.ndarray
    labels: np.ndarray
    plot_features: np.ndarray

    @property
    def info(self):
        return dict(dataset=self.spec.name, points=int(self.features.shape[0]),
                    dimensions=int(self.features.shape[1]),
                    clusters=int(np.unique(self.labels).size))

    def save_mat(self, filename):
        savemat(filename, {"data": self.features, "class": self.labels.reshape(-1, 1)})


def dense(values):
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def validate_name(name):
    # This name is both the output directory and the case-insensitive summary key.
    if not name or name in {".", ".."} or any(c in name for c in '/\\\r\n:*?"<>|'):
        raise ValueError(f"Invalid dataset directory name: {name!r}")
    if name.startswith("_") or name.startswith("."):
        raise ValueError("Dataset names cannot start with '.' or '_'")


def load_dataset(spec, source):
    validate_name(spec.name)
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() == ".mat":
        content = loadmat(source, variable_names=[key for key in
                          (spec.x_key, spec.y_key, spec.pos_key) if key])
    elif source.suffix.lower() in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            content = pickle.load(handle)
    else:
        raise ValueError("Input must be a MAT or PKL file")
    features = dense(content[spec.x_key])
    labels = np.asarray(content[spec.y_key]).reshape(-1)
    if not np.issubdtype(labels.dtype, np.number):
        _, labels = np.unique(labels, return_inverse=True)
    if (features.ndim != 2 or features.shape[0] != labels.size or
            labels.size < 2 or features.shape[1] < 1):
        raise ValueError("Expected an n-by-d feature matrix and n labels, n >= 2")
    if (not np.issubdtype(features.dtype, np.number) or np.iscomplexobj(features) or
            np.iscomplexobj(labels) or not np.isfinite(features).all() or
            not np.isfinite(labels).all()):
        raise ValueError("Features and labels must be finite real numbers")
    plot_features = dense(content[spec.pos_key]) if spec.pos_key else features
    if (plot_features.ndim != 2 or plot_features.shape[0] != labels.size or
            plot_features.shape[1] < 1 or not np.issubdtype(plot_features.dtype, np.number)
            or np.iscomplexobj(plot_features) or not np.isfinite(plot_features).all()):
        raise ValueError("Plot coordinates must have one finite row per sample")
    return Dataset(spec, source, features, labels, plot_features)
