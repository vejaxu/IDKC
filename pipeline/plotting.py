"""KBC-style plots of the saved best seed under the mean-best PSI.

Full-sample plotting, projection, scatter styling and integer-label alignment
follow the current KBC implementation in ``../KBC/utils.py``:
all samples are plotted, t-SNE uses ``random_state=42``, and both panels share
one color map keyed by the sorted true classes, with a reserved gray color for
clusters that could not be matched to a class.
Plotting seed and clustering seed are independent.
"""

import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .files import read_csv, read_one


def _project(values, seed):
    if values.shape[1] <= 2:
        if values.shape[1] == 1:
            return np.column_stack((values[:, 0], np.zeros(values.shape[0])))
        return values
    if values.shape[0] < 3:
        return values[:, :2]
    perplexity = min(30.0, float(values.shape[0] - 1))
    return TSNE(n_components=2, random_state=seed, init="pca",
                learning_rate="auto", perplexity=perplexity).fit_transform(values)


def _scatter(values, labels, path, classes=None):
    unique = np.unique(labels) if classes is None else np.asarray(classes)
    if unique.size <= 10:
        colors = [plt.get_cmap("tab10")(index) for index in range(unique.size)]
    elif unique.size <= 20:
        colors = [plt.get_cmap("tab20")(index) for index in range(unique.size)]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, unique.size))
    # Reserve one color for clusters that could not be matched to a class.
    colors = list(colors) + [(0.5, 0.5, 0.5, 1.0)]
    color_map = ListedColormap(colors)
    lookup = {label: index for index, label in enumerate(unique)}
    encoded = np.array([lookup.get(label, len(unique)) for label in labels])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        values[:, 0],
        values[:, 1],
        c=encoded,
        cmap=color_map,
        vmin=-0.5,
        vmax=len(unique) + 0.5,
        alpha=0.7,
        s=15,
    )
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xlabel("")
    plt.gca().set_ylabel("")
    plt.title("")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def align_labels(y_true, y_pred):
    """Match observed clusters to observed classes; keep unmatched clusters distinct."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("Labels must be nonempty and have the same shape")
    classes, truth = np.unique(y_true, return_inverse=True)
    clusters, pred = np.unique(y_pred, return_inverse=True)
    counts = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(counts, (pred, truth), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = {int(row): int(classes[col]) for row, col in zip(rows, cols)}
    start = int(classes.min()) - len(clusters) - 1
    return np.array([mapping.get(int(i), start - int(i)) for i in pred])


def sha256(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def plot_best_run(dataset, directory, *, seed=42):
    directory = Path(directory)
    run = read_one(directory / "best_run.csv")
    average = read_one(directory / "best_psi.csv")
    seeds = read_csv(directory / "best_psi_seeds.csv")
    if (run["Psi"] != average["Psi"] or
            [row for row in seeds if row["Seed"] == run["Seed"]] != [run]
            or float(run["NMI"]) != max(float(row["NMI"]) for row in seeds)):
        raise ValueError("Saved best PSI / seed is inconsistent")
    labels_file = directory / "best_labels.csv"
    seed_file = directory / f"best_psi_labels_seed_{run['Seed']}.csv"
    if sha256(labels_file) != sha256(seed_file):
        raise ValueError("best_labels.csv does not match the selected seed")
    table = np.genfromtxt(labels_file, delimiter=",", names=True, encoding="utf-8")
    truth = np.asarray(table["TrueLabel"]).reshape(-1)
    raw = np.asarray(table["RawLabel"]).reshape(-1)
    if (not np.array_equal(truth, dataset.labels)
            or not np.array_equal(table["SampleIndex"], np.arange(1, truth.size + 1))):
        raise ValueError("Saved labels do not match the source sample order")
    _, truth_ids = np.unique(truth, return_inverse=True)
    _, raw_ids = np.unique(raw, return_inverse=True)
    # IDKC defines NMI=0 for zero entropy, and ARI=0 for two single groups.
    nmi = (0.0 if np.unique(truth).size == 1 or np.unique(raw).size == 1 else
           normalized_mutual_info_score(truth_ids, raw_ids, average_method="geometric"))
    ari = (0.0 if np.unique(truth).size == np.unique(raw).size == 1 else
           adjusted_rand_score(truth_ids, raw_ids))
    if abs(nmi - float(run["NMI"])) > 0.00006 or abs(ari - float(run["ARI"])) > 0.00006:
        raise ValueError("Saved labels disagree with best-run NMI/ARI")
    evaluation = np.asarray(table["AlignedLabel"]).reshape(-1)
    f1 = np.mean([2 * np.count_nonzero((truth == label) & (evaluation == label)) /
                  (np.count_nonzero(truth == label) + np.count_nonzero(evaluation == label))
                  for label in np.unique(truth)])
    if abs(f1 - float(run["MacroF1"])) > 0.00006:
        raise ValueError("Saved labels disagree with best-run macro-F1")
    indices = np.arange(truth.size)
    coordinates = _project(dataset.plot_features, seed)
    if coordinates.shape != (indices.size, 2) or not np.isfinite(coordinates).all():
        raise ValueError("Invalid plot coordinates")
    classes = np.unique(truth)
    aligned = align_labels(truth, raw)
    _scatter(coordinates, aligned, directory / "original_space_clustering.jpg", classes)
    _scatter(coordinates, truth, directory / "original_space_true_labels.jpg", classes)
    np.savetxt(directory / "kbc_style_label.csv", aligned, fmt="%d", delimiter=",",
               header="label", comments="")
    np.savez_compressed(directory / "best_run_plot_coordinates.npz",
                        sample_indices=indices + 1, coordinates=coordinates)
    projection = ("spatial locations" if dataset.spec.pos_key else
                  "raw" if dataset.plot_features.shape[1] <= 2 else "t-SNE")
    metadata = dict(
        dataset=dataset.spec.name, psi=int(run["Psi"]), clustering_seed=int(run["Seed"]),
        q=int(run["Q"]), plotting_seed=seed, source_input=str(dataset.source),
        input_key=dataset.spec.pos_key or dataset.spec.x_key,
        sample_size=int(truth.size), total_samples=int(truth.size),
        sample_policy="all samples", plot_input_dimensions=int(dataset.plot_features.shape[1]),
        projection=projection,
        features="locations" if dataset.spec.pos_key else "unscaled input",
        prediction="KBC-style Hungarian alignment on all samples",
        label_encoding="indices into sorted true class values",
        color_policy="one color map keyed by sorted true classes; gray for unmatched clusters",
        unassigned_raw_zero=int(np.count_nonzero(raw == 0)), labels_file="best_labels.csv",
        labels_sha256=sha256(labels_file), plotting_code_sha256=sha256(__file__),
        nmi=float(run["NMI"]), ari=float(run["ARI"]), macro_f1=float(run["MacroF1"]),
        load_to_label_seconds=float(run["LoadToLabelSeconds"]),
        perplexity=min(30.0, float(truth.size - 1)) if projection == "t-SNE" else None,
        runtime=dict(python=platform.python_version(), numpy=np.__version__,
                     scipy=scipy.__version__, sklearn=sklearn.__version__,
                     matplotlib=matplotlib.__version__))
    (directory / "best_run_plot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
