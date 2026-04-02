from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


RATING_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
USER_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip"]


def configure_matplotlib() -> None:
    """Configure a clean, PPT-friendly plotting style with Chinese font fallback."""
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Heiti TC",
        "STHeiti",
        "Songti SC",
        "Kaiti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    selected_font = next(
        (font_name for font_name in preferred_fonts if font_name in available_fonts),
        "DejaVu Sans",
    )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#E8EDF3",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    ratings_path = Path(data_dir) / "ratings.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(f"未找到 ratings.dat: {ratings_path}")
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=RATING_COLUMNS,
        encoding="latin-1",
    )
    ratings["Rating"] = ratings["Rating"].astype(np.float32)
    return ratings


def load_users(data_dir: str | Path) -> pd.DataFrame:
    users_path = Path(data_dir) / "users.dat"
    if not users_path.exists():
        raise FileNotFoundError(f"未找到 users.dat: {users_path}")
    return pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=USER_COLUMNS,
        encoding="latin-1",
    )


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_id_maps(ratings: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    user_ids = sorted(ratings["UserID"].unique().tolist())
    item_ids = sorted(ratings["MovieID"].unique().tolist())
    user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_map = {movie_id: idx for idx, movie_id in enumerate(item_ids)}
    return user_map, item_map


def dataframe_to_arrays(
    df: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_idx = df["UserID"].map(user_map).to_numpy(dtype=np.int32)
    item_idx = df["MovieID"].map(item_map).to_numpy(dtype=np.int32)
    ratings = df["Rating"].to_numpy(dtype=np.float32)
    return user_idx, item_idx, ratings


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


class MatrixFactorizationSGD:
    """Mini-batch matrix factorization with user/item bias terms."""

    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 15,
        learning_rate: float = 0.01,
        reg: float = 0.05,
        batch_size: int = 50000,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.batch_size = batch_size
        self.random_state = random_state
        self.global_mean_: float | None = None
        self.user_bias_: np.ndarray | None = None
        self.item_bias_: np.ndarray | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.user_counts_: np.ndarray | None = None
        self.item_counts_: np.ndarray | None = None

    def fit(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        sample_weights: np.ndarray | None = None,
    ) -> "MatrixFactorizationSGD":
        rng = np.random.default_rng(self.random_state)
        scale = 0.1 / np.sqrt(self.n_factors)
        if sample_weights is None:
            sample_weights = np.ones(len(ratings), dtype=np.float32)
        else:
            sample_weights = sample_weights.astype(np.float32, copy=False)

        self.global_mean_ = float(np.average(ratings, weights=sample_weights))
        self.user_bias_ = np.zeros(n_users, dtype=np.float32)
        self.item_bias_ = np.zeros(n_items, dtype=np.float32)
        self.user_factors_ = rng.normal(
            0.0, scale, size=(n_users, self.n_factors)
        ).astype(np.float32)
        self.item_factors_ = rng.normal(
            0.0, scale, size=(n_items, self.n_factors)
        ).astype(np.float32)
        self.user_counts_ = np.bincount(user_idx, minlength=n_users).astype(np.int32)
        self.item_counts_ = np.bincount(item_idx, minlength=n_items).astype(np.int32)

        order = np.arange(len(ratings))
        lr = self.learning_rate
        reg = self.reg

        for _ in range(self.n_epochs):
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                u = user_idx[batch]
                i = item_idx[batch]
                r = ratings[batch]
                w = sample_weights[batch]

                pu = self.user_factors_[u]
                qi = self.item_factors_[i]
                bu = self.user_bias_[u]
                bi = self.item_bias_[i]

                pred = self.global_mean_ + bu + bi + np.sum(pu * qi, axis=1)
                err = r - pred
                weighted_err = w * err

                pu_old = pu.copy()
                qi_old = qi.copy()

                unique_u, inverse_u = np.unique(u, return_inverse=True)
                unique_i, inverse_i = np.unique(i, return_inverse=True)

                sum_err_u = np.bincount(
                    inverse_u, weights=weighted_err, minlength=len(unique_u)
                ).astype(np.float32)
                sum_err_i = np.bincount(
                    inverse_i, weights=weighted_err, minlength=len(unique_i)
                ).astype(np.float32)
                weight_sum_u = np.bincount(
                    inverse_u, weights=w, minlength=len(unique_u)
                ).astype(np.float32)
                weight_sum_i = np.bincount(
                    inverse_i, weights=w, minlength=len(unique_i)
                ).astype(np.float32)

                user_bias_grad = -sum_err_u + reg * weight_sum_u * self.user_bias_[unique_u]
                item_bias_grad = -sum_err_i + reg * weight_sum_i * self.item_bias_[unique_i]

                user_factor_grad = np.zeros(
                    (len(unique_u), self.n_factors), dtype=np.float32
                )
                item_factor_grad = np.zeros(
                    (len(unique_i), self.n_factors), dtype=np.float32
                )
                np.add.at(user_factor_grad, inverse_u, -weighted_err[:, None] * qi_old)
                np.add.at(item_factor_grad, inverse_i, -weighted_err[:, None] * pu_old)

                user_factor_grad += (
                    weight_sum_u[:, None] * reg * self.user_factors_[unique_u]
                )
                item_factor_grad += (
                    weight_sum_i[:, None] * reg * self.item_factors_[unique_i]
                )

                self.user_bias_[unique_u] -= lr * user_bias_grad
                self.item_bias_[unique_i] -= lr * item_bias_grad
                self.user_factors_[unique_u] -= lr * user_factor_grad
                self.item_factors_[unique_i] -= lr * item_factor_grad

        return self

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit().")

        user_seen = self.user_counts_[user_idx] > 0
        item_seen = self.item_counts_[item_idx] > 0

        pu = self.user_factors_[user_idx].copy()
        qi = self.item_factors_[item_idx].copy()
        pu[~user_seen] = 0.0
        qi[~item_seen] = 0.0

        bu = np.where(user_seen, self.user_bias_[user_idx], 0.0)
        bi = np.where(item_seen, self.item_bias_[item_idx], 0.0)

        pred = self.global_mean_ + bu + bi + np.sum(pu * qi, axis=1)
        return np.clip(pred, 1.0, 5.0)


def fit_model(
    train_df: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    n_factors: int = 20,
    n_epochs: int = 15,
    learning_rate: float = 0.01,
    reg: float = 0.05,
    batch_size: int = 50000,
    random_state: int = 42,
    sample_weights: np.ndarray | None = None,
) -> MatrixFactorizationSGD:
    user_idx, item_idx, ratings = dataframe_to_arrays(train_df, user_map, item_map)
    model = MatrixFactorizationSGD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        reg=reg,
        batch_size=batch_size,
        random_state=random_state,
    )
    model.fit(
        user_idx,
        item_idx,
        ratings,
        len(user_map),
        len(item_map),
        sample_weights=sample_weights,
    )
    return model


def predict_for_dataframe(
    model: MatrixFactorizationSGD,
    df: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
) -> np.ndarray:
    user_idx, item_idx, _ = dataframe_to_arrays(df, user_map, item_map)
    return model.predict(user_idx, item_idx)


def save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def describe_delta(change_pct: float, improved_when_negative: bool = True) -> str:
    improved = change_pct <= 0 if improved_when_negative else change_pct >= 0
    if improved:
        return f"改善 {abs(change_pct):.1f}%"
    return f"恶化 {abs(change_pct):.1f}%"


def describe_gap_change(old_gap: float, new_gap: float) -> str:
    if old_gap <= 1e-12:
        return "差距基本不变"
    delta_pct = (new_gap - old_gap) / old_gap * 100.0
    if delta_pct < 0:
        return f"缩小 {abs(delta_pct):.1f}%"
    if delta_pct > 0:
        return f"扩大 {abs(delta_pct):.1f}%"
    return "差距保持不变"


def annotate_bars(
    ax: plt.Axes,
    bars: Iterable,
    fmt: str = "{:.4f}",
    fontsize: int = 9,
) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )
