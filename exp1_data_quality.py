from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mf_utils import (
    RATING_COLUMNS,
    annotate_bars,
    build_id_maps,
    configure_matplotlib,
    fit_model,
    load_ratings,
    predict_for_dataframe,
    rmse,
    save_figure,
    split_ratings,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiment1"


def inject_noise(
    train_df: pd.DataFrame,
    noise_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, int]:
    noisy_df = train_df.copy()
    rng = np.random.default_rng(random_state)
    num_noisy = int(round(len(noisy_df) * noise_ratio))
    # 翻转噪声聚焦于原始高分记录，更接近恶意差评/刷低分场景，也能确保翻转后与正常偏好显著背离。
    candidate_positions = noisy_df.index[noisy_df["Rating"] == 5].to_numpy()
    if len(candidate_positions) < num_noisy:
        candidate_positions = noisy_df.index[noisy_df["Rating"] >= 4].to_numpy()
    if len(candidate_positions) < num_noisy:
        candidate_positions = noisy_df.index[noisy_df["Rating"] != 3].to_numpy()
    if len(candidate_positions) < num_noisy:
        candidate_positions = noisy_df.index.to_numpy()
    noisy_positions = rng.choice(candidate_positions, size=num_noisy, replace=False)
    noisy_df.loc[noisy_positions, "Rating"] = 6.0 - noisy_df.loc[noisy_positions, "Rating"]
    noisy_df["Rating"] = noisy_df["Rating"].astype(np.float32)
    return noisy_df, int(num_noisy)


def build_reference_profiles(clean_train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_profiles = clean_train_df.groupby("UserID")["Rating"].agg(
        user_q1=lambda s: float(s.quantile(0.25)),
        user_q3=lambda s: float(s.quantile(0.75)),
        user_median="median",
        user_count="count",
    )
    user_profiles["user_iqr"] = user_profiles["user_q3"] - user_profiles["user_q1"]

    movie_profiles = clean_train_df.groupby("MovieID")["Rating"].agg(
        movie_mean="mean",
        movie_std=lambda s: float(s.std(ddof=0)),
        movie_count="count",
    )
    return user_profiles, movie_profiles


def clean_with_iqr_and_movie_stats(
    noisy_df: pd.DataFrame,
    user_profiles: pd.DataFrame,
    movie_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = noisy_df.join(user_profiles, on="UserID").join(movie_profiles, on="MovieID")

    user_lower = merged["user_q1"] - 1.5 * merged["user_iqr"]
    user_upper = merged["user_q3"] + 1.5 * merged["user_iqr"]

    conservative_lower = merged["user_median"] - 2.0
    conservative_upper = merged["user_median"] + 2.0

    user_iqr_mask = (
        (merged["user_count"] >= 15)
        & (
            (
                (merged["user_iqr"] > 0)
                & (
                    (merged["Rating"] < user_lower)
                    | (merged["Rating"] > user_upper)
                )
            )
            | (
                (merged["user_iqr"] == 0)
                & (
                    (merged["Rating"] < conservative_lower)
                    | (merged["Rating"] > conservative_upper)
                )
            )
        )
    )

    movie_threshold = np.maximum(2.0 * merged["movie_std"].fillna(0.0), 2.0)
    movie_global_mask = (
        (merged["movie_count"] >= 50)
        & ((merged["Rating"] - merged["movie_mean"]).abs() > movie_threshold)
    )

    remove_mask = user_iqr_mask | movie_global_mask
    keep_mask = ~remove_mask
    cleaned_df = merged.loc[keep_mask, RATING_COLUMNS].reset_index(drop=True)
    return cleaned_df, {
        "removed_total": int(remove_mask.sum()),
        "removed_user_iqr": int(user_iqr_mask.sum()),
        "removed_movie_global": int(movie_global_mask.sum()),
        "removed_intersection": int((user_iqr_mask & movie_global_mask).sum()),
    }


def recovery_rate(baseline_rmse: float, noisy_rmse: float, cleaned_rmse: float) -> float:
    degraded = noisy_rmse - baseline_rmse
    if degraded <= 1e-12:
        return 0.0
    return (noisy_rmse - cleaned_rmse) / degraded * 100.0


def describe_recovery(recovery_pct: float) -> str:
    if recovery_pct >= 0:
        return f"恢复 {recovery_pct:.1f}%"
    return f"恶化 {abs(recovery_pct):.1f}%"


def plot_experiment_one(
    baseline_rmse: float,
    results: list[dict[str, float]],
    output_dir: Path,
) -> None:
    labels = [f"{int(item['noise_ratio'] * 100)}%" for item in results]
    baseline_values = [baseline_rmse] * len(results)
    noisy_values = [item["noisy_rmse"] for item in results]
    cleaned_values = [item["cleaned_rmse"] for item in results]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, baseline_values, width, color="#4CAF50", label="Baseline")
    bars2 = ax.bar(x, noisy_values, width, color="#F44336", label="注入噪声后")
    bars3 = ax.bar(x + width, cleaned_values, width, color="#2196F3", label="IQR+全局检测清洗后")

    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    annotate_bars(ax, bars3)

    ax.set_title("实验一：数据质量对推荐精度的影响", fontsize=15, pad=14)
    ax.set_xlabel("噪声比例")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.45)
    ax.legend()

    save_figure(fig, output_dir / "figure1_data_quality_bar.png")

    trend_x = [0, 5, 10, 20]
    trend_noisy = [baseline_rmse] + noisy_values
    trend_cleaned = [baseline_rmse] + cleaned_values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        trend_x,
        trend_noisy,
        marker="o",
        linewidth=2.3,
        color="#F44336",
        label="注入噪声后",
    )
    ax.plot(
        trend_x,
        trend_cleaned,
        marker="o",
        linewidth=2.3,
        color="#2196F3",
        label="清洗后",
    )
    for x_value, y_value in zip(trend_x, trend_noisy):
        ax.text(x_value, y_value, f"{y_value:.4f}", ha="center", va="bottom", fontsize=9)
    for x_value, y_value in zip(trend_x, trend_cleaned):
        ax.text(x_value, y_value, f"{y_value:.4f}", ha="center", va="top", fontsize=9)

    ax.set_title("实验一：RMSE 随噪声水平变化趋势", fontsize=15, pad=14)
    ax.set_xlabel("噪声比例 (%)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(trend_x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.45)
    ax.legend()

    save_figure(fig, output_dir / "figure2_data_quality_trend.png")

    plot_experiment_one_compact(baseline_rmse, results, output_dir)


def plot_experiment_one_compact(
    baseline_rmse: float,
    results: list[dict[str, float]],
    output_dir: Path,
) -> None:
    labels = [f"{int(item['noise_ratio'] * 100)}%" for item in results]
    baseline_values = [baseline_rmse] * len(results)
    noisy_values = [item["noisy_rmse"] for item in results]
    cleaned_values = [item["cleaned_rmse"] for item in results]
    recovery_values = [item["recovery_pct"] for item in results]

    x = np.arange(len(labels))
    width = 0.23

    fig = plt.figure(figsize=(11.2, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.16)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_trend = fig.add_subplot(grid[0, 1])

    bars1 = ax_bar.bar(x - width, baseline_values, width, color="#4CAF50", label="Baseline")
    bars2 = ax_bar.bar(x, noisy_values, width, color="#F44336", label="注入噪声后")
    bars3 = ax_bar.bar(x + width, cleaned_values, width, color="#2196F3", label="清洗后")
    annotate_bars(ax_bar, bars1)
    annotate_bars(ax_bar, bars2)
    annotate_bars(ax_bar, bars3)

    ax_bar.set_title("核心结果", fontsize=12, pad=8)
    ax_bar.set_xlabel("噪声比例")
    ax_bar.set_ylabel("RMSE")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", alpha=0.3)

    trend_x = [0, 5, 10, 20]
    trend_noisy = [baseline_rmse] + noisy_values
    trend_cleaned = [baseline_rmse] + cleaned_values
    ax_trend.plot(
        trend_x,
        [baseline_rmse] * len(trend_x),
        color="#90A4AE",
        linestyle="--",
        linewidth=1.5,
        label="Baseline",
    )
    ax_trend.plot(
        trend_x,
        trend_noisy,
        marker="o",
        linewidth=2.0,
        color="#F44336",
        label="注入噪声后",
    )
    ax_trend.plot(
        trend_x,
        trend_cleaned,
        marker="o",
        linewidth=2.0,
        color="#2196F3",
        label="清洗后",
    )
    ax_trend.fill_between(trend_x, trend_noisy, trend_cleaned, color="#BBDEFB", alpha=0.25)
    for x_value, y_value in zip(trend_x[1:], trend_cleaned[1:]):
        ax_trend.text(x_value, y_value, f"{y_value:.4f}", ha="center", va="top", fontsize=8.5)

    recovery_text = "\n".join(
        [
            f"{label} 恢复 {recovery:.1f}%"
            for label, recovery in zip(labels, recovery_values)
        ]
    )
    ax_trend.text(
        0.98,
        0.08,
        recovery_text,
        transform=ax_trend.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D7DE"),
    )
    ax_trend.set_title("趋势变化", fontsize=12, pad=8)
    ax_trend.set_xlabel("噪声比例 (%)")
    ax_trend.set_ylabel("RMSE")
    ax_trend.set_xticks(trend_x)
    ax_trend.spines["top"].set_visible(False)
    ax_trend.spines["right"].set_visible(False)
    ax_trend.grid(axis="y", alpha=0.3)

    handles, legend_labels = ax_bar.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / "exp1_data_quality_compact.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    configure_matplotlib()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(args.data_dir)
    user_map, item_map = build_id_maps(ratings)
    train_df, test_df = split_ratings(ratings, test_size=0.2, random_state=args.random_state)
    user_profiles, movie_profiles = build_reference_profiles(train_df)
    y_test = test_df["Rating"].to_numpy(dtype=np.float32)

    baseline_model = fit_model(
        train_df,
        user_map,
        item_map,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        reg=args.reg,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    baseline_pred = predict_for_dataframe(baseline_model, test_df, user_map, item_map)
    baseline_rmse = rmse(y_test, baseline_pred)

    results: list[dict[str, float]] = []
    for noise_ratio in (0.05, 0.10, 0.20):
        noisy_train = inject_noise(
            train_df, noise_ratio=noise_ratio, random_state=args.random_state + int(noise_ratio * 1000)
        )
        noisy_train, flipped_count = noisy_train
        noisy_model = fit_model(
            noisy_train,
            user_map,
            item_map,
            n_factors=args.n_factors,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            reg=args.reg,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        noisy_pred = predict_for_dataframe(noisy_model, test_df, user_map, item_map)
        noisy_rmse = rmse(y_test, noisy_pred)

        cleaned_train, clean_stats = clean_with_iqr_and_movie_stats(
            noisy_train,
            user_profiles=user_profiles,
            movie_profiles=movie_profiles,
        )
        cleaned_model = fit_model(
            cleaned_train,
            user_map,
            item_map,
            n_factors=args.n_factors,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            reg=args.reg,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        cleaned_pred = predict_for_dataframe(cleaned_model, test_df, user_map, item_map)
        cleaned_rmse = rmse(y_test, cleaned_pred)

        results.append(
            {
                "noise_ratio": noise_ratio,
                "noisy_rmse": noisy_rmse,
                "cleaned_rmse": cleaned_rmse,
                "noise_change_pct": (noisy_rmse / baseline_rmse - 1.0) * 100.0,
                "recovery_pct": recovery_rate(baseline_rmse, noisy_rmse, cleaned_rmse),
                "flipped_count": float(flipped_count),
                "removed_count": float(clean_stats["removed_total"]),
                "removed_user_iqr": float(clean_stats["removed_user_iqr"]),
                "removed_movie_global": float(clean_stats["removed_movie_global"]),
                "removed_intersection": float(clean_stats["removed_intersection"]),
            }
        )

    plot_experiment_one(baseline_rmse, results, output_dir)
    print_experiment_one_summary(baseline_rmse, results, len(train_df))

    return {
        "baseline_rmse": baseline_rmse,
        "results": results,
        "output_dir": str(output_dir),
    }


def print_experiment_one_summary(
    baseline_rmse: float,
    results: list[dict[str, float]],
    train_size: int,
) -> None:
    print("========== 实验一结果 ==========")
    print(f"Baseline RMSE:      {baseline_rmse:.4f}")
    for item in results:
        noise_pct = int(item["noise_ratio"] * 100)
        removed_pct = item["removed_count"] / train_size * 100.0
        print(
            f"{noise_pct:<2d}% 噪声 RMSE:      {item['noisy_rmse']:.4f} "
            f"(↑ {item['noise_change_pct']:.1f}%)   清洗后: {item['cleaned_rmse']:.4f} "
            f"({describe_recovery(item['recovery_pct'])})"
        )
        print(
            f"    翻转噪声记录: {int(item['flipped_count'])} 条 "
            f"({item['flipped_count'] / train_size * 100.0:.2f}%)"
        )
        print(
            f"    IQR+全局检测剔除: {int(item['removed_count'])} 条 "
            f"({removed_pct:.2f}%)，其中用户IQR命中 {int(item['removed_user_iqr'])} 条，"
            f"电影全局命中 {int(item['removed_movie_global'])} 条，重叠 {int(item['removed_intersection'])} 条"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一：数据质量对推荐精度的影响")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="MovieLens 1M 数据目录；默认使用项目内置的 data/ml-1m",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="图表输出目录；默认输出到项目内的 outputs/experiment1",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-factors", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=50000)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
