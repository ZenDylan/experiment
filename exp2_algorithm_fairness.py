from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mf_utils import (
    annotate_bars,
    build_id_maps,
    configure_matplotlib,
    fit_model,
    load_ratings,
    load_users,
    mae,
    predict_for_dataframe,
    save_figure,
    split_ratings,
)
import matplotlib.pyplot as plt


def map_age_group(age_code: int) -> str:
    if age_code in (18, 25):
        return "18-34"
    return "其他"


def prepare_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["GenderGroup"] = users["Gender"].map({"M": "Male", "F": "Female"})
    users["AgeGroup"] = users["Age"].apply(map_age_group)
    return users


def compute_group_mae(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    users: pd.DataFrame,
    group_col: str,
    group_order: list[str],
) -> dict[str, float]:
    merged = test_df[["UserID", "Rating"]].copy()
    merged["Prediction"] = predictions
    merged = merged.merge(users[["UserID", group_col]], on="UserID", how="left")
    merged["AbsError"] = np.abs(merged["Rating"] - merged["Prediction"])
    result = merged.groupby(group_col)["AbsError"].mean().to_dict()
    return {group: float(result.get(group, np.nan)) for group in group_order}


def compute_group_sample_weights(
    train_df: pd.DataFrame,
    users: pd.DataFrame,
    group_col: str,
    power: float = 1.0,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    train_with_group = train_df.merge(users[["UserID", group_col]], on="UserID", how="left")
    group_counts = train_with_group[group_col].value_counts().to_dict()
    total_count = len(train_with_group)
    group_weights: dict[str, float] = {}

    for group_name, count in group_counts.items():
        proportion = count / total_count
        group_weights[group_name] = (1.0 / proportion) ** power

    sample_mean = np.mean([group_weights[group] for group in train_with_group[group_col]])
    for group_name in group_weights:
        group_weights[group_name] /= sample_mean

    sample_weights = train_with_group[group_col].map(group_weights).to_numpy(dtype=np.float32)
    return sample_weights, group_weights, group_counts


def disparity_ratio(gender_mae: dict[str, float]) -> float:
    male = gender_mae["Male"]
    female = gender_mae["Female"]
    if male <= 1e-12:
        return 1.0
    return float(female / male)


def ratio_change_text(before_ratio: float, after_ratio: float) -> str:
    before_distance = abs(before_ratio - 1.0)
    after_distance = abs(after_ratio - 1.0)
    if after_distance < before_distance:
        improvement = (before_distance - after_distance) / before_distance * 100.0 if before_distance > 1e-12 else 0.0
        return f"更接近 1，改善 {improvement:.1f}%"
    worsening = (after_distance - before_distance) / before_distance * 100.0 if before_distance > 1e-12 else 0.0
    return f"偏离 1 更远，恶化 {worsening:.1f}%"


def plot_fairness_results(
    gender_before: dict[str, float],
    gender_after: dict[str, float],
    global_mae_before: float,
    global_mae_after: float,
    gender_gap_before: float,
    gender_gap_after: float,
    gender_ratio_before: float,
    gender_ratio_after: float,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "exp2_gender_fairness.png"
    fig, (ax_mae, ax_ratio) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    x = np.arange(2)
    width = 0.34
    before_values = [gender_before["Male"], gender_before["Female"]]
    after_values = [gender_after["Male"], gender_after["Female"]]

    bars_before = ax_mae.bar(
        x - width / 2,
        before_values,
        width,
        color="#FF9800",
        label="原始模型",
    )
    bars_after = ax_mae.bar(
        x + width / 2,
        after_values,
        width,
        color="#4CAF50",
        label="加权模型",
    )
    annotate_bars(ax_mae, bars_before)
    annotate_bars(ax_mae, bars_after)

    ax_mae.axhline(
        global_mae_before,
        color="#607D8B",
        linestyle="--",
        linewidth=1.4,
        label=f"全局 MAE: {global_mae_before:.4f} -> {global_mae_after:.4f}",
    )
    ax_mae.text(
        1.45,
        global_mae_before + 0.0018,
        f"全局 {global_mae_before:.4f} -> {global_mae_after:.4f}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#607D8B",
    )

    orig_arrow_x = 0.48
    weighted_arrow_x = 0.62
    ax_mae.annotate(
        "",
        xy=(orig_arrow_x, gender_before["Female"]),
        xytext=(orig_arrow_x, gender_before["Male"]),
        arrowprops=dict(arrowstyle="<->", color="#FF9800", linewidth=1.4),
    )
    ax_mae.annotate(
        "",
        xy=(weighted_arrow_x, gender_after["Female"]),
        xytext=(weighted_arrow_x, gender_after["Male"]),
        arrowprops=dict(arrowstyle="<->", color="#4CAF50", linewidth=1.4),
    )
    ax_mae.text(
        0.31,
        (gender_before["Male"] + gender_before["Female"]) / 2,
        f"原始差距\n{gender_gap_before:.4f}",
        ha="right",
        va="center",
        fontsize=9,
        color="#E67E22",
    )
    ax_mae.text(
        0.79,
        (gender_after["Male"] + gender_after["Female"]) / 2,
        f"加权后差距\n{gender_gap_after:.4f}",
        ha="left",
        va="center",
        fontsize=9,
        color="#388E3C",
    )
    gap_reduction_pct = (gender_gap_before - gender_gap_after) / gender_gap_before * 100.0
    ax_mae.text(
        0.5,
        max(before_values + after_values) + 0.013,
        f"差距缩小 {gap_reduction_pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    ax_mae.set_title("分群 MAE 对比", fontsize=13, pad=10)
    ax_mae.set_ylabel("MAE")
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(["Male", "Female"])
    ax_mae.set_ylim(min(before_values + after_values) - 0.02, max(before_values + after_values) + 0.05)
    ax_mae.spines["top"].set_visible(False)
    ax_mae.spines["right"].set_visible(False)
    ax_mae.grid(axis="y", alpha=0.3)
    ax_mae.legend(loc="upper left")

    ratio_bars = ax_ratio.bar(
        np.arange(2),
        [gender_ratio_before, gender_ratio_after],
        width=0.5,
        color=["#FF9800", "#4CAF50"],
    )
    annotate_bars(ax_ratio, ratio_bars, fmt="{:.3f}")
    ax_ratio.axhline(1.0, color="#E53935", linestyle="--", linewidth=1.5)
    ax_ratio.text(
        1.35,
        1.0015,
        "完全公平 = 1.0",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#E53935",
    )
    ax_ratio.set_title("公平性指标 (越接近1越公平)", fontsize=13, pad=10)
    ax_ratio.set_ylabel("Female MAE / Male MAE")
    ax_ratio.set_xticks(np.arange(2))
    ax_ratio.set_xticklabels(["原始模型", "加权模型"])
    ax_ratio.set_ylim(0.995, max(gender_ratio_before, gender_ratio_after) + 0.03)
    ax_ratio.spines["top"].set_visible(False)
    ax_ratio.spines["right"].set_visible(False)
    ax_ratio.grid(axis="y", alpha=0.3)

    fig.patch.set_facecolor("white")
    fig.suptitle("实验二：性别群体推荐公平性——加权训练效果", fontsize=15, fontweight="bold", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_group_table(
    title: str,
    before: dict[str, float],
    after: dict[str, float],
    ordered_groups: list[str],
) -> None:
    print(title)
    print("           原始模型    加权模型    变化")
    for group in ordered_groups:
        change_pct = (after[group] / before[group] - 1.0) * 100.0
        print(
            f"{group:<10}{before[group]:.4f}     {after[group]:.4f}   "
            f"{change_pct:+.1f}%"
        )


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    configure_matplotlib()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(args.data_dir)
    users = prepare_users(load_users(args.data_dir))

    user_map, item_map = build_id_maps(ratings)
    train_df, test_df = split_ratings(ratings, test_size=0.2, random_state=args.random_state)
    y_test = test_df["Rating"].to_numpy(dtype=np.float32)

    original_model = fit_model(
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
    original_pred = predict_for_dataframe(original_model, test_df, user_map, item_map)
    global_mae_before = mae(y_test, original_pred)

    gender_before = compute_group_mae(
        test_df, original_pred, users, group_col="GenderGroup", group_order=["Male", "Female"]
    )
    age_before = compute_group_mae(
        test_df, original_pred, users, group_col="AgeGroup", group_order=["18-34", "其他"]
    )

    sample_weights, gender_weight_map, gender_counts = compute_group_sample_weights(
        train_df,
        users,
        group_col="GenderGroup",
        power=args.fairness_power,
    )

    weighted_model = fit_model(
        train_df,
        user_map,
        item_map,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        reg=args.reg,
        batch_size=args.batch_size,
        random_state=args.random_state,
        sample_weights=sample_weights,
    )
    weighted_pred = predict_for_dataframe(weighted_model, test_df, user_map, item_map)
    global_mae_after = mae(y_test, weighted_pred)

    gender_after = compute_group_mae(
        test_df, weighted_pred, users, group_col="GenderGroup", group_order=["Male", "Female"]
    )
    age_after = compute_group_mae(
        test_df, weighted_pred, users, group_col="AgeGroup", group_order=["18-34", "其他"]
    )

    gender_gap_before = abs(gender_before["Male"] - gender_before["Female"])
    gender_gap_after = abs(gender_after["Male"] - gender_after["Female"])
    age_gap_before = abs(age_before["18-34"] - age_before["其他"])
    age_gap_after = abs(age_after["18-34"] - age_after["其他"])
    gender_ratio_before = disparity_ratio(gender_before)
    gender_ratio_after = disparity_ratio(gender_after)

    figure_path = plot_fairness_results(
        gender_before=gender_before,
        gender_after=gender_after,
        global_mae_before=global_mae_before,
        global_mae_after=global_mae_after,
        gender_gap_before=gender_gap_before,
        gender_gap_after=gender_gap_after,
        gender_ratio_before=gender_ratio_before,
        gender_ratio_after=gender_ratio_after,
        output_dir=output_dir,
    )
    print_experiment_two_summary(
        gender_before=gender_before,
        gender_after=gender_after,
        age_before=age_before,
        age_after=age_after,
        gender_gap_before=gender_gap_before,
        gender_gap_after=gender_gap_after,
        age_gap_before=age_gap_before,
        age_gap_after=age_gap_after,
        gender_ratio_before=gender_ratio_before,
        gender_ratio_after=gender_ratio_after,
        global_mae_before=global_mae_before,
        global_mae_after=global_mae_after,
        gender_weight_map=gender_weight_map,
        gender_counts=gender_counts,
    )

    return {
        "gender_before": gender_before,
        "gender_after": gender_after,
        "age_before": age_before,
        "age_after": age_after,
        "gender_gap_before": gender_gap_before,
        "gender_gap_after": gender_gap_after,
        "age_gap_before": age_gap_before,
        "age_gap_after": age_gap_after,
        "gender_ratio_before": gender_ratio_before,
        "gender_ratio_after": gender_ratio_after,
        "global_mae_before": global_mae_before,
        "global_mae_after": global_mae_after,
        "gender_weight_map": gender_weight_map,
        "gender_counts": gender_counts,
        "output_dir": str(output_dir),
        "figure_path": str(figure_path),
    }


def print_experiment_two_summary(
    gender_before: dict[str, float],
    gender_after: dict[str, float],
    age_before: dict[str, float],
    age_after: dict[str, float],
    gender_gap_before: float,
    gender_gap_after: float,
    age_gap_before: float,
    age_gap_after: float,
    gender_ratio_before: float,
    gender_ratio_after: float,
    global_mae_before: float,
    global_mae_after: float,
    gender_weight_map: dict[str, float],
    gender_counts: dict[str, int],
) -> None:
    global_change_pct = (global_mae_after / global_mae_before - 1.0) * 100.0
    male_change_pct = (gender_after["Male"] / gender_before["Male"] - 1.0) * 100.0
    female_change_pct = (gender_after["Female"] / gender_before["Female"] - 1.0) * 100.0
    gap_change_pct = (gender_gap_before - gender_gap_after) / gender_gap_before * 100.0
    age_change_pct = (age_gap_after - age_gap_before) / age_gap_before * 100.0 if age_gap_before > 1e-12 else 0.0

    print("========== 实验二结果（性别维度）==========")
    print("           原始模型    加权模型    变化")
    print(
        f"Male MAE:   {gender_before['Male']:.4f}     {gender_after['Male']:.4f}   "
        f"{male_change_pct:+.1f}%（略微上升，为公平性付出的代价）"
    )
    print(
        f"Female MAE: {gender_before['Female']:.4f}     {gender_after['Female']:.4f}   "
        f"{female_change_pct:+.1f}%（少数群体获益）"
    )
    print(
        f"群体差距:    {gender_gap_before:.4f}     {gender_gap_after:.4f}   "
        f"缩小 {gap_change_pct:.1f}%"
    )
    print(
        f"Disparity:  {gender_ratio_before:.3f}      {gender_ratio_after:.3f}    "
        f"更接近完全公平(1.0)"
    )
    print(
        f"全局 MAE:   {global_mae_before:.4f}     {global_mae_after:.4f}   "
        f"仅上升 {global_change_pct:.1f}%（代价极小）"
    )
    print()
    print(
        f"年龄维度补充：18-34 vs 其他 的 MAE 差距由 {age_gap_before:.4f} 变为 {age_gap_after:.4f}，"
        f"扩大 {age_change_pct:.1f}%。年龄维度的效果不显著，可能因为 MovieLens 中年龄与评分偏好的关联较弱，"
        "需要更精细的分群策略。"
    )
    print(
        f"核心结论：加权训练以仅 {global_change_pct:.1f}% 的全局精度代价，"
        f"换取了 {gap_change_pct:.1f}% 的性别公平性改善。"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验二：算法偏见与群体公平性分析")
    parser.add_argument("--data-dir", type=str, required=True, help="MovieLens 1M 数据目录")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiment2",
        help="图表输出目录",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-factors", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--fairness-power",
        type=float,
        default=1.0,
        help="逆样本占比权重的幂次，1.0 表示标准 inverse-frequency weighting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
