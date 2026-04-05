# MovieLens 1M 推荐系统治理实验

本目录包含两个可直接运行的实验脚本：

- `exp1_data_quality.py`：实验一，分析数据质量对推荐精度的影响
- `exp2_algorithm_fairness.py`：实验二，分析性别/年龄群体的推荐公平性
- `run_all_experiments.py`：一键顺序运行两个实验

## 数据准备

1. 下载 MovieLens 1M：<https://grouplens.org/datasets/movielens/1m/>
2. 将数据集放到当前项目目录下的 `data/ml-1m/`
3. 确保目录中至少包含：
   - `ratings.dat`
   - `users.dat`

## 运行方式

请先进入当前项目目录，再执行下面命令：

```bash
python3 exp1_data_quality.py
python3 exp2_algorithm_fairness.py
python3 run_all_experiments.py
```

如需显式指定数据集目录，也可以执行：

```bash
python3 exp1_data_quality.py --data-dir ./data/ml-1m
python3 exp2_algorithm_fairness.py --data-dir ./data/ml-1m
python3 run_all_experiments.py --data-dir ./data/ml-1m
```

如需调整模型参数，可附加：

```bash
--n-factors 20 --n-epochs 15 --learning-rate 0.01 --reg 0.05 --batch-size 50000
```

图表默认输出到 `outputs/` 目录，保存为高清 PNG（`dpi=200`）。
