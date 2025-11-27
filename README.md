# SDSS17 Stellar Classification (STAR/GALAXY/QSO)

用机器学习对 Kaggle 的 **Stellar Classification Dataset — SDSS17** 数据进行三分类（STAR / GALAXY / QSO）。本仓库主线方法为 **XGBoost**、**MLP（神经网络）**和**RandomForest**，**Logistic** 作为参考基线。

> 数据集网址：https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data

---

## 环境与安装

建议创建独立虚拟环境，环境名：**sdss17-ml-py311**（Python 3.11）。

- **Conda**
  ```bash
  conda create -n sdss17-ml-py311 python=3.11 -y
  conda activate sdss17-ml-py311
  pip install -r requirements.txt
  ```

- **venv（Windows）**
  ```bash
  py -3.11 -m venv .venv
  .\.venv\Scripts\activate
  pip install -r requirements.txt
  ```

`requirements.txt` 包含：
```
scikit-learn~=1.7.2
xgboost~=3.1.1
numpy~=2.3.4
pandas~=2.3.3
joblib~=1.5.2
matplotlib~=3.10.7
tqdm~=4.67.1
```

> 注：MLP 使用的是 scikit-learn 自带的 `MLPClassifier`，XGBoost 需单独安装。

---

## 数据放置

将 Kaggle 下载的 CSV 放到：
```
data/sdss17.csv
```
脚本会自动丢弃 ID/采集元数据列，保留并派生光度学相关特征（`u,g,r,i,z, redshift` 以及颜色指数 `u-g, g-r, r-i, i-z, u-r, g-i`）。

---

## 快速开始

> 直接使用 `train.py` 进行训练与评测，无需进行超参数搜索。

### Logistic （基线）
```bash
  python -m scripts.train --csv data/sdss17.csv --model logistic --outdir outputs/logistic
```

### RandomForest
```bash
  python -m scripts.train --csv data/sdss17.csv --model random_forest --outdir outputs/random_forest
```

### XGBoost
```bash
  python -m scripts.train --csv data/sdss17.csv --model xgboost --outdir outputs/xgboost
```

### MLP（神经网络对照）
```bash
  python -m scripts.train --csv data/sdss17.csv --model mlp --outdir outputs/mlp
```

训练完成后，产物位于 `outputs/<model>/`：：
- `*_final.joblib`：已训练好的模型包（含流水线与标签编码器 `LabelEncoder` 以及特征列顺序）。
- `cv_*_metrics.json`：K 折评测汇总（Accuracy、Macro F1、Weighted F1、每折细节与混淆矩阵）。
- `cm_*.png`：混淆矩阵图（直观查看误判模式）。

---

## 超参数搜索

> 如果你希望进一步提升效果，可使用 `scripts/search.py` 进行随机搜索（评分使用 **Macro F1**）。

### Logistic
```bash
  python -m scripts.search --csv data/sdss17.csv --model logistic --n_iter 20 --folds 5 --outdir outputs/logistic
```

### RandomForest
```bash
  python -m scripts.search --csv data/sdss17.csv --model random_forest --n_iter 20 --folds 5 --outdir outputs/random_forest
```

### XGBoost
  ```bash
  python -m scripts.search --csv data/sdss17.csv --model xgboost --n_iter 20 --folds 5 --outdir outputs/xgboost
  ```
### MLP
  ```bash
  python -m scripts.search --csv data/sdss17.csv --model mlp --n_iter 20 --folds 5 --outdir outputs/mlp
  ```

自动生成：
- `search_<model>_cv_results.csv`：所有候选配置的 CV 成绩表  
- `search_<model>_summary.json`：最佳配置 + 均值/方差 + 每折混淆矩阵  
- `<model>_best_final.joblib`：最优超参下全量数据训练的最终模型  
- `top_features_<model>.png`：全局 Top 特征（XGBoost 使用 `feature_importances_`；MLP 无内置全局重要性，数值可能为 0）

---

## 代码结构

```
sdss17_ml/
├── data/
│   └── sdss17.csv
├── figs/                 # 可视化输出目录
│   ├── logistic/
│   ├── mlp/
│   ├── random_forest/
│   ├── xgboost/
│   └── leaderboard_all_models.png
├── outputs/              # 训练输出目录
│   ├── logistic/
│   ├── mlp/
│   ├── random_forest/
│   └── xgboost/
├── scripts/
│   ├── search.py
│   ├── train.py
│   └── visualize.py
├── src/
│   ├── data_utils.py    # 读CSV，拆分X/y，丢弃ID/元数据/天区坐标
│   ├── features.py      # 颜色指数与数值化、缺失值处理
│   ├── metrics.py       # 统一计算 Accuracy / Macro F1 / 混淆矩阵
│   └── models.py        # 定义 xgboost / mlp / logistic (基线） / random_forest
├── README.md
└── requirements.txt

```

---

## 可视化

本项目提供了一套完整的可视化工具，用于分析和展示不同机器学习模型在恒星分类任务上的性能表现。通过 `scripts/visualize.py` 脚本可以生成多种类型的图表。

### 运行可视化

```bash
  python -m scripts.visualize --outputs_dir outputs --figs_dir figs
```
该命令会扫描 outputs/ 目录下的所有模型结果，并在 figs/ 目录中生成相应的可视化图表。

### 生成的图表类型

1.模型排行榜 (Leaderboard)
- 文件：`leaderboard_all_models.png`
- 说明：展示各模型的宏观F1分数(CV)对比，帮助快速比较不同算法的整体性能

2.交叉验证折线图 (Fold Curves)
- 文件：`figs/<model>/cv_fold_scores.png`
- 说明：显示每个交叉验证折的准确率和宏观F1分数，用于评估模型稳定

3.混淆矩阵网格 (Confusion Matrix Grid)
- 文件：`figs/<model>/cm_grid_row_norm.png`
- 说明：展示所有交叉验证折的归一化混淆矩阵，直观显示各类别间的分类准确性及误判模式

4.类别级指标柱状图 (Per-class Metrics)
- 文件：`figs/<model>/per_class_bars_lastfold.png`
- 说明：显示最后一折中每个类别的精确率(Precision)、召回率(Recall)和F1分数

5.全局特征重要性柱状图 (Global Feature Importance)
- 文件：`figs/<model>/top_features_from_joblib.png`
- 说明：展示模型中最重要的特征，对于树模型显示特征重要性，对于线性模型显示系数绝对

6.随机森林树深度分布 (Random Forest Tree Depth)
- 文件：` figs/random_forest/rf_tree_depth.png`
- 说明：仅针对随机森林模型，显示各决策树的深度分布情况

### 主要评估指标

- **宏观F1分数 (Macro F1)**：各类别F1分数的算术平均，是本项目的主要评估指标
- **准确率 (Accuracy)**：整体分类正确的样本比例
- **精确率 (Precision)**：预测为正类中实际为正类的比例
- **召回率 (Recall)**：实际为正类中预测为正类的比例

### 结果解读
- **关键特征**：常见重要特征包括 `redshift` 与颜色指数（如 `g-r`, `u-g`）
- **常见误判**：`STAR ↔ QSO` 之间容易产生误判
- **模型稳定性**：通过观察各折性能差异判断模型稳定性

---

## 常见问题

- **训练较慢**：XGBoost 可先减小 `n_estimators`（如 300）。  
- **GPU 加速**：若已正确配置 NVIDIA GPU，可在 `src/models.py` 将 XGBoost 的 `tree_method` 改为 `"gpu_hist"`。  

---

## 致谢

- Kaggle: *Stellar Classification Dataset — SDSS17*  
- The Sloan Digital Sky Survey (SDSS) DR17 团队
