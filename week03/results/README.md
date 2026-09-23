# Week 03 · 结果文件索引

[返回课程报告](../README.md) · [复现说明](../docs/PROTOCOL.md)

## 阅读图与结果

- [完整轨迹图集](gallery/README.md)：六种模型各自的全部 12 个场景，以及全部 64 人对照。
- [统一 ADE/FDE 表](tables/extended_summary.csv)：六模型与匀速参考，验证集和测试集，三个种子均值及标准差。
- [图表目录](figures)：第一阶段总览、邻域示意、逐场景配对热图、局部案例。
- [统计与明细](tables)：邻居数量、速度变化、配对误差；`summary.csv` 仅为第一阶段四模型汇总，最终比较使用 `extended_summary.csv`。

## 复核实验

| 目录 | 模型 | 已训练权重 |
| :--- | :--- | ---: |
| [runs/baseline](runs/baseline) | 原始、屏蔽邻居、2 m、8 m | 12 |
| [runs/sum_pool](runs/sum_pool) | 等权求和 | 3 |
| [runs/directional_sum](runs/directional_sum) | 方向加权求和 | 3 |

每个实验目录采用相同结构：`checkpoints/` 为权重，`predictions.npz` 为预测与真实轨迹，`metrics.csv` 为逐种子指标，`per_scene.csv` 为逐场景指标，`learning_curves.csv` 为验证曲线，`training.json` 为种子、参数与哈希记录，`validation_selection.json` 为选轮配置，`evaluation.json` 为评价协议。

[audit/](audit) 记录方向加权的逐种子变化、局部案例的事后选择规则及图件说明。目录整理不改变权重、数据或实验数值。
