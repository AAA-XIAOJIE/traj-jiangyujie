# 实验协议与复现

[返回课程报告](../README.md)


直接使用教师包的三份 NDJSON，逐字节校验。原始 TXT 与第二周完全相同；按前述 Xiao 等（2018）实验设置使用 25 fps、厘米转米，采样间隔 2 帧。因此 8 个观测点首尾相隔 **0.56 s**，12 步预测覆盖 **0.96 s**，不是常见基准的 3.2/4.8 s 设置。

按时间先分 60%/20%/20%，再截窗。训练/验证/测试分别为 56/12/12 个场景，原始帧区间分别为 37–289、291–375、377–461，不共享帧。主要预测对象固定为 ID 1、17、33、49，其余 63 人作为潜在邻居。测试集只有 **3 个时间窗口 × 4 个主要行人**，同一集合内窗口重叠，不能把 12 个场景视为独立实验。

共同条件：种子 42/43/44，每组实际训练 10 轮，batch=2，Adam 学习率 0.001、weight decay=0.0001，StepLR 第 10 轮后衰减，无数据增强；CPU 两线程。每个种子下，四种配置的初始参数哈希和每轮样本顺序完全相同。所有 12 组训练和验证共约 10.9 分钟，记录见 [training.json](../results/runs/baseline/training.json)。

训练沿用教师的概率损失、真实邻居未来的 teacher forcing 和训练期远邻筛选。**验证与测试仅把 8 点历史传入模型，保留全部 64 人；所有人的未来均由模型递推，不用真实未来筛选邻居，也不输入真实目标。** 每轮按这种历史输入协议计算验证 ADE，取最低者保存权重；跨种子的平均验证 ADE 用于确定展示方案。全部选择固定后才计算正式测试指标。一次单轮代码检查仅用于确认接口可运行，未用于调整配置。

采用单次均值轨迹，计算主要行人在 12 个未来点的欧氏距离均值 ADE，以及第 12 点的 FDE；先算每个场景，再等权汇总。没有使用 best-of-K，也没有把训练 loss 当成 ADE/FDE。

## 六种模型的训练记录

原始四种配置属于第一阶段，共 12 组训练；等权求和与方向加权属于后续探索，共 6 组。六种配置均使用种子 42/43/44，各训练 10 轮，相同种子下的初始化和样本顺序一致。后两种机制在看到首批测试结果后提出，因此复用该测试集的结果只能作为探索证据。

- [原始四组](../results/runs/baseline/training.json) · [等权求和](../results/runs/sum_pool/training.json) · [方向加权](../results/runs/directional_sum/training.json)
- [逐步速度统计](../results/tables/split_diagnostics.csv) · [邻居统计](../results/tables/neighbour_audit.csv)
- [来源、许可证与 AI 辅助](../model/SOURCE.md)

## 安装环境

在仓库根目录使用独立 Python 3.9/3.10 环境。本次实际运行环境是 Python 3.9.23、PyTorch 2.5.1、CPU 两线程。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r week03/requirements.txt
```

## 用已提交的 18 份权重复核

```bash
python -X utf8 -m week03.model.run --evaluate-only
python -X utf8 -m week03.model.run --config week03/model/sum_config.json --output week03/results/runs/sum_pool --evaluate-only
python -X utf8 -m week03.model.run --config week03/model/direction_config.json --output week03/results/runs/directional_sum --evaluate-only
python -X utf8 -m week03.model.gallery
```

`gallery` 输出六模型完整图集及统一汇总表，固定种子 42，不按测试误差挑选场景。

可选诊断：[邻域网格](../results/figures/neighbour_information.png) · [原始四组与等权求和的配对热图](../results/figures/paired_scene_errors.png)。可分别用 `python -X utf8 -m week03.model.plots` 和 `python -X utf8 -m week03.model.details` 重画；它们生成的旧版总览、局部案例和重复汇总仅供本地查看，不加入 Git。

## 从头训练

```bash
python -X utf8 -m week03.model.run --output .local/week03_retrain/baseline
python -X utf8 -m week03.model.run --config week03/model/sum_config.json --output .local/week03_retrain/sum_pool
python -X utf8 -m week03.model.run --config week03/model/direction_config.json --output .local/week03_retrain/directional_sum
python -X utf8 -m week03.model.plots --results .local/week03_retrain/baseline
```

最后一条绘制重训后的第一阶段结果。完整图集脚本固定读取 `week03/results/runs/` 中的提交结果；它不会自动读取 `.local` 中的重训结果。

## 自动检查

```bash
python -m pip install pytest==8.4.2
python -m pytest -q tests/test_week03_model.py tests/test_week03_sum.py tests/test_week03_direction.py
```

15 项检查覆盖：教师源码与数据哈希、固定划分和主要目标、同初始权重和样本顺序、历史输入预测、屏蔽邻居、求和梯度及排列不变性、方向权重与静止回退，以及权重重载、预测数组、ADE/FDE 和验证选轮的一致性。
