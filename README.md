# traj-jiangyujie · 行人轨迹课程

本仓库用于逐周完成行人轨迹课程。当前已完成 **Week 01 / A1：ETH/UCY 五场景轨迹可视化**，后续周次独立存放。

![ETH/UCY 五场景轨迹总览](week01/figures/eth_ucy_trajectory_overview.png)

**查看本周作业：** [完整报告](week01/README.md) · [一段结论](week01/conclusion.md) · [矢量 PDF](week01/figures/eth_ucy_trajectory_overview.pdf) · [可编辑 SVG](week01/figures/eth_ucy_trajectory_overview.svg)

## 已完成的内容

- 从 GitHub 固定提交下载 `seq_eth`、`seq_hotel`、`zara01`、`zara02`、`students03` 五个场景；`students03` 对应源目录 `univ`、原始名称 `students003`。
- 解析原生标注，按场景与个体 ID 分组、按时间排序，生成每人独立的轨迹。
- 生成五张“全场景 + 12 s 细节”图和一张总览图，均有 PNG、PDF、SVG。
- 提供数据来源与 SHA-256 校验、标准化字段、逐轨迹统计、断线审计、阈值敏感性与运行记录。

当前输入是已有标注的轨迹坐标，属于**轨迹读取与可视化任务**；A1 不需要重新训练目标检测器或从视频跟踪。A2–A5 的速度、间距/TTC、基本图和行为判别留待后续周次。

## 安装与一键复现

已验证：Python 3.12。推荐使用独立虚拟环境。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -X utf8 scripts/run_week01.py --download
.\.venv\Scripts\python.exe -m pytest -q
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python scripts/run_week01.py --download
.venv/bin/python -m pytest -q
```

首次下载约 3.1 MB，共 25 个文件，包含五个轨迹数据文件及来源/坐标核验资料，不下载视频。原始数据保存在 `data/raw/`，经统一字段后的数据保存在 `data/processed/`。这两个目录由脚本生成，不加入 Git；图、结论、统计和复现代码随仓库发布。所有处理均可在数据下载完成后离线运行。

`--download` 对已有文件只核验哈希；文件被改动会报错，不会静默覆盖。缺少数据时，可单独运行 `python -m traj_course download`。使用 `python -m traj_course verify` 进行纯离线校验。上述命令默认在仓库根目录执行；`scripts/run_week01.py` 也支持从其他工作目录调用。

## 目录

```text
traj-jiangyujie/
├── README.md                  # 安装、结果入口和运行方法
├── pyproject.toml             # 可安装的 Python 包与命令行入口
├── requirements.txt           # 本次验证使用的依赖版本
├── CITATION.cff                # 课程代码引用元数据
├── NOTICE.md                  # 来源与授权范围
├── configs/week01.json        # 五场景、帧率、断线规则、图像分辨率
├── data/manifest.json         # 固定源 URL、提交、大小、SHA-256
├── data/raw/                  # 本地下载的原始文件（Git 忽略）
├── data/processed/            # 统一字段的数据（Git 忽略）
├── docs/                      # 数据说明、代码复用、方法、提交说明
├── src/traj_course/           # 下载 / 解析 / 几何 / 绘图 / 报告 / CLI
├── scripts/run_week01.py      # 从任意目录运行的复现入口
├── tests/                     # 解析、身份、时间及数据完整性测试
├── week01/                    # 本周图、结论和机器可读结果
└── week02/ … week08/           # 后续周次目录，尚未提交作业
```

## 方法说明

ETH 的原生点间隔为 0.4 s；UCY 是不等间隔轨迹控制点，图中的直线是几何连接，不是额外观测。默认断开 UCY 超过 4 s 的控制点间隔，完整保留原数据与被断开区间清单。总览颜色表示场景内部相对时间，五个子图空间尺度一致，坐标原点各自保留。总览叠加了不同时刻的路径，不能据交叉线直接判断避让、碰撞或密度。

代码复用了已有轨迹工作台按 ID 分组的绘图逻辑，以及 PEM-Flow 论文中的样式和连续段处理，并做了面向 ETH/UCY 的独立适配。[复用记录](docs/REUSE.md)列出原文件哈希和具体函数，不依赖原工作台、本机硬编码路径或未公开实验数据。

更多细节：[数据卡与字段](docs/DATA_CARD.md) · [本周报告](week01/README.md) · [Git 提交与发布](docs/GITHUB.md)。

## 数据来源

使用 [erichhhhho/DataExtraction](https://github.com/erichhhhho/DataExtraction) 的固定版本 `74006729b1bafafa4f9530150af3ba282b1b0ef3`。该仓库提供 ETH/UCY 的格式转换结果；本项目不是原始数据采集方。来源文件与校验值均列入 [manifest](data/manifest.json)。
