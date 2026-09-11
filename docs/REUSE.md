# 现有工作台代码复用记录

本课程代码是从现有轨迹分析工程和 PEM-Flow 论文制图代码中提取通用逻辑后形成的独立包。原工作台文件保持原样。

## 实际采用的代码

- 原 `visualization.py::plot_trajectory_results` 的按 `id` 分组、每个个体独立绘制世界坐标路径的结构，适配为 `analysis.py::build_geometry` 与 `plotting.py::draw_all`。新增显式时间排序、轨迹 ID 隔离和无交叉连接约束。
- 原 `重绘_fig8_轨迹总览.py::contiguous_groups` 的“按帧号排序 → 检查相邻帧 → 分段”实现结构，迁移到 `analysis.py::contiguous_groups`。ETH 改用源数据的 6/10 帧间隔；UCY 因控制点不等间隔，采用可配置时间上限。
- 原 `pemflow_style.py::apply`、`panel_label` 的统一 rcParams、Arial 字体、白底、轴脊控制、TrueType PDF 字体与字母分面标签，适配至 `style.py`。保留论文行人蓝色 `#0072B2` 作为强调色。
- 原论文总览图的 `LineCollection`、时间颜色编码、独立时间色条与 PNG/PDF/SVG 导出策略，迁移至 `plotting.py`。将固定 G08/30 fps 数据依赖替换为五场景配置；时间色板采用 viridis。

`show_trajectories.py` 用于确认原工程的世界坐标与等比例坐标语义；其随机散点采样和交互 ROI 框选不适用于本次“每个人一条线”的 A1 交付，因此没有把该脚本直接作为主入口。

## 原文件核验

源文件的 SHA-256 和复用角色记录在 `reuse_manifest.json`。该清单只记录文件名与哈希，不包含本机磁盘路径。包中只保留完成本课程所需的通用实现，不导入原工程的私有数据、绝对路径、相机参数、车辆类别或现场照片。

本任务没有运行混合交通 Voronoi、流量估计或训练模型，也没有将已有论文的结论搬到 ETH/UCY。课程结论来自本次下载并核验的数据。
