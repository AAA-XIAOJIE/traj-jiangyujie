# ETH / UCY A1 数据卡

## 任务特征

数据已包含逐人的轨迹标注，不需要 YOLO 检测、相机内参重估或重新跟踪。本周只回答“有哪些轨迹、怎样组织、如何正确画出路径”。每个 ID 只在所在场景内有意义，跨场景不能直接合并为同一个人。

固定源仓库：[DataExtraction @ 74006729](https://github.com/erichhhhho/DataExtraction/tree/74006729b1bafafa4f9530150af3ba282b1b0ef3)。

## 五个场景

- `seq_eth` → 源目录 `seq_eth`：360 个轨迹 ID、8,908 个观测点；源帧间隔为 6，对应 0.4 s，帧索引时间基准为 15 fps。
- `seq_hotel` → 源目录 `seq_hotel`：390 个轨迹 ID、6,544 个观测点；源帧间隔为 10，对应 0.4 s，帧索引时间基准为 25 fps。
- `zara01` → 源目录 `zara01`：148 个轨迹 ID、1,520 个不等间隔控制点；原始 `crowds_zara01.vsp`。
- `zara02` → 源目录 `zara02`：204 个轨迹 ID、2,375 个不等间隔控制点；原始 `crowds_zara02.vsp`。
- `students03` → 源目录 `univ`：434 个轨迹 ID、5,779 个不等间隔控制点；原始 `students003.vsp`。课程图片中的 students03 与此处名称映射明确保留。

UCY 时间基准为 25 fps；控制点并不是每 0.4 s 一条记录。不可把控制点行号除以 2.5 当作秒数，也不使用上游带 `adjustedFrameID`、`normalized`、`inter` 后缀的再加工坐标文件。

## 字段

原始 `world_coordinate.csv` 为无表头 **4 × N** 矩阵。四行依次为源帧号、轨迹 ID、世界 y、世界 x。标准化后的 N 行表格为：

- `scene`：课程场景名。
- `frame_id`：原始帧索引，整数；不重编号。
- `track_id`：场景内的个体编号，整数。
- `x_m`、`y_m`：原世界坐标，单位 m。
- `time_s`：`(frame_id - min_scene_frame) / fps`，以首个场景标注为零点。

坐标与原数据一致，不作 min–max 归一化、不交换 x/y、不旋转、不按 ROI 筛选。输入非有限值、非整数 ID、重复 `(track_id, frame_id)` 会报错；不会静默删行。

## 时间与连线

ETH 仅连接符合原生 6/10 帧间隔的相邻点。UCY 仅连接同 ID、时间递增且间隔不大于 4 s 的控制点。长间隔仍保留在原始数据和标准化数据中，只在图上断线。4 s 是显示策略，不意味着每个断点都属于跟踪故障；2/4/8 s 的断线数量另存敏感性 CSV。孤立点仍显示。

细节图使用 12 s 窗口：从整秒候选起点中，选取含有效连线个体最多的窗口，并列取最早。仅对已接受的相邻点连线做时间边界裁剪，不向端点外推，不跨被断开的区间插值。窗口选择和个体数保存于 `scene_summary.json`。

全时段图的 1,536 指五个场景的场景内 ID 数之和，不是经过跨镜头身份去重后的独立自然人数。观察覆盖时间为首末标注时间差，不代表视频的完整长度。显示范围为统一的 26 × 20 m 坐标跨度，只为读图，不作为密度 ROI。

## 来源证据与交叉核验

- [ETH info.txt](https://github.com/erichhhhho/DataExtraction/blob/74006729b1bafafa4f9530150af3ba282b1b0ef3/seq_eth/info.txt)与[Hotel info.txt](https://github.com/erichhhhho/DataExtraction/blob/74006729b1bafafa4f9530150af3ba282b1b0ef3/seq_hotel/info.txt)：0.4 s 标注间隔。ETH 的 15 fps 由 6 帧/0.4 s 推得，不能误套 Hotel 的 25 fps。
- [UCY 解析脚本](https://github.com/erichhhhho/DataExtraction/blob/74006729b1bafafa4f9530150af3ba282b1b0ef3/univ/ExtractDataandInterpolation.m)：明确读取 `students003.vsp`，保留控制点帧号，并给出像素/世界坐标列约定。
- [CMU-TBD 数据加载器](https://github.com/CMU-TBD/social_group/blob/master/data_loader.py)：提供 ETH 15 fps / UCY 25 fps 的独立代码佐证，以及 `students003` 的序列映射。
- ETH 与源 `obsmat.txt` 第 3、5 列逐键核对；UCY 由 `pixel.csv` 和 `H.txt` 重投影并做齐次归一化核对。允许的 0.002 m 差异用于兼容上游 CSV 的数值舍入，不代表已验证的物理测量精度。

## 复现与使用范围

所有 25 个源文件的固定提交、字节数、SHA-256 记录在 `data/manifest.json`。下载器遇到哈希不符立即报错。原始数据本地下载，不随 Git 再分发。仓库内派生图表保留来源署名；上游数据权利仍归原作者，详见 [NOTICE](../NOTICE.md)。
