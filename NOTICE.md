# 来源与授权说明

第二周 `week02/model/orca.py` 改编自 UNC 的 RVO2 `Agent.cc`（提交 `1c25b27258d191e26c2df83ff6e31abc8efbdaed`），按 Apache-2.0 使用，版权标识保留在源码，许可证见 [RVO2-LICENSE](week02/model/RVO2-LICENSE.txt)。改动包括 Python/NumPy 移植、全体邻居约束及可行性诊断。个体侧偏偏好和评估流程为本课程的扩展，不是原始 ORCA 自带行为规则。

圆环 TXT 为用户提供的课程数据，随本次课程提交保留；原始数据权利归相应权利人，不因本仓库公开或 RVO2 的代码许可证而取得新的数据许可。实验设置与帧率依据 Xiao 等（2018），坐标单位依据初始几何尺度核对，具体说明见第二周 README。

本课程代码包含本次编写的 ETH/UCY 适配器，以及作者既有轨迹工程和 PEM-Flow 制图逻辑的通用部分。版权归各自作者所有；本次发布未代作者指定新的开源许可证。

ETH/UCY 数据及格式转换文件来自 [erichhhhho/DataExtraction](https://github.com/erichhhhho/DataExtraction/tree/74006729b1bafafa4f9530150af3ba282b1b0ef3)，权利归原始作者及相应权利人。本仓库不对原数据授予新许可，仅提供固定版本的下载链接及校验值。轨迹图来自本次 ETH/UCY 分析，不包含原论文的私有轨迹或现场照片。
