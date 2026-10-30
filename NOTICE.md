# 来源与授权说明

第七周独立实现速度型人工势场、三维有限候选速度障碍物和一阶联合屏障速度优化，参考资料、原论文与本次实现的区别及 AI 辅助范围见 [week07/README.md](week07/README.md)。未复制外部模型源码；通过 Python 包调用 OSQP 与 SciPy。球面初态与三维轨迹均为课程合成数据，制图复用第一周样式。该实验不建质量、力、推力或加速度动力学，不构成真实无人机飞行验证。

第六周依据社会力、IDM 和 ORCA 的文献独立编写统一交互适配与几何诊断，复用第二周 RVO2/ORCA 线性规划求解器、圆环读取和评估，以及第一周绘图样式。社会力圆形简化、IDM 方向扩展、矩形支撑近似及通路/安全干预均为本次课程改动，不等同于原作者的原始完整模型。RVO2 的 Apache-2.0 许可证和来源标识保持不变；真实圆环数据仍引用第二周原文件。合成人车场景没有实测真值。原始资料、修改范围、公式单位、AI 辅助、失败和验证边界见 [week06/README.md](week06/README.md)。

第五周根据 Bando 等（1995）、Jiang 等（2001）与 Treiber 等（2000）的原始论文独立实现 OVM、FVDM 和 IDM 公式，未复制第三方跟驰代码。完整书目信息、原文链接、SI 适配与数值边界说明见 [week05/README.md](week05/README.md)。使用第四周同源 NGSIM SI 数据，随附可核验的稳定跟驰对、原始来源哈希及候选筛选记录；原始论文 PDF 仅用于本地核对，仓库提供来源链接。制图复用既有 week01 样式，AI 辅助范围已说明。

第四周使用教师提供的 NGSIM I-80 SI 数据包，原始来源为 [USDOT NGSIM](https://data.transportation.gov/d/8ect-6jqj)。仓库随附可复算的第 2 车道支持子集，保留源文件和子集 SHA-256、原教师元数据、筛选规则以及已发现的元数据不一致说明。课程数据来源及原有权利范围不因本次提交而改变。本周独立实现 Edie、断面、一维 Voronoi 与核测量，引用、工作台复用和 AI 辅助范围见 [week04/README.md](week04/README.md)。

第三周基于教师提供的 TrajNet++ 课堂包，保留 EPFL VITA LSTM 代码的 MIT 许可证及逐文件哈希。网格信息屏蔽和邻域范围对照为课程改动；来源、依赖变化和 AI 辅助说明见 [week03/model/SOURCE.md](week03/model/SOURCE.md)。第三周的 NDJSON 是同一课程圆环 TXT 的既有转换，数据权利范围沿用原提供方约定。

第二周 `week02/model/orca.py` 改编自 UNC 的 RVO2 `Agent.cc`（提交 `1c25b27258d191e26c2df83ff6e31abc8efbdaed`），按 Apache-2.0 使用，版权标识保留在源码，许可证见 [RVO2-LICENSE](week02/model/RVO2-LICENSE.txt)。改动包括 Python/NumPy 移植、全体邻居约束及可行性诊断。个体侧偏偏好和评估流程为本课程的扩展，不是原始 ORCA 自带行为规则。

圆环 TXT 为用户提供的课程数据，随本次课程提交保留；原始数据权利归相应权利人，不因本仓库公开或 RVO2 的代码许可证而取得新的数据许可。实验设置与帧率依据 Xiao 等（2018），坐标单位依据初始几何尺度核对，具体说明见第二周 README。

本课程代码包含本次编写的 ETH/UCY 适配器，以及作者既有轨迹工程和 PEM-Flow 制图逻辑的通用部分。版权归各自作者所有；本次发布未代作者指定新的开源许可证。

ETH/UCY 数据及格式转换文件来自 [erichhhhho/DataExtraction](https://github.com/erichhhhho/DataExtraction/tree/74006729b1bafafa4f9530150af3ba282b1b0ef3)，权利归原始作者及相应权利人。本仓库不对原数据授予新许可，仅提供固定版本的下载链接及校验值。轨迹图来自本次 ETH/UCY 分析，不包含原论文的私有轨迹或现场照片。
