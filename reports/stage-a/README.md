# Stage A 运行产物

本页对应单卡 RTX 5090 上 16,384 路、30 轮 PPO 的正式运行。核心结果见 [Policy V3 技术报告](../v1_rl_technical_report.md)。

| 仓库内记录 | 内容 |
|---|---|
| [完整运行汇总](summary.json) | C0–C4、留出结果、并发选择与补测 |
| [PPO 逐轮记录](ppo_progress.jsonl) · [学习曲线](c3_c4_learning_curves.png) | 30 轮 PPO、5 次 C4 验证及物理审计 |
| [C1](c1_dataset_summary.json) · [C2](c2_training_metrics.json) | 组切分、BCE、AUROC、校准与线性探针 |
| [留出闭环](held_out_evaluation.json) · [分工况图](closed_loop_by_regime.png) | 200 个初态、7 组闭环、分层配对置信区间 |
| [并发基准](scaling_benchmark.json) | 1,024 至 32,768 路端到端吞吐和显存 |
| [蒙皮同帧对照](yam_surrogate_skin_comparison.png) | 原四块代理和半透明外袋的同轨迹回放 |

## Release 下载

[公开 Release v3-stage-a](https://github.com/pig27777/yam-bag-rl/releases/tag/v3-stage-a)按用途提供压缩包和独立回放。

| 文件 | 内容 |
|---|---|
| `training-run-and-c1-data.tar.zst` | C0 审计和轨迹、C1 原始采集及三份分组数据、checkpoint、指标、视频、可视化和早期尝试记录 |
| `ppo-rollouts-01-05.tar.zst` 至 `ppo-rollouts-26-30.tar.zst` | 六个分包，覆盖正式 30 轮收集后保存的有效 rollout 缓存 |
| `python-environment-linux-x86_64.tar.zst` | 本轮工作目录的 Python 虚拟环境快照 |
| `python-environment-requirements.txt` | 已安装 Python 包和版本 |
| `baseline-replay-inputs.tar.zst` | 迁移阶段两条复放输入 |
| `policy_round_030.pt` | 最终 Actor、Critic、三种评分器、动作掩码和冻结统计 |
| `yam_surrogate_demo.mp4` · `yam_surrogate_skin_demo.mp4` | 原代理与带半透明蒙皮的单路回放 |
| `SHA256SUMS` · `artifact-manifest.json` | 文件校验和与压缩包字节数 |

环境快照用于归档，对应 Linux x86_64，并保留原 Python 路径与符号链接。迁移到其他机器时，从依赖清单和仓库首页说明重建环境。压缩包内采用相对路径，可在新目录解包。解包前可按 `SHA256SUMS` 验证下载完整性。
