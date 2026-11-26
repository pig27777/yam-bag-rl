# Yam_B 迁移测试报告

2026-11-27 01:14（北京时间），`ubun`。使用指定的 Python 3.14.7、MuJoCo/MJWarp 4.0.1、Warp 1.19.0、Torch 2.15.0+cpu。全部门禁从新工程根目录运行。

```bash
scripts/run_py.sh scripts/verify_migration.py
```

| 门禁 | 结果 | 实测 |
| --- | --- | --- |
| 零旧源码/资产依赖 | PASS | 递归导入本地包，检查 `sys.path`、`sys.modules`、静态导入与 Python/XML/YAML/JSON 路径。第三方库来自指定解释器环境。 |
| Yam_B 资产、FK、控制与重力偏置 | PASS | 原始 MJCF 引用的 15 个 STL 可独立编译。ABC 双相机、正运动学与 40 ms 控制步通过。两个姿态的六轴重力偏置与直接编译 MJCF 在 `dq=0` 的 `qfrc_bias` 一致。 |
| 双柔索复放 | PASS | 65 g 和 250 g 各 4,800 步。16 和 18 个轨迹数组逐元素最大误差均为 0，门禁为 `atol=1e-6, rtol=0`。悬停柔索净力 0.637638927 / 2.452421665 N。峰值净空 164.723158 / 262.172997 mm。 |
| Policy V3 回归 | PASS | `test_v3.py` 的 12 项和 `test_yam_substep.py` 的 4 项共 16/16 通过。另检查了 TCN-FiLM 的有限推理输出与归一化 checkpoint 缓冲。 |

首轮验真受并行工作流切换解释器影响，恢复指定解释器后复验。工位检查定位 ABC 相机名称，保留模型已有相机。原有 Flex 掩码测试改为三顶点测试夹具，项目内未带入高保真袋膜。

原仓库用 `GIT_OPTIONAL_LOCKS=0 git status --short` 与四个冻结参考及同一工作起点的路径级 `git diff --exit-code` 检查。迁移前后的状态快照逐字节一致，Git index 哈希相同，冻结检查 5/5 通过。其他会话在新工程目录并行创建的虚拟环境和高并发实验文件在本次基准提交范围之外。

本地原始记录位于 `runs/migration/`。Stage A 的公开复核记录见 [回归日志](stage-a/regression-check.log)。
