# Yam_B 双柔索代理与 Policy V3

独立的仿真和策略组件工程。包含 Yam_B 机械臂、四块双柔索包裹代理、GPU 批量环境和 CPU Policy V3 学习器。项目使用 `scripts/run_py.sh` 统一启动。

- [Policy V3 技术报告](reports/v1_rl_technical_report.md)包含模型结构、动作、观测、奖励和留出结果。
- [Stage A 运行产物](reports/stage-a/README.md)列出训练数据、逐轮缓存、模型、视频和 Python 环境快照。

## 启动与验真

在仓库根目录运行以下命令。启动脚本使用本工程 `src/` 作为 `PYTHONPATH`，设置 OMP、OpenBLAS、MKL 为单线程，并保留调用者的 `CUDA_VISIBLE_DEVICES`。

```bash
scripts/run_py.sh --bootstrap -m venv .venv
scripts/run_py.sh -m pip install mujoco==4.0.1 mujoco-warp==4.0.1 warp-lang==1.19.0 matplotlib imageio-ffmpeg pytest
scripts/run_py.sh -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
scripts/run_py.sh scripts/verify_migration.py
```

```bash
mkdir -p runs/replay
scripts/run_py.sh -m yam_bag_rl.bag.rope_replay \
  tests/fixtures/neck_98mm_input.npz runs/replay/neck_98mm.npz
scripts/run_py.sh -m yam_bag_rl.bag.proxy_contact \
  tests/fixtures/center_250g_input.npz runs/replay/center_250g.npz
```

`yam_bag_rl.envs.make_smoke_env()` 提供刚体夹具上的 `reset`、13D `execute_probe`、11D 候选/抓取以及结果接口。`robot.yam_model.nominal_gravity_bias(q)` 用标称 MJCF 和零速度求六轴 `qfrc_bias`，单位 N·m。Policy V3 默认使用拼接 Scorer。高并发代理工作流显式使用 TCN-FiLM。

```bash
scripts/run_surrogate_v3.sh
```

此入口从 C0 运行至留出评估，产物写入 `runs/surrogate-v1-large-scale/`。调试某一阶段时使用 `--stage c0`、`--stage scaling`、`--stage c1` 或 `--stage train`。PPO 续训使用 `--stage train --resume runs/surrogate-v1-large-scale/policy_round_NNN.pt`。

周一版本以虚拟柔索直接向四块内容物施力。本工程验证复放等价性，模型仍属于低阶代理近似。刚体夹具的控制与抓放结果只覆盖软件回归。验真命令不启动 RL 训练。

## 目录

```text
assets/yam/            # 原厂 MJCF、15 个引用 STL、上游 MIT 许可
configs/workstation/   # ABC、名义控制与执行器配置
configs/surrogate/     # 2 ms 双柔索参数、65 g 与 250 g 工况
src/yam_bag_rl/
  robot/               # 装配、FK、CPU/MJWarp 控制与静重力偏置
  bag/                 # 冻结方程、两个周一复放入口
  policy_v3/           # 宏动作、历史、评分器、PPO、评估与工作器
  envs/                # GPU 四块双柔索环境、高并发运行入口和刚体烟测
scripts/               # 统一 Python 启动和一键验真
tests/                # 自包含的复放数组与回归
reports/               # 技术报告、验证和实测指标
runs/                  # 本地运行产物，Git 忽略
```

## 来源与迁移清单

以下源路径相对于原工程根目录。资产源为 I2RT Robotics 的 MuJoCo Menagerie `i2rt_yam`，归档上游提交 `f054586a8e90465d49ee5be15335c4a0c7f57caf`。MJCF 的唯一改动是相对 `meshdir`，所引用 STL 与 MIT 许可逐字节复制。配置、资产、冻结方程和黄金参考的本地校验值见 [冻结清单](tests/fixtures/frozen_migration.json)。

| 新工程位置 | 源路径及迁移范围 |
| --- | --- |
| `assets/yam/` | `experience/yam_platform/assets/i2rt-yam-menagerie/upstream/yam_linear_4310.xml`、该 XML 引用的 15 个 `assets/**/*.stl` 和 `LICENSE`。 |
| `robot/yam_model.py` | `experience/yam_platform/yam_model.py` 及 `robot.py` 的旋转误差函数。新增静态 `qfrc_bias` 查询。 |
| `robot/abc_workcell.py`、`configs/workstation/abc.json` | `experience/yam_platform/abc_workcell.py`、`src/policy_v3/yam_scene.py` 的篮子装配，以及 `configs/route_b_yam_scene.json`、`route_b_workcell.yaml`、`route_b_single_arm_bin.yaml` 的实际读取字段。 |
| `robot/yam_{interface,history,warp,warp_kernels}.py`、`configs/workstation/yam_{nominal,randomization}.json` | `experience/yam_platform/` 的同名文件。 |
| `robot/kinematics.py`、`robot/controller.py` | `src/policy_v3/yam_kinematics.py` 与 `src/policy_v3/yam_backend.py` 的 `CPURigidDrive`。 |
| `bag/{lumped_contact_proxy,macro_parcel_sim,unified_payload,neck_damping,pad_torsion}.py` | `src/low_order_parcel/` 五个同名文件，字节完全一致。 |
| `bag/proxy_contact.py`、`bag/rope_replay.py` | `test/runs/2026-11-23/bag/real-grasp-skirt-c-r2/proxy_contact.py` 和 `pe-yam-film-comparison-r1/proxy.py`。65 g 复放通过本地名义机械臂重算指垫 FK。 |
| `configs/surrogate/`、`bag/replay_config.py` | `configs/proxy_material_envelopes.yaml` 的 PE 名义参数、`configs/payload_template_X.yaml` 的默认值与 `folded_clothes`，以及两条周一复放的工程参数。 |
| `policy_v3/{data,environment,evaluation,geometry,history,models,perception,training,worker_process,workers,yam_backend,yam_gpu,yam_scorer}.py` | `src/policy_v3/` 的同名文件。`yam_scene.py` 只保留其中的刚体烟测场景。 |
| `tests/policy_v3/` | `tests/policy_v3/test_v3.py` 和 `test_yam_substep.py`。原 Git 冻结检查改用本地 SHA-256。原 Flex 掩码检查使用三顶点原生测试夹具，保留碰撞与坐标系断言。工作器线程测试对齐新启动合约。 |
| `tests/fixtures/` | 周一的 `umi-pe-yam-hold-r1/{q_ref,proxy_episode}.npz` 和 `real-grasp-skirt-c-r2/{q_ref_top_center,proxy_probe}.npz` 的最小输入与黄金轨迹。力和净空黄金值来自 `test/runs/2026-11-26/yam-surrogate-v1-rl/monday-rope-audit.json`。 |

薄板片袋体、旧捏取模型、高保真 Flex 膜和历史视频未纳入基准。迁移结果见 [测试报告](reports/migration-test.md)。
