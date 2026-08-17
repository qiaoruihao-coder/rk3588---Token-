# 执行人组 1.1 模块 —— 输出Token强制控制 + 动态调控

> 作者：乔瑞浩、张倬铭
> 位置：L1 执行人层
> 状态：可交付

---

## 目录

- [文件夹结构](#文件夹结构)
- [快速开始](#快速开始)
- [五个 API 函数](#五个-api-函数)
- [动态控制系统](#动态控制系统)
- [上板部署步骤](#上板部署步骤)
- [常见问题](#常见问题)

---

## 文件夹结构

```
揭榜挂帅/
│
├── 📦 核心模块
│   ├── sglang_1_1_module.py            1.1 模块主文件（RK3588 版，已适配）
│   ├── sglang_1_1_module_WSL_备份.py    WSL/GPU 开发阶段验证代码（仅参考）
│   └── sglang_1_1_ollama.py            Ollama 版（边缘低功耗）
│
├── 🧠 动态控制（新增）
│   ├── dynamic_control/
│   │   ├── __init__.py             包入口
│   │   ├── config.py               阈值配置
│   │   ├── monitor.py              指标采集（温度/频率/内存/NPU）
│   │   ├── evaluator.py            状态机 + 滞回控制
│   │   ├── actuator.py             调控执行（调频/切后端/调参）
│   │   ├── backends.py             推理后端（llama.cpp 约束解码 + CPU 兜底）
│   │   └── controller.py           统一入口 DynamicController
│   ├── test_dynamic_control.py     集成测试（8 用例）
│   └── demo_dynamic_control.py     场景演示脚本
│
├── 📐 架构图
│   └── architecture_diagrams.html  架构图/流程图/决策表（浏览器打开，截图贴 PPT）
│
├── 🧪 测试（原有）
│   ├── test_11_module.py           单元测试（28 用例）
│   ├── final_check.py              验收测试（33 项检查）
│   └── quick_test.py               一键演示
│
├── 📄 文档（原有）
│   ├── API文档_1.1模块.md          接口文档
│   ├── 1.1模块_答辩PPT.pptx        答辩 PPT
│   ├── 1.1模块_进度汇报.docx        个人进度报告
│   └── 执行人组1.1_联合进度汇报.docx 双人联合报告
│
└── 🔧 工具（原有）
    └── run_11_module.sh            WSL 一键启动
```

---

## ⭐ 约束解码后端（核心任务 1.1 的正式实现）

老师 1.1 要求"**依托底层正则解码算子，在机床(采样/token)层面控制输出，强制只出 JSON/Diff/纯结论，熔断废话**"。
为此 `dynamic_control/` 提供两个**真正的 token 层硬约束解码**后端（不是事后校验）：

| 后端 | 文件 | 机制 | 适用 |
|------|------|------|------|
| **llama.cpp** | `llamacpp_backend.py` | JSON-schema → grammar，采样层硬锁合法 JSON | **RK3588 正式部署（主）** |
| **Ollama** | `ollama_backend.py` | 原生 `format=json/schema` structured outputs | RK3588 备选 / 快速调试 |
| CPU(规则) | `backends.py` | 兜底，非真推理 | 无模型时保底 |

`DynamicController.generate_json` 默认走 llama.cpp 约束解码（`structured=True`），
由后端在生成时锁死格式，彻底熔断"推理过程/自然语言废话"，只返回合法 dict。

> 旧 `RKNBackend`（rknnlite）已弃用：`rknnlite.inference()` 是图像单次前向 API，
> 不是 LLM 工具链，无法做约束解码。正式部署改用上述约束解码后端。

---

## 三套版本定位表（部署优先级）

仓库里的三个入口**对外 API 完全一致**（`generate_json` / `generate_label` / `generate_value` / `generate_diff`），
调度员/上层不关心底层用哪个。**部署优先级**如下：

| 版本 | 入口 | 底层 | 跑在哪 | 定位 | 是否用于 RK3588 正式部署 |
|------|------|------|--------|------|--------------------------|
| **RK3588 版（主）** | `sglang_1_1_module.py` / `dynamic_control.DynamicController` | llama.cpp 约束解码（主）+ Ollama（备）+ CPU 兜底 | **RK3588（4GB，离线）** | **正式部署，第一优先级** | ✅ 是 |
| **Ollama 版** | `sglang_1_1_ollama.py` | Ollama 原生 JSON/structured 约束 | RK3588(ARM Ollama) 或任意有 CPU 的机器 | RK3588 的**备选后端**（已并入 dynamic_control 的 `ollama_backend`）；也可独立快速调试 | ⚠️ 备选 |
| **GPU / WSL 版** | `sglang_1_1_module_WSL_备份.py` | GPU + `outlines`（底层正则/FSM 约束解码） | x86 + N 卡 | **开发参考 / 答辩演示**：验证约束解码内核；生产（RK3588）**不用 GPU**。若团队选择走"GPU 跑真 SGLang/outlines"方案才启用 | ❌ 否（RK3588 不用 GPU） |

> - **RK3588 正式部署无 GPU**（板子 4GB，只用 llama.cpp/Ollama 约束解码）。
> - **GPU 版保留**：① 演示约束解码的机制内核；② 若未来改走 GPU 方案（老师架构里的"边缘 GPU 节点"）可用。
> - 三套同 API，切换只需换 import，不动上层逻辑。

---

## 快速开始

### 1.1 模块主文件（RK3588，正式部署用）

```python
# 一行导入，自动初始化 NPU + 动态控制
from sglang_1_1_module import generate_json, generate_label

result = generate_json("输出学生张三的信息", schema)
# -> {'name': '张三', 'age': 20}

# 查询系统状态
from sglang_1_1_module import get_system_status, get_health
print(get_health())  # {'level': 'normal', 'backend': 'npu', ...}
```

**环境变量**（可选）：
```bash
# 指定 llama.cpp 服务地址 / 模型（默认 127.0.0.1:8080，自动取第一个模型）
export LLAMACPP_BASE_URL=http://127.0.0.1:8080
# 若用 Ollama 备选：
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:1.5b
export RK3588_TEMP_WARN_C=60                     # 自定义温度阈值
```

> 部署（llama-server/Ollama 安装+启动+约束解码验证）见 **`SETUP_RK3588.md`**。

### Ollama 版（开发调试用）

```python
from sglang_1_1_ollama import generate_json, generate_label

result = generate_json("输出学生张三的信息", schema)
# -> {'name': '张三', 'age': 20}

label = generate_label("产品质量?", ["qualified", "defective"])
# -> 'defective'
```

**前置条件**：Ollama 已安装并拉取了 `qwen2.5:1.5b`。

### 动态控制版（RK3588，正式部署用）

```python
from dynamic_control import DynamicController

ctrl = DynamicController()
ctrl.init()

# API 完全兼容
result = ctrl.generate_json("输出张三信息", schema)
# -> {'name': '张三', 'age': 20}

# 查询系统状态（给调度员用）
status = ctrl.status()
# -> {"system": {...}, "decision_level": "normal", "backend": "npu", ...}

# 轻量健康检查
health = ctrl.health()
# -> {"level": "normal", "backend": "npu", "mem_avail_gb": 1.5, ...}
```

---

## 五个 API 函数

| 函数 | 功能 | 示例 |
|------|------|------|
| `generate_json(prompt, schema)` | JSON Schema 约束 | `{"name":"张三","age":20}` |
| `generate_json_pydantic(prompt, model)` | Pydantic 类型安全 | 同上 + 类型校验 |
| `generate_label(prompt, labels)` | 分类标签约束 | `"defective"` |
| `generate_value(prompt, schema)` | 数值+结论约束 | `{"area":12.5,"quality":"defective"}` |
| `generate_diff(prompt, old_code)` | 代码 Diff 补丁 | `--- a/code\n+++ b/code\n...` |

三个版本（`sglang_1_1_module.py` / Ollama / DynamicControl）API 完全一致，上层调用者无需关心底层用哪个。

### 场景模板

工业检测：
```python
SCHEMA_INDUSTRIAL = {
    "type": "object",
    "properties": {
        "defect_type": {"type": "string", "enum": ["crack", "scratch", "dent", "none"]},
        "area_mm2":    {"type": "number"},
        "severity":    {"type": "string", "enum": ["minor", "moderate", "critical"]},
        "quality":     {"type": "string", "enum": ["qualified", "defective"]}
    },
    "required": ["defect_type", "area_mm2", "severity", "quality"]
}
```

交通监控：
```python
SCHEMA_TRAFFIC = {
    "type": "object",
    "properties": {
        "vehicle_count": {"type": "integer"},
        "avg_speed_kmh": {"type": "number"},
        "status":        {"type": "string", "enum": ["smooth", "slow", "congested", "blocked"]},
        "action":        {"type": "string", "enum": ["none", "adjust_timing", "alert_police"]}
    },
    "required": ["vehicle_count", "avg_speed_kmh", "status", "action"]
}
```

---

## 动态控制系统

### 设计目标

RK3588 三核异构 (CPU + GPU + NPU)，4GB 内存。根据 NPU 使用率、芯片温度、内存占用，**实时自动**调整推理后端、上下文长度、max_tokens、NPU 频率。

### 架构

```
调度员 (2.1) ──调用──▶ DynamicController
                         ├── Monitor   (采集: 温度/频率/内存/NPU)
                         ├── Evaluator (决策: 4级状态机 + 滞回)
                         ├── Actuator  (执行: 调频/切后端/调参)
                         └── Backend   (推理: llama.cpp 约束解码 / CPU 兜底)
```

**每次请求自动执行** 采集→决策→执行→路由→推理，前4步 < 1ms。

### 四级状态机

```
IDLE ──(NPU<10%)──▶ NORMAL ──(NPU>90% 或 温度>65°C)──▶ WARNING ──(温度>75°C 或 内存<300MB)──▶ CRITICAL
  ▲                    ▲                                  ▲                                      │
  └────(30s 稳定)──────┴────────(30s 恢复)────────────────┴────────────(60s 恢复)──────────────────┘
```

- **升级即时生效**（不等待，立刻保护硬件）
- **降级需稳定 N 秒**（防抖动，避免在阈值附近反复横跳）

### 决策对照表

| 场景 | NPU% | 温度 | 可用内存 | 等级 | 后端 | Context | Tokens | NPU频率 |
|------|------|------|---------|------|------|---------|--------|---------|
| 正常 | 40% | 45°C | 1.5 GB | NORMAL | NPU | 2048 | 200 | high |
| NPU 过载 | 95% | 55°C | 1.0 GB | WARNING | NPU | 1024 | 100 | mid |
| 温度告警 | 50% | 68°C | 0.8 GB | WARNING | NPU | 614 | 100 | mid |
| 内存不足 | 20% | 50°C | 0.2 GB | CRITICAL | CPU | 512 | 50 | low |
| CPU 过热 | 10% | 86°C | 1.0 GB | CRITICAL | refuse | — | — | — |

### 默认阈值

| 参数 | 值 | 环境变量 |
|------|-----|---------|
| 温度告警 | 65°C | `RK3588_TEMP_WARN_C` |
| 温度临界 | 75°C | `RK3588_TEMP_CRIT_C` |
| 内存告警 | 0.6 GB | `RK3588_MEM_WARN_GB` |
| 内存临界 | 0.3 GB | `RK3588_MEM_CRIT_GB` |
| NPU 过载 | 90% | `RK3588_NPU_UTIL_WARN_PCT` |
| 滞回稳定时间 | 30s | `RK3588_STABILITY_SEC` |

```bash
# 板子散热差，收紧阈值
export RK3588_TEMP_WARN_C=55
export RK3588_TEMP_CRIT_C=65
```

### 架构图

打开 `architecture_diagrams.html` 浏览器查看，7 张图可直接截图贴 PPT：

1. 系统三层架构
2. 一次请求完整链路
3. 四级状态机 + 滞回控制
4. 6 场景决策对照表
5. 滞回有无对比
6. 代码结构
7. 动态 vs 静态改善指标

---

## 上板部署步骤

> **完整、权威、可执行的部署细节见 [`SETUP_RK3588.md`](SETUP_RK3588.md)**
> （llama-server / Ollama 安装、启动、GGUF 模型、约束解码验证、快速排查）。
> 本处只给**流程总览**，避免与 SETUP 内容重复导致不一致。

| 步骤 | 做什么 | 去哪看 |
|------|--------|--------|
| 1 | 确认板子 sysfs 路径（温度/频率/内存） | `SETUP_RK3588.md` §3 |
| 2 | 装 Python 依赖 `psutil pydantic` | `SETUP_RK3588.md` §2 |
| 3 | 部署约束解码推理：**llama.cpp（主）/ Ollama（备）** | `SETUP_RK3588.md` §4 |
| 4 | 准备 GGUF 模型（Qwen2.5-1.5B q4_k_m） | `SETUP_RK3588.md` §4 |
| 5 | 验证 Monitor 采集 + 跑约束解码验证 | `SETUP_RK3588.md` §5 |
| 6 | 动态控制验证 / 联调 / 快速排查 | `SETUP_RK3588.md` §6 / 快速排查 |

快速验证命令（完整版见 SETUP）：

```bash
# 1. 服务可达
curl http://127.0.0.1:8080/v1/models        # llama.cpp
# 或
curl http://127.0.0.1:11434/api/tags        # Ollama

# 2. 约束解码（只出合法 JSON，无废话）
python -c "from dynamic_control import DynamicController; c=DynamicController(); c.init(); print(c.generate_json('输出学生张三,20岁,计算机专业',{'type':'object','properties':{'name':{'type':'string'},'age':{'type':'integer'},'major':{'type':'string'}},'required':['name','age','major']}))"
# 预期: {'name': '张三', 'age': 20, 'major': '计算机'}
```

> JSON-schema 硬约束由**后端**（llama.cpp GBNF grammar / Ollama structured outputs）
> 在**机床(token采样)层面**完成，**不会**夹带"推理过程/自然语言废话"。
> 旧 `RKNBackend`（rknnlite 单次前向）非 LLM 工具链，已弃用。

---

## 常见问题

### Q: Monitor 返回全是 0/None

sysfs 路径不对。按 `SETUP_RK3588.md` §3 确认 thermal zone 和 devfreq;不同板子编号不同,把输出发我。

### Q: 连不上约束解码服务

- llama.cpp：确认 `curl http://127.0.0.1:8080/v1/models` 有响应；地址不符设 `LLAMACPP_BASE_URL`
- Ollama：确认 `curl http://127.0.0.1:11434/api/tags`；地址不符设 `OLLAMA_BASE_URL`
- 服务版本老、不支持 `response_format`/`format`：换最新 llama.cpp / Ollama

### Q: 推理时 OOM

4GB 跑 Qwen2.5-1.5B + 系统开销很紧。尝试：
- 用更小的量化（q4_k_m）+ 更小的模型（1.5B 而非 3B）
- 减小 context：llama-server 的 `-c 1024`，或 Ollama `num_ctx`

### Q: 如何强制指定后端（调试用）

```python
# 强制用 llama.cpp
result = ctrl.generate_json(prompt, schema, force_backend="npu")
# 强制用 Ollama
result = ctrl.generate_json(prompt, schema, force_backend="ollama")
# 强制用 CPU 规则兜底
result = ctrl.generate_json(prompt, schema, force_backend="cpu")
```

### Q: 状态频繁切换（抖动）

增大滞回稳定时间：
```bash
export RK3588_STABILITY_SEC=60
```

---

## 比赛指标达成

| 指标 | 要求 | 动态控制（RK3588 NPU） | Ollama 版（ARM CPU） |
|------|------|----------------------|---------------------|
| 内存 | ≤1.5 GB | < 1 GB (NPU int8) ✅ | 0.14 GB ✅ |
| 输出干净率 | 100% | 100% | 100% |
| 成功率 | ≥95% | 95%+ | 90% |
| 崩溃率 | 0% | 0%（有过热保护） ✅ | 0% |
| 动态调控 | — | ✅ 温度/频率/内存/后端 | — |
| 覆盖场景 | ≥2 | 工业+交通+动态降级 | 工业+交通 |

---

## 依赖

| RK3588 主文件 | Ollama 版 | WSL 备份 |
|-------------|----------|---------|
| Python 3.10+ | Python 3.14 + Ollama | Python 3.12 + CUDA 12.9 |
| llama.cpp(llama-server) 或 Ollama + psutil + pydantic | openai + pydantic + psutil | torch + transformers + outlines |
| Qwen2.5-1.5B (.gguf, q4_k_m) | Qwen2.5-1.5B (Ollama q4) | Qwen2.5-1.5B (HuggingFace) |

> **部署详细步骤见 `SETUP_RK3588.md`**（llama-server / Ollama 安装 + 启动 + 约束解码验证）。

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 7/16 | Day 1-2：环境搭建 + 核心函数 |
| 7/17 | Day 3：异常处理 + 场景测试 |
| 7/18 | Day 11-14：Ollama 部署，内存达标 |
| 7/19 | Day 15-22：单元测试 + 文档 + PPT |
| 8/11 | Day 23+：动态控制系统架构 + 实现 + 架构图 |
| 8/17 | 部署路线对齐：RK3588 用 llama.cpp/Ollama 约束解码；README 上板步骤改为总览并统一指向 SETUP |
| 8/17 | 完善动态控制(模型档位动态选择) + README 新增三套版本定位表 |
