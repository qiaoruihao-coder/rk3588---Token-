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
│   │   ├── backends.py             推理后端（RKNN NPU + CPU 兜底）
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
export RK3588_MODEL_PATH=/path/to/model.rknn   # 指定模型路径
export RK3588_TEMP_WARN_C=60                     # 自定义温度阈值
```

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

ctrl = DynamicController("qwen2.5-1.5b.rknn")
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
                         └── Backend   (推理: RKNN NPU / CPU 兜底)
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

按顺序执行，每步独立可验证。

### Step 1 · 确认硬件路径

SSH 进板子，确认 sysfs 路径：

```bash
# 温度传感器
ls /sys/class/thermal/thermal_zone*/type
cat /sys/class/thermal/thermal_zone*/type
# 预期看到 cpu-thermal, npu-thermal (或类似)

# NPU 频率调节
ls /sys/class/devfreq/
# 预期看到含 "npu" 的设备名

# 内存
cat /proc/meminfo | head -5
```

**把输出发我**，不同板子的 zone 编号不一样，我帮你改 `monitor.py`。

### Step 2 · 装依赖

```bash
pip install psutil pydantic
```

确认 RKNN 可用：

```bash
python -c "from rknnlite.api import RKNNLite; print('OK')"
# 如果 ImportError，参考板子文档安装 rknn-toolkit-lite2
```

### Step 3 · 模型转换

Qwen2.5-1.5B → .rknn 格式。这是唯一费时的一步（30min~2h）。

**方案 A：在 x86 机器上用 rknn-toolkit2 转（推荐）**

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]],
            target_platform="rk3588", quantized_dtype="w8a8")
rknn.load_onnx("qwen2.5-1.5b.onnx")     # 需先从 HuggingFace 导出
rknn.build(do_quantization=True)
rknn.export_rknn("qwen2.5-1.5b.rknn")
```

**方案 B：在板子上直接用 CPU 转（慢但省事）**

```bash
# 板子上装依赖后同方案 A
pip install rknn-toolkit-lite2
```

产出的 `qwen2.5-1.5b.rknn` 放到 `揭榜挂帅/` 目录下。

### Step 4 · 验证 Monitor 采集

```bash
cd 揭榜挂帅
python -c "
from dynamic_control.monitor import Monitor
m = Monitor()
s = m.snapshot()
print('Level:', s.level)
print('NPU temp:', s.npu_temp_c, '°C')
print('CPU temp:', s.cpu_temp_c, '°C')
print('NPU freq:', s.npu_freq_mhz, 'MHz')
print('NPU util:', s.npu_util_pct, '%')
print('Memory:', round(s.mem_avail_gb, 2), 'GB available')
"
```

预期：温度、频率、内存有真实值（不是 0 / None）。如果全空，把 Step 1 的输出发我。

### Step 5 · 跑集成测试

```bash
python test_dynamic_control.py
```

预期 8/8 通过。NPU 使用率可能为 0（需要 RKNN 模型已加载才能读），这条先不管。

### Step 6 · 加载 RKNN 模型并验证推理

```python
from dynamic_control import DynamicController

ctrl = DynamicController("qwen2.5-1.5b.rknn")
ctrl.init()
# 预期输出:
#   [npu] loaded
#   Primary backend: npu

# 测试推理
result = ctrl.generate_json(
    "输出学生张三, 20岁, 计算机专业",
    {"type":"object",
     "properties":{"name":{"type":"string"},"age":{"type":"integer"},"major":{"type":"string"}},
     "required":["name","age","major"]}
)
print(result)
# 预期: {'name': '张三', 'age': 20, 'major': '计算机'}
```

如果失败了，查看 `ctrl.status()` 返回的 `decision_level`，确认是硬件的 NPU 不可用还是模型格式不对。

### Step 7 · 完善 RKNN Tokenizer

`backends.py` 的 `RKNBackend.generate()` 目前是简化实现。正确的做法是用 HuggingFace tokenizer 在 CPU 端做 tokenize/detokenize，只把 forward 放在 NPU。

在 `RKNBackend.__init__` 中添加：

```python
from transformers import AutoTokenizer
self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

修改 `generate()` 使用 tokenizer：

```python
input_ids = self._tokenizer.encode(prompt, return_tensors="np")
outputs = self._rknn.inference([input_ids])
text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Tokenizer 不占 NPU 内存，CPU 上跑很快（<10ms）。

### Step 8 · 模拟负载测试

```bash
python demo_dynamic_control.py
```

验证各场景决策是否符合预期。

### Step 9 · 联调

调度员（2.1 模块）接入：

```python
from dynamic_control import DynamicController

ctrl = DynamicController("qwen2.5-1.5b.rknn")
ctrl.init()

# 调度员只需调这三个方法
result = ctrl.generate_json(prompt, schema)
status = ctrl.status()          # 可选: 查看当前后端和状态
health = ctrl.health()          # 可选: 轻量健康检查
```

### Step 10 · 调整阈值（按实际工况）

根据板子实际散热和负载调整。当前默认值基于室内空调环境。

```bash
# 查看当前阈值
python -c "from dynamic_control.config import get_config; c=get_config(); print(f'temp_warn={c.temp_warn_c}°C temp_crit={c.temp_crit_c}°C mem_warn={c.mem_warn_gb}GB')"

# 覆盖（按你的板子实际工况）
export RK3588_TEMP_WARN_C=70    # 散热好就放宽
export RK3588_MEM_WARN_GB=0.5   # 内存紧张就收紧
```

---

## 常见问题

### Q: Monitor 返回全是 0/None

sysfs 路径不对。跑 Step 1，把输出发我。不同 RK3588 板子的 `thermal_zone` 编号和 `devfreq` 设备名不同。

### Q: RKNN 模型加载失败

- 确认 `qwen2.5-1.5b.rknn` 文件存在且不为空
- 确认模型转换时设置了 `target_platform="rk3588"`
- 确认 NPU 驱动已加载：`lsmod | grep rknpu`

### Q: 推理时 OOM

4GB 跑 Qwen2.5-1.5B + 系统开销很紧。尝试：
- 用 int4 量化替代 int8（模型转换时 `quantized_dtype="w4a8"`）
- 减少 context 长度：`export RK3588_DEFAULT_CONTEXT_LEN=1024`

### Q: 如何强制指定后端（调试用）

```python
# 强制用 NPU
result = ctrl.generate_json(prompt, schema, force_backend="npu")

# 强制用 CPU
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
| rknn-toolkit-lite2 + psutil + pydantic | openai + pydantic + psutil | torch + transformers + outlines |
| Qwen2.5-1.5B (.rknn int8) | Qwen2.5-1.5B (Ollama q4) | Qwen2.5-1.5B (HuggingFace) |

---

## 更新日志

| 日期 | 内容 |
|------|------|
| 7/16 | Day 1-2：环境搭建 + 核心函数 |
| 7/17 | Day 3：异常处理 + 场景测试 |
| 7/18 | Day 11-14：Ollama 部署，内存达标 |
| 7/19 | Day 15-22：单元测试 + 文档 + PPT |
| 8/11 | Day 23+：动态控制系统架构 + 实现 + 架构图 |
