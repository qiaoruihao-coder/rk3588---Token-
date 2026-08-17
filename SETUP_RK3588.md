# RK3588 上板适配指南

> 当前代码在 WSL (x86 + NVIDIA GPU) 上开发。部署到 RK3588 (ARM + NPU) 需要注意以下事项。

---

## 文件兼容性一览

| 文件 | WSL/GPU | RK3588 | 说明 |
|------|---------|--------|------|
| `sglang_1_1_module.py` | ✅ 主用 | ❌ 不可用 | CUDA + outlines + x86，ARM 上跑不了。仅作开发参考 |
| `sglang_1_1_ollama.py` | ✅ 可用 | ⚠️ 可跑 | 需在 RK3588 上装 Ollama ARM 版 `curl -fsSL https://ollama.com/install.sh \| sh` |
| `dynamic_control/` | ❌ — | ✅ 主用 | 为 RK3588 设计（sysfs + RKNN API） |
| `test_dynamic_control.py` | ✅ | ✅ | 跨平台，RK3588 上跑更有意义 |
| `demo_dynamic_control.py` | ✅ | ✅ | 模拟 + 真实采集混合 |
| `test_11_module.py` | ✅ | ⚠️ | WSL 版测试会失败，Ollama 版可跑 |
| `final_check.py` | ✅ | ⚠️ | 硬编码了 Windows 路径，需改 `BASE` 变量 |
| `run_11_module.sh` | ✅ | ❌ | WSL 专用启动脚本 |
| `run_rk3588.sh` | ❌ | ✅ | RK3588 一键启动脚本 |
| `architecture_diagrams.html` | ✅ | ✅ | 浏览器打开，截图用 |
| 文档 (.docx/.pptx/.md) | ✅ | ✅ | 通用 |

---

## RK3588 从头部署步骤

### 1. 装系统依赖

```bash
# 板子刷好 Armbian / Ubuntu ARM 后
sudo apt update
sudo apt install python3 python3-pip

# 确认 NPU 驱动已加载
lsmod | grep rknpu
# 没输出的话: sudo modprobe rknpu
```

### 2. 装 Python 依赖

```bash
cd 揭榜挂帅

# 核心依赖（RK3588 只需要这个）
pip install psutil pydantic

# RKNN 推理库（板子自带或按厂商文档装）
pip install rknn-toolkit-lite2
# 验证:
python3 -c "from rknnlite.api import RKNNLite; print('RKNN OK')"
```

**不需要装** `torch`、`transformers`、`outlines`、`openai` — 这些是 WSL 版的依赖，RK3588 上只用 `dynamic_control/`。

### 3. 确认 sysfs 路径

不同厂家的 RK3588 板子，thermal zone 编号不同：

```bash
# 查温度传感器
cat /sys/class/thermal/thermal_zone*/type
# 典型输出:
#   cpu-thermal
#   gpu-thermal
#   (有些板子有独立的 npu-thermal)

# 查 NPU 频率调节节点
ls /sys/class/devfreq/
# 找含 "npu" 的那个，比如 fdab0000.npu
```

把输出记下来。如果你的板子 thermal zone 编号和代码里默认的不一致，有两种改法：

**方法 A（推荐）**: 不改代码，用环境变量指定：

不涉及，thermal zone 的映射在 `monitor.py` 的 `_scan_thermal_zones()` 里会自动扫描 `/sys/class/thermal/thermal_zone*/type`，匹配含 `cpu`、`npu`、`gpu` 关键词的 zone。只要 type 文件里有这些关键词就会自动找到。

**方法 B**: 如果自动扫描没匹配到，编辑 `dynamic_control/monitor.py` 的 `_scan_thermal_zones()` 方法，加一行你的 type 名称。

### 4. 转模型（最关键的一步）

Qwen2.5-1.5B → RKNN 格式。需要 ONNX 模型作为中间格式。

```bash
# Step A: 准备 ONNX 模型 (在 x86 机器上做，也可以在板子上用 CPU 慢慢跑)
pip install transformers torch
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
# 导出 ONNX (这一步比较复杂，建议直接 HuggingFace 搜现成的 ONNX)
# 或者用 optimum-cli:
# optimum-cli export onnx --model Qwen/Qwen2.5-1.5B-Instruct qwen_onnx/
"

# Step B: ONNX → RKNN (在 x86 上用 rknn-toolkit2)
pip install rknn-toolkit2
python3 -c "
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588', quantized_dtype='w8a8')
rknn.load_onnx('qwen2.5-1.5b.onnx')
rknn.build(do_quantization=True)
rknn.export_rknn('qwen2.5-1.5b.rknn')
"

# Step C: 把 .rknn 文件 scp 到板子上
scp qwen2.5-1.5b.rknn user@rk3588:~/揭榜挂帅/
```

### 5. 验证

```bash
cd ~/揭榜挂帅

# 验证监控采集
python3 -c "from dynamic_control.monitor import Monitor; print(Monitor().summary())"

# 验证完整流程（需要 .rknn 模型）
python3 run_rk3588.sh qwen2.5-1.5b.rknn
```

### 6. 如果暂时没有 RKNN 模型

动态控制系统可以在纯 CPU 模式下跑，先验证监控和决策逻辑：

```bash
# 即使没有 .rknn 模型也能采集系统状态
python3 test_dynamic_control.py

# 看决策逻辑是否正常
python3 demo_dynamic_control.py
```

Controller 初始化时会自动 fallback 到 CPU 后端（规则引擎），不会崩溃。等模型转好了再切回 NPU。

---

## 两版 API 对比（给调度员联调用）

| 环境 | 导入方式 | 说明 |
|------|---------|------|
| WSL 开发机 | `from sglang_1_1_module import generate_json` | GPU 高精度，答辩演示用 |
| RK3588 部署 | `from dynamic_control import DynamicController` | NPU + 动态调控，正式运行用 |

API 调用方式完全一样：

```python
# WSL 版
from sglang_1_1_module import generate_json
result = generate_json("输出张三信息", schema)

# RK3588 版（动态控制）
from dynamic_control import DynamicController
ctrl = DynamicController("qwen2.5-1.5b.rknn")
ctrl.init()
result = ctrl.generate_json("输出张三信息", schema)
```

---

## 快速排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `from rknnlite.api import RKNNLite` 报错 | 未装 RKNN 库 | `pip install rknn-toolkit-lite2` |
| Monitor 采集全是 0/None | sysfs 路径不匹配 | 按 Step 3 确认 thermal zone 和 devfreq |
| `ctrl.init()` 失败 | 无 .rknn 模型文件 | 先不考虑 NPU，CPU 兜底也能跑 |
| Ollama 版在板子上跑不起来 | Ollama 没装 ARM 版 | `curl -fsSL https://ollama.com/install.sh \| sh` |
| 内存 OOM | 4GB 跑模型 + 系统太紧 | int4 量化或换更小模型 |
