# RK3588 上板适配指南（约束解码版）

> **核心任务 1.1 的正确定义**：在**机床(token 采样)层面**用"底层正则解码算子"硬约束模型，
> 强制它只输出 JSON / Diff / 纯结论，熔断"推理过程/自然语言废话"。
>
> **RK3588 上唯一可行且正确的实现** = **llama.cpp 的 JSON-schema grammar 约束解码**（主），
> **Ollama 原生 structured outputs**（备）。
> ⚠️ SGLang/vLLM 是 x86+GPU 框架，RK3588(ARM) 上无法运行。
> ⚠️ 旧 `rknnlite/RKNN` 路线是**图像单次前向 API，不是 LLM 工具链**，无法约束解码，已弃用。

---

## 文件兼容性一览

| 文件 | GPU/WSL | RK3588 | 说明 |
|------|---------|--------|------|
| `sglang_1_1_module_WSL_备份.py` | ✅ 主用 | ❌ | GPU + outlines 约束解码（开发/答辩演示） |
| `sglang_1_1_ollama.py` | ✅ | ⚠️ 可跑 | Ollama 原生 JSON mode |
| `dynamic_control/` | — | ✅ 主用 | 含 **llama.cpp(主) + Ollama(备)** 约束解码后端 + 动态控制 |
| `run_rk3588.sh` | ❌ | ✅ | 一键启动 |
| `test_dynamic_control.py` | ✅ | ✅ | 跨平台，RK3588 上跑更有意义 |
| `demo_dynamic_control.py` | ✅ | ✅ | 6 场景决策演示 |
| 其余测试/文档 | — | — | 通用 |

---

## RK3588 从头部署步骤

### 1. 装系统依赖

```bash
# 板子刷好 Armbian / Ubuntu ARM 后
sudo apt update
sudo apt install -y python3 python3-pip curl git build-essential

# 内存充足检查
free -h   # 4GB 板：注意余量
```

### 2. 装 Python 依赖

```bash
cd ~/1.1模块
pip install psutil pydantic
```

**不需要**装 `torch` / `transformers` / `outlines` / `rknn-toolkit` —— 那是 GPU/PC 转换用的。
RK3588 上 LLM 推理由 **llama-server 或 Ollama** 提供，本代码用标准库 urllib 通过 HTTP 调用。

### 3. 确认 sysfs 路径（动态控制监控用）

```bash
cat /sys/class/thermal/thermal_zone*/type      # cpu-thermal / npu-thermal / gpu-thermal
ls /sys/class/devfreq/                          # 含 "npu" 的设备名
free -h                                         # 内存
```

> `monitor.py` 会自动扫描含 `cpu/npu/gpu` 关键词的 thermal zone，通常无需手动改；
> 若板子 type 特殊，编辑 `dynamic_control/monitor.py` 的 `_scan_thermal_zones()` 补一行。

---

## 4. 部署约束解码推理（核心）——二选一

### 方案 A：llama.cpp（主推，硬约束最强）

#### A.1 获取量化模型（.gguf，在 PC 或板子上下载）

```bash
# 推荐 Qwen2.5-1.5B-Instruct 的 GGUF（内存友好，4GB 板稳）
# 从 HF 或魔搭下载，例如:
#   Qwen/Qwen2.5-1.5B-Instruct-GGUF 里的 qwen2.5-1.5b-instruct-q4_k_m.gguf
# 拷到板子:
scp qwen2.5-1.5b-instruct-q4_k_m.gguf user@rk3588:/data/
```

#### A.2 编译 / 安装 llama-server（ARM）

源码编译较费时，建议优先用预编译包或系统包：

```bash
# 方式1：若板子系统有 llama.cpp 包（部分发行版）
sudo apt install -y llama-cpp   # 或有则用

# 方式2：源码编译（约 20-40 分钟）
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) llama-server
# 产物: build/bin/llama-server
```

#### A.3 启动 llama-server（带 JSON-schema 约束能力）

```bash
cd ~/llama.cpp/build/bin
./llama-server \
  -m /data/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  --n-gpu-layers 0 \         # 若想用 NPU 需另配；默认 CPU
  -c 2048                    # context 长度（按内存调，4GB 建议 ≤2048）
# 后台运行建议用 systemd / nohup
```

> **说明**：llama-server 的 `/v1/chat/completions` 支持 `response_format={"type":"json_schema",...}`，
> 内部把 JSON Schema 编译成 **GBNF grammar**，在**每个 token 采样时**强制只能输出合法 JSON ——
> 这正是"机床层正则解码"。代码 `llamacpp_backend.py` 已封装此请求。

#### A.4 验证服务与约束解码

```bash
# 1) 服务可达
curl http://127.0.0.1:8080/v1/models

# 2) 约束解码实测：要求只能出 JSON
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"输出张三年龄20岁"}],
    "response_format":{"type":"json_schema",
        "json_schema":{"name":"s","schema":{"type":"object",
           "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
           "required":["name","age"]}}},
    "max_tokens":100
  }'
# 期望：只返回 {"name":"...","age":20}，不会夹带"推理过程/废话"
```

### 方案 B：Ollama（备选，省事）

```bash
# 安装 ARM 版
curl -fsSL https://ollama.com/install.sh | sh

# 拉模型
ollama pull qwen2.5:1.5b

# 启动服务
ollama serve &   # 默认 http://127.0.0.1:11434

# 约束解码验证：Ollama 的 /api/chat 支持 format=json_schema
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen2.5:1.5b",
    "messages":[{"role":"user","content":"输出张三年龄20岁"}],
    "format":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]},
    "stream":false
  }'
```

---

## 5. 对接本代码

默认代码连 `llama.cpp`(127.0.0.1:8080)；可用环境变量切换备选：

```bash
# 默认（llama.cpp）
export LLAMACPP_BASE_URL=http://127.0.0.1:8080

# 若用 Ollama 备选：
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:1.5b
```

跑通：

```bash
cd ~/1.1模块
python3 -c "
from dynamic_control import DynamicController
c=DynamicController(); c.init()
print(c.generate_json('输出学生张三,20岁,计算机专业',
  {'type':'object','properties':{'name':{'type':'string'},'age':{'type':'integer'},'major':{'type':'string'}},'required':['name','age','major']}))
"
# 期望: {'name': '张三', 'age': 20, 'major': '计算机'}  （无废话，纯JSON）
```

`bash run_rk3588.sh` 一键启动。

---

## 6. 动态控制验证（不依赖模型也行）

```bash
python3 test_dynamic_control.py   # 8 用例
python3 demo_dynamic_control.py   # 6 场景
```

---

## 两块 API 对比（给调度员联调）

| 环境 | 导入 | 说明 |
|------|------|------|
| GPU/WSL | `sglang_1_1_module_WSL_备份.py` | outlines 约束解码，开发/答辩 |
| RK3588 | `from dynamic_control import DynamicController` | llama.cpp/Ollama 约束解码 + 动态调控 |

API：`ctrl.generate_json(prompt, schema)`、`generate_label`、`generate_value`、`generate_diff`、`status()`、`health()`。

---

## 快速排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `curl :8080/v1/models` 无响应 | llama-server 未启动/端口不对 | 见 4.A 启动；确认 `LLAMACPP_BASE_URL` |
| `generate_*` 连不上 | 服务未起/地址不对 | `export LLAMACPP_BASE_URL=...` 或 `OLLAMA_BASE_URL=...` |
| 服务起了但报错 | llama-server 版本老，不支持 response_format | 用最新 llama.cpp 重编 |
| 返回里有废话/非JSON | 没用约束解码 | 确认走后端 `generate_json`（structured），别调 `generate` |
| 内存 OOM | 模型大 / context 长 | 用 q4_k_m 量化 + `-c 1024/2048` |
| 监控全 0/None | sysfs 路径 | 按 Step3 确认 thermal zone |