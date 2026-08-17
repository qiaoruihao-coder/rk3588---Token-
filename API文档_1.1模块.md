# 1.1 模块 API 文档

> **执行人层 - 输出Token强制控制**
> 作者：乔瑞浩、张倬铭 | 对接：调度员2.1（韩乐瞳）

---

## 快速开始

```python
# WSL版 (高精度, 需GPU)
from sglang_1_1_module import generate_json, generate_label, generate_value, generate_diff

# Ollama版 (低内存, Windows原生)
from sglang_1_1_ollama import generate_json, generate_label, generate_value, generate_diff
```

---

## API 列表

### 1. generate_json — JSON Schema 约束输出

强制模型输出符合指定 JSON Schema 的对象，不含任何废话。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str` | 用户输入 |
| `schema` | `dict` | JSON Schema 定义 |
| `retries` | `int` | 失败重试次数, 默认3 |

```python
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "major": {"type": "string"}
    },
    "required": ["name", "age", "major"]
}

result = generate_json("输出学生张三的信息", schema)
# -> {'name': '张三', 'age': 20, 'major': 'CS'}
```

---

### 2. generate_label — 分类标签约束

强制模型输出预设标签中的一个词。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str` | 用户输入 |
| `labels` | `list[str]` | 可选标签 |
| `retries` | `int` | 失败重试次数, 默认3 |

```python
result = generate_label(
    "摄像头发现产品表面裂纹, 判断质量",
    ["qualified", "defective"]
)
# -> 'defective'
```

---

### 3. generate_value — 数值+结论输出

包装了 `generate_json`，内置工业检测等场景的默认 Schema。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str` | 用户输入 |
| `schema` | `dict` | 可选, 自定义 Schema |
| `retries` | `int` | 失败重试次数, 默认3 |

```python
result = generate_value("轴承表面检测到12.5平方毫米裂纹")
# -> {'area_mm2': 12.5, 'quality': 'defective'}
```

---

### 4. generate_diff — 代码Diff补丁

输出标准 unified diff 格式的代码改动。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str` | 改动描述 |
| `old_code` | `str` | 原始代码（可选） |

```python
result = generate_diff(
    "把变量x重命名为count",
    old_code="x = 1\nprint(x)"
)
# -> '--- a/code.py\n+++ b/code.py\n@@ -1,2 +1,2 @@\n-x = 1\n-print(x)\n+count = 1\n+print(count)'
```

---

## 场景 templates（调度员直接用）

### 工业检测

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

### 交通监控

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

## 两版对比

| | WSL版 | Ollama版 |
|------|------|-----|
| 文件 | `sglang_1_1_module.py` | `sglang_1_1_ollama.py` |
| 准确率 | 100% | ~90% |
| 内存 | 6.2 GB | 0.14 GB |
| 速度 | 1.65s | 0.37s |
| 运行环境 | WSL + GPU | Windows 原生 |
| 适用场景 | 对准确率要求高的任务 | 边缘设备/内存受限 |

---

## 错误处理

所有函数内置自动重试（默认3次）。如果3次都失败，抛出 `RuntimeError`。

```python
try:
    result = generate_json(prompt, schema, retries=3)
except RuntimeError as e:
    print(f"生成失败: {e}")
    result = None
```

---

## 性能基线

| 指标 | 实测值 | 比赛要求 |
|------|--------|---------|
| 输出无废话率 | 100% | 100% |
| 单次推理内存 (Ollama) | 0.14 GB | ≤1.5 GB ✅ |
| JSON生成速度 (Ollama) | 0.37s | <0.2s (端到端) |
| 冲突率 | 0% | ≤5% |
| 批量稳定率 | 100% (WSL) | - |

---

## 联调 checklist

- [ ] 调度员(2.1)能调用 `generate_json` 并获得可解析的 dict
- [ ] 调度员能传入自定义 Schema 并得到合规输出
- [ ] 异常情况（网络断、模型崩）有错误返回而非挂死
- [ ] 两个版本(WSL/Ollama)都已验证可用
- [ ] 工业检测、交通监控场景各通过5次测试
