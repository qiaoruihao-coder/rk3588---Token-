"""
推理后端抽象接口 + 具体实现
============================
- base.py: 抽象基类
- rknn_backend.py: RKNN NPU 推理
- cpu_backend.py: 纯 CPU 兜底推理 (轻量，不用 Ollama)
"""

from abc import ABC, abstractmethod
from typing import Optional
import json, re, time


# ── 抽象基类 ─────────────────────────────────────────────

class BaseBackend(ABC):
    """所有推理后端的统一接口"""

    # 是否为"占位规则引擎"(未接入真实模型)。真实后端应为 False。
    # controller 对 rule_only 后端默认拒绝返回假数据, 需显式 RK3588_ALLOW_RULE_BACKEND=1 才放行。
    rule_only: bool = False

    @abstractmethod
    def generate(self, prompt: str, schema: Optional[dict] = None,
                 max_tokens: int = 200, context_len: int = 2048) -> str:
        """原始生成，返回文本"""
        ...

    def generate_json(self, prompt: str, schema: dict,
                      max_tokens: int = 200, context_len: int = 2048,
                      retries: int = 3, tier: str = "full") -> dict:
        """JSON 约束生成 (带重试，各后端可覆盖)"""
        schema_str = json.dumps(schema)
        system = (
            "You are a JSON-only assistant. Output valid JSON matching this schema:\n"
            f"{schema_str}\n"
            "No markdown, no explanation."
        )
        for attempt in range(1, retries + 1):
            try:
                full_prompt = f"{system}\n\nUser: {prompt}\nAssistant:"
                text = self.generate(full_prompt, schema, max_tokens, context_len)
                text = text.strip()
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
                parsed = json.loads(text)
                if "required" in schema:
                    for f in schema["required"]:
                        if f not in parsed:
                            raise KeyError(f"Missing: {f}")
                return parsed
            except (json.JSONDecodeError, KeyError) as e:
                if attempt >= retries:
                    raise RuntimeError(f"JSON generation failed: {e}")
        return {}

    def generate_label(self, prompt: str, labels: list,
                       max_tokens: int = 10, context_len: int = 2048,
                       retries: int = 3) -> str:
        """标签约束生成"""
        system = (
            "Reply with EXACTLY ONE WORD from: " + str(labels) + ". "
            "No explanation."
        )
        for attempt in range(1, retries + 1):
            full_prompt = f"{system}\n\nUser: {prompt}\nLabel:"
            text = self.generate(full_prompt, None, max_tokens, context_len)
            text = text.strip().lower()
            for label in labels:
                if label.lower() in text:
                    return label
            if attempt >= retries:
                raise RuntimeError(f"Label not in {labels}: {text}")
        return labels[0]  # unreachable

    def is_available(self) -> bool:
        """检查此后端是否可用"""
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ── RKNN 后端 ────────────────────────────────────────────

class RKNBackend(BaseBackend):
    """
    RKNN NPU 推理后端。

    需要:
      - rknn-toolkit2 已安装
      - 模型已转换为 .rknn 格式
      - RKNN 实例已初始化

    模型转换命令 (Qwen2.5 → RKNN):
      python -m rknn.api.export \
        --model qwen2.5-1.5b.onnx \
        --target rk3588 \
        --quantization i8 \
        --output qwen2.5-1.5b.rknn
    """

    def __init__(self, model_path: str = "qwen2.5-1.5b.rknn"):
        self._model_path = model_path
        self._rknn = None
        self._tokenizer = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "rknn"

    def load(self) -> bool:
        """加载 RKNN 模型。返回 True 表示成功。"""
        try:
            from rknnlite.api import RKNNLite
            self._rknn = RKNNLite()
            ret = self._rknn.load_rknn(self._model_path)
            if ret != 0:
                raise RuntimeError(f"load_rknn failed with {ret}")

            ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                raise RuntimeError(f"init_runtime failed with {ret}")

            self._loaded = True
            return True
        except ImportError:
            print("[RKNBackend] rknnlite not installed, fallback to CPU")
            return False
        except Exception as e:
            print(f"[RKNBackend] load failed: {e}")
            return False

    def is_available(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, schema: Optional[dict] = None,
                 max_tokens: int = 200, context_len: int = 2048) -> str:
        """RKNN 推理 (单次前向)"""
        if not self._loaded:
            raise RuntimeError("RKNN model not loaded")

        try:
            from rknnlite.api import RKNNLite
            # 截断 prompt 到 context_len 范围内
            # (简化实现: token 数 ≈ len/4 估算中文, len/2 英文)
            max_input_chars = min(len(prompt), context_len * 2)
            truncated = prompt[:max_input_chars]

            # RKNNLite inference — 具体输入格式取决于 tokenizer
            # 这里是简化框架，实际需配 tokenizer
            inputs = [truncated.encode('utf-8')[:context_len * 4]]
            outputs = self._rknn.inference(inputs)

            # 解码输出
            if outputs and len(outputs) > 0:
                try:
                    return bytes(outputs[0]).decode('utf-8', errors='replace')
                except Exception:
                    return str(outputs[0])
            return ""
        except Exception as e:
            raise RuntimeError(f"RKNN inference failed: {e}")

    def __del__(self):
        if self._rknn:
            try:
                self._rknn.release()
            except Exception:
                pass


# ── CPU 兜底后端 ─────────────────────────────────────────

class CPUBackend(BaseBackend):
    """
    纯 CPU 推理兜底，用于 NPU 过载/过热时的降级推理。

    策略: 用最简单的模板匹配 + 规则引擎，不做真正的 LLM 推理。
    适用于 4GB RK3588 —— 没办法在内存里同时存 NPU 模型和 CPU LLM。

    实际项目中，如果你有独立的轻量 CPU 模型（如 100MB 的 tinyllama.cpp），
    替换这里的 _generate_rule 实现。
    """

    # 占位规则引擎: 输出的 unknown/0 是"编造占位值", 非真实推理结果。
    # controller 默认会拒绝放行, 需 RK3588_ALLOW_RULE_BACKEND=1 才允许(调试/无模型时)。
    rule_only: bool = True

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "cpu"

    def generate(self, prompt: str, schema: Optional[dict] = None,
                 max_tokens: int = 200, context_len: int = 2048,
                 tier: str = "full") -> str:
        """
        CPU 规则引擎推理。

        TODO: 替换为实际的 CPU 推理 (如 llama.cpp 轻量模型)。
        当前用规则兜底，保证系统在 critical 状态下不会完全不可用。
        """
        # 规则匹配：从 prompt 中提取关键信息
        return self._generate_rule(prompt, schema)

    def _generate_rule(self, prompt: str, schema: Optional[dict]) -> str:
        """基于规则的简单解析 (CPU fallback 专属)"""
        # 尝试从 prompt 中提取 JSON 结构
        if schema and isinstance(schema, dict):
            # 枚举约束(label 场景): 从 enum 中按关键词匹配返回
            if "enum" in schema:
                text = (prompt or "").lower()
                for cand in schema["enum"]:
                    if str(cand).lower() in text:
                        return str(cand)
                return str(schema["enum"][0]) if schema["enum"] else "unknown"
            props = schema.get("properties", {})
            result = {}
            text = (prompt or "").lower()
            for key, prop in props.items():
                # 枚举约束: 从 enum 中按关键词匹配(占位引擎增强)
                if "enum" in prop:
                    matched = None
                    for cand in prop["enum"]:
                        if str(cand).lower() in text:
                            matched = str(cand)
                            break
                    result[key] = matched if matched else str(prop["enum"][0])
                elif prop.get("type") == "string":
                    result[key] = "unknown"
                elif prop.get("type") == "integer":
                    result[key] = 0
                elif prop.get("type") == "number":
                    result[key] = 0.0
            return json.dumps(result)

        # 纯文本 prompt: 返回空或 echo 关键部分
        return prompt[:200]

    def is_available(self) -> bool:
        return True
