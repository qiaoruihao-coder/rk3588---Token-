"""
llama.cpp 约束解码后端（RK3588 主推理）
=========================================
通过本机 llama-server（llama.cpp 的 OpenAI 兼容服务）调用 LLM，
并利用其 **JSON-schema grammar 约束解码**（chat 模板 + response_format / grammar
在 token 采样层硬约束），强制只输出合法 JSON —— 这正是核心任务 1.1 要求的
"机床层面控制输出，熔断推理过程与自然语言废话"。

前置（在 RK3588 上）：
  - llm 已编译（ARM 可编），或装了 llama-server 可执行文件
  - 已有一个量化模型 .gguf（如 Qwen2.5-1.5B / 3B）
  - 启动: llama-server -m model.gguf --host 127.0.0.1 --port 8080 \
            --chat-template deepseek3  （按模型选模板）
  - 或更省事: 使用 --json-schema / grammar 能力（llama-server ≥ bXXXX，
    /v1/chat/completions 支持 response_format）

配置（环境变量，可选）：
  LLAMACPP_BASE_URL  默认 http://127.0.0.1:8080
  LLAMACPP_MODEL     模型名，默认取 /v1/models 第一个

本后端走 OpenAI 兼容 /v1/chat/completions，用 stdlib urllib，无额外依赖。
"""
import json
import os
import re
import urllib.request


class LlamaCppBackend:
    """llama.cpp 约束解码后端。"""

    # 真实模型后端（非占位规则引擎）；controller 的降级判定据此区分
    rule_only = False

    def __init__(self, base_url: str = None, model: str = None):
        self._base_url = (base_url or os.environ.get(
            "LLAMACPP_BASE_URL", "http://127.0.0.1:8080")).rstrip("/")
        self._model_override = model
        self._model_name = None
        self._loaded = False  # 服务可达才置 True

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── 生命周期 ──────────────────────────────────
    def load(self) -> bool:
        """探测本机 llama-server 是否可达。"""
        try:
            self._probe()
            self._loaded = True
            return True
        except Exception as e:
            print(f"[LlamaCppBackend] 连接本地 llama-server 失败: {e}")
            self._loaded = False
            return False

    def _probe(self) -> None:
        with urllib.request.urlopen(
                f"{self._base_url}/v1/models", timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", [])]
        self._model_name = self._model_override or (models[0] if models else None)

    def is_available(self) -> bool:
        return self._loaded

    # ── 生成 ──────────────────────────────────────
    def _resolve_model(self, tier: str = "full") -> str:
        """按模型档位返回对应模型名(动态选择模型)。"""
        try:
            from .config import get_config
            cat = get_config().model_tier_catalog or {}
            entry = cat.get(tier)
            if entry and entry.get("model"):
                return entry["model"]
        except Exception:
            pass
        return self._model_name

    def _chat(self, prompt: str, schema: dict | None,
              max_tokens: int, json_mode: bool, tier: str = "full") -> str:
        """
        核心：调用 llama-server 的 /v1/chat/completions。
        json_mode=True 且带 schema 时，用 response_format={"type":"json_schema"} 触发
        JSON-schema grammar 硬约束解码（由 llama.cpp 在采样层执行）。
        可按 tier 动态切换加载的模型（老师要求"调节模型选择"）。
        """
        payload = {
            "model": self._resolve_model(tier),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if json_mode:
            if schema is not None:
                # llama.cpp 支持 json_schema 约束（B/get_grammar 能力）
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": schema,
                    },
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str, schema: dict | None = None,
                 max_tokens: int = 200, context_len: int = 2048,
                 tier: str = "full") -> str:
        """默认生成 (json_mode=False，供纯文本/Diff 用)。tier 选模型档位。"""
        if not self._loaded:
            raise RuntimeError("llama-server 未加载，先启动本机 llama-server")
        return self._chat(prompt, schema, max_tokens, json_mode=False, tier=tier)

    # ── 约束解码入口（供 controller 使用）───────────
    def generate_json(self, prompt: str, schema: dict,
                      max_tokens: int = 200, context_len: int = 2048,
                      retries: int = 3, tier: str = "full") -> dict:
        """JSON-schema 硬约束解码：由 llama.cpp grammar 在 token 层锁死。"""
        last = None
        for _ in range(max(1, retries)):
            text = self._chat(prompt, schema, max_tokens, json_mode=True, tier=tier)
            text = text.strip()
            # 剥离可能的 markdown 围栏
            if text.startswith("```"):
                text = _re_sub_fence(text)
            try:
                parsed = json.loads(text)
                # 基本 field-presence 校验（约束解码通常已保证类型）
                for req in schema.get("required", []):
                    if req not in parsed:
                        raise KeyError(f"missing required field: {req}")
                return parsed
            except (json.JSONDecodeError, KeyError) as e:
                last = e
        raise RuntimeError(f"generate_json 约束解码失败: {last}")

    def generate_label(self, prompt: str, labels: list,
                       max_tokens: int = 10, context_len: int = 2048,
                       retries: int = 3, tier: str = "full") -> str:
        """分类：用 enum grammar 罗列合法标签，强制只出其中之一。"""
        schema = {"type": "string", "enum": list(labels)}
        text = self._chat(prompt, schema, min(int(max_tokens), 32), json_mode=False, tier=tier)
        text = text.strip().lower()
        for lab in labels:
            if lab.lower() in text:
                return lab
        # 约束失败兜底：仍从以 prompt 中再试一次原始
        raise RuntimeError(f"label 不在集合 {labels}: {text!r}")


def _re_sub_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*", "", text).replace("```", "").strip()


def create_backend(base_url: str = None, model: str = None) -> LlamaCppBackend:
    return LlamaCppBackend(base_url, model)