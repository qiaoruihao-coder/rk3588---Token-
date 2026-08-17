"""
Ollama 约束解码后端（RK3588 备选推理）
=========================================
调用本机 Ollama 服务（ARM 版），利用其原生 **JSON 模式 / structured outputs**
（`format=json` 或 `format=<schema>`）在后端做结构化约束，只输出合法 JSON。
作为 llama.cpp 的备选（若板子上装了 Ollama 更省事）。

前置（在 RK3588 上）：
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5:1.5b          # 或 qwen2.5:3b
  ollama serve &

配置（环境变量，可选）：
  OLLAMA_BASE_URL  默认 http://127.0.0.1:11434
  OLLAMA_MODEL     默认取拉取的模型（如 qwen2.5:1.5b）
"""
import json
import os
import urllib.request


class OllamaBackend:
    """Ollama 约束解码后端。"""

    rule_only = False  # 真实模型后端

    def __init__(self, base_url: str = None, model: str = None):
        self._base_url = (base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self._model_override = model
        self._model_name = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
        self._loaded = False

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def base_url(self) -> str:
        return self._base_url

    # ── 生命周期 ──────────────────────────────────
    def load(self) -> bool:
        try:
            self._probe()
            self._loaded = True
            return True
        except Exception as e:
            print(f"[OllamaBackend] 连接 Ollama 失败: {e}")
            self._loaded = False
            return False

    def _probe(self) -> None:
        with urllib.request.urlopen(
                f"{self._base_url}/api/tags", timeout=4) as resp:
            json.loads(resp.read().decode("utf-8"))

    def is_available(self) -> bool:
        return self._loaded

    # ── 生成 ──────────────────────────────────────
    def _chat(self, prompt: str, schema: dict | None,
              max_tokens: int, json_mode: bool) -> str:
        payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": int(max_tokens)},
        }
        if json_mode:
            if schema is not None:
                # Ollama 支持 format=<json schema> 做 structured outputs
                payload["format"] = schema
            else:
                payload["format"] = "json"

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]

    def generate(self, prompt: str, schema: dict | None = None,
                 max_tokens: int = 200, context_len: int = 2048,
                 tier: str = "full") -> str:
        if not self._loaded:
            raise RuntimeError("Ollama 未连接，先 ollama serve")
        return self._chat(prompt, schema, max_tokens, json_mode=False)

    def generate_json(self, prompt: str, schema: dict,
                      max_tokens: int = 200, context_len: int = 2048,
                      retries: int = 3, tier: str = "full") -> dict:
        """JSON schema structured 输出。tier 供模型档位选择(默认取当前模型)。"""
        last = None
        for _ in range(max(1, retries)):
            text = self._chat(prompt, schema, max_tokens, json_mode=True)
            text = text.strip()
            try:
                parsed = json.loads(text)
                for req in schema.get("required", []):
                    if req not in parsed:
                        raise KeyError(f"missing required field: {req}")
                return parsed
            except (json.JSONDecodeError, KeyError) as e:
                last = e
        raise RuntimeError(f"generate_json (ollama) 失败: {last}")

    def generate_label(self, prompt: str, labels: list,
                       max_tokens: int = 10, context_len: int = 2048,
                       retries: int = 3, tier: str = "full") -> str:
        schema = {"type": "string", "enum": list(labels)}
        text = self._chat(prompt, schema, min(int(max_tokens), 32), json_mode=False)
        text = text.strip().lower()
        for lab in labels:
            if lab.lower() in text:
                return lab
        raise RuntimeError(f"label 不在集合 {labels}: {text!r}")


def create_backend(base_url: str = None, model: str = None) -> OllamaBackend:
    return OllamaBackend(base_url, model)