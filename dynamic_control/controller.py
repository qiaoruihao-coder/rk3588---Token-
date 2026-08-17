"""
DynamicController — 动态推理控制器
===================================
统一入口，每次请求自动:
  1. 采集系统状态 (Monitor)
  2. 评估决策 (Evaluator)
  3. 执行调整 (Actuator)
  4. 路由到正确的推理后端 (Backend)

对上层调度员完全透明，API 与现有 sglang_1_1_ollama.py 兼容。
"""

import json
import time
import traceback
from typing import Optional

from .monitor import Monitor, SystemState
from .evaluator import Evaluator, Decision
from .actuator import Actuator
from .backends import CPUBackend, BaseBackend
from .llamacpp_backend import LlamaCppBackend
from .ollama_backend import OllamaBackend


class DynamicController:
    """
    运行时动态控制的主入口。

    用法:
        ctrl = DynamicController()
        ctrl.init()  # 加载模型

        # 与现有 API 完全兼容
        result = ctrl.generate_json("输出张三信息", schema)

        # 查询状态
        status = ctrl.status()

        # 强制指定后端 (调试用)
        result = ctrl.generate_label("质量?", ["ok","bad"], force_backend="cpu")
    """

    def __init__(self, rknn_model_path: str = "qwen2.5-1.5b.rknn"):
        self.monitor = Monitor()
        self.evaluator = Evaluator()
        self.actuator = Actuator()

        # 注册后端:
        #  - "npu"  主推理 = llama.cpp 约束解码(保留npU键名兼容调度员, 底层是真约束解码)
        #  - "ollama" 备选 = Ollama JSON 约束解码
        #  - "cpu"  兜底 = 规则引擎
        self._backends: dict[str, BaseBackend] = {
            "npu": LlamaCppBackend(),
            "ollama": OllamaBackend(),
            "cpu": CPUBackend(),
        }
        self._primary = "npu"

        # 统计
        self._stats = {
            "total_requests": 0,
            "by_backend": {"npu": 0, "ollama": 0, "cpu": 0, "refuse": 0},
            "by_level": {"idle": 0, "normal": 0, "warning": 0, "critical": 0},
            "refused": 0,
        }

    # ── 初始化 ──────────────────────────────────────────
    def init(self) -> bool:
        """初始化主后端 (RKNN)。返回 True 表示成功。"""
        print("[DynamicController] Initializing...")
        for name, backend in self._backends.items():
            if hasattr(backend, 'load'):
                ok = backend.load()
                if ok:
                    print(f"  [{name}] loaded")
                else:
                    print(f"  [{name}] unavailable, will use fallback")

        # 确保至少有一个后端可用
        available = [b for b in self._backends.values() if b.is_available()]
        if not available:
            print("  WARNING: no backend available, all requests will be refused")
            self._primary = "refuse"
        else:
            self._primary = "npu" if self._backends["npu"].is_available() else "cpu"
            print(f"  Primary backend: {self._primary}")
        return len(available) > 0

    # ── 核心流程: 请求 → 采集 → 决策 → 执行 → 推理 ────
    def _handle(self, prompt: str, schema: Optional[dict],
                max_tokens_base: int, context_len_base: int,
                force_backend: Optional[str] = None,
                structured: bool = False) -> tuple:
        """
        一次完整的动态请求处理。

        structured=True 时走后端的"约束解码"路径(generate_json, 采样层硬约束),
        False 时走普通 generate(供纯文本/Diff/label 文本用)。

        Returns: (result, meta: dict)
        """
        self._stats["total_requests"] += 1
        meta = {"timestamp": time.time()}

        # Step 1: 采集系统状态
        state = self.monitor.snapshot()
        meta["state"] = state

        # Step 2: 决策
        decision = self.evaluator.evaluate(state)
        meta["decision"] = decision
        self._stats["by_level"][decision.level] = \
            self._stats["by_level"].get(decision.level, 0) + 1

        # Step 3: 执行 (调参)
        apply_result = self.actuator.apply(decision)
        meta["applied"] = apply_result

        # Step 4: 确定后端
        if force_backend:
            backend_name = force_backend
        else:
            backend_name = decision.backend

        # 拒绝请求
        if not decision.can_infer or backend_name == "refuse":
            self._stats["refused"] += 1
            self._stats["by_backend"]["refuse"] = \
                self._stats["by_backend"].get("refuse", 0) + 1
            raise RuntimeError(f"推理被拒绝: {decision.refuse_reason or '系统资源不足'}")

        # 实际推理参数 (Actuator 当前值优先)
        ctx_len = self.actuator.context_len or decision.context_len
        max_tok = self.actuator.max_tokens or decision.max_tokens

        # 获取后端
        backend = self._backends.get(backend_name)
        if backend is None or not backend.is_available():
            # fallback: 尝试任一可用后端
            available = [(n, b) for n, b in self._backends.items() if b.is_available()]
            if not available:
                raise RuntimeError("没有可用的推理后端")
            backend_name, backend = available[0]

        self._stats["by_backend"][backend_name] = \
            self._stats["by_backend"].get(backend_name, 0) + 1
        meta["actual_backend"] = backend_name
        meta["actual_ctx_len"] = ctx_len
        meta["actual_max_tokens"] = max_tok

        # Step 5: 推理 (约束解码 / 普通生成)
        t0 = time.time()
        if structured and hasattr(backend, "generate_json"):
            raw = backend.generate_json(prompt, schema, max_tokens=max_tok,
                                        context_len=ctx_len)
        else:
            raw = backend.generate(prompt, schema, max_tokens=max_tok, context_len=ctx_len)
        meta["inference_time"] = time.time() - t0

        return raw, meta

    # ── 公开 API (与 sglang_1_1_ollama.py 兼容) ─────────

    def generate_json(self, prompt: str, schema: dict,
                      retries: int = 3, force_backend: Optional[str] = None) -> dict:
        """
        JSON Schema 约束输出。动态调控透明。

        调用方无需关心是用 NPU 还是 CPU —— Controller 自动决定。
        """
        meta = {}
        for attempt in range(1, retries + 1):
            try:
                result, meta = self._handle(prompt, schema,
                                            max_tokens_base=200,
                                            context_len_base=2048,
                                            force_backend=force_backend,
                                            structured=True)
                # 约束解码后端已返回 dict(如 llama.cpp/Ollama) 
                if isinstance(result, dict):
                    return result
                # 兜底(如 CPU 规则返回了 str)：解析
                import re
                text = result.strip() if isinstance(result, str) else str(result)
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
                parsed = json.loads(text)
                if "required" in schema:
                    for f in schema["required"]:
                        if f not in parsed:
                            raise KeyError(f"Missing required field: {f}")
                return parsed

            except (json.JSONDecodeError, KeyError) as e:
                if attempt >= retries:
                    raise RuntimeError(
                        f"generate_json failed after {retries} retries: {e}"
                    )
            except RuntimeError:
                raise  # 拒绝请求，不重试

        raise RuntimeError("Unreachable")

    def generate_json_pydantic(self, prompt: str, model_class,
                               retries: int = 3, force_backend: Optional[str] = None) -> dict:
        """
        Pydantic 约束输出。
        """
        schema = model_class.model_json_schema()
        result = self.generate_json(prompt, schema, retries, force_backend)
        model_class(**result)  # Pydantic 验证
        return result

    def generate_label(self, prompt: str, labels: list,
                       retries: int = 3, force_backend: Optional[str] = None) -> str:
        """分类标签约束输出"""
        backend = self._get_backend(force_backend)
        return backend.generate_label(prompt, labels,
                                       max_tokens=self.actuator.max_tokens,
                                       context_len=self.actuator.context_len,
                                       retries=retries)

    def generate_value(self, prompt: str, schema: dict = None,
                       retries: int = 3, force_backend: Optional[str] = None) -> dict:
        """数值+结论输出"""
        if schema is None:
            schema = {
                "type": "object",
                "properties": {
                    "area_mm2": {"type": "number"},
                    "quality": {"type": "string", "enum": ["qualified", "defective"]}
                },
                "required": ["area_mm2", "quality"]
            }
        return self.generate_json(prompt, schema, retries, force_backend)

    def generate_diff(self, prompt: str, old_code: str = "",
                      force_backend: Optional[str] = None) -> str:
        """代码 Diff 输出"""
        backend = self._get_backend(force_backend)
        if old_code:
            full = (f"Original:\n```\n{old_code}\n```\n"
                    f"Change: {prompt}\nOutput ONLY unified diff.")
        else:
            full = f"Output unified diff for: {prompt}"
        return backend.generate(full, None,
                                max_tokens=self.actuator.max_tokens,
                                context_len=self.actuator.context_len)

    def _get_backend(self, force: Optional[str] = None) -> BaseBackend:
        """获取当前应使用的后端"""
        name = force or self.actuator.get_backend() or self._primary
        be = self._backends.get(name)
        if be and be.is_available():
            return be
        available = [(n, b) for n, b in self._backends.items() if b.is_available()]
        if available:
            return available[0][1]
        raise RuntimeError("No backend available")

    # ── 状态查询 ─────────────────────────────────────────
    def status(self) -> dict:
        """
        获取当前完整状态，供调度员查询。

        Returns:
            {
                "system": {...},       # 温度/频率/内存
                "decision": "...",     # 当前决策等级
                "backend": "npu",      # 当前后端
                "params": {...},       # 当前推理参数
                "stats": {...},        # 累计统计
            }
        """
        state = self.monitor.snapshot()
        decision = self.evaluator.evaluate(state)
        return {
            "system": self.monitor.summary(),
            "decision_level": decision.level,
            "backend": self.actuator.get_backend(),
            "params": {
                "context_len": self.actuator.context_len,
                "max_tokens": self.actuator.max_tokens,
            },
            "stats": dict(self._stats),
        }

    def health(self) -> dict:
        """轻量健康检查，不触发决策"""
        s = self.monitor.snapshot()
        return {
            "level": s.level,
            "backend": self.actuator.get_backend(),
            "npu_avail": self._backends["npu"].is_available(),
            "cpu_avail": self._backends["cpu"].is_available(),
            "mem_avail_gb": round(s.mem_avail_gb, 2),
            "cpu_temp_c": s.cpu_temp_c,
        }


# ── 便捷工厂 ──────────────────────────────────────────────

def create_controller(rknn_model: str = "qwen2.5-1.5b.rknn") -> DynamicController:
    """创建并初始化控制器"""
    ctrl = DynamicController(rknn_model)
    ctrl.init()
    return ctrl
