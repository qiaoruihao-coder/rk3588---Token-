"""
Evaluator — 状态机 + 决策引擎
=============================
根据 Monitor 采集的 SystemState，输出调控决策。

决策输出:
  Decision(action, params) — Actuator 按此执行
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from .monitor import SystemState
from .config import get_config


@dataclass
class Decision:
    """一次决策输出"""

    # 推荐后端: "npu" | "cpu" | "refuse" (拒绝请求)
    backend: str = "npu"

    # 推荐的推理参数
    max_tokens: int = 200
    context_len: int = 2048

    # NPU 频率调整: "high" | "mid" | "low" | None (不调整)
    npu_freq_level: Optional[str] = None

    # 模型档位: "full" | "mid" | "tiny" (随状态动态选模型, 老师要求的"调节模型选择")
    model_tier: str = "full"

    # 是否允许继续推理
    can_infer: bool = True

    # 拒绝原因 (can_infer=False 时填写)
    refuse_reason: str = ""

    # 元信息
    level: str = "normal"
    timestamp: float = field(default_factory=time.time)


class Evaluator:
    """
    状态机评估器。

    用法:
        ev = Evaluator()
        state = monitor.snapshot()
        decision = ev.evaluate(state)
    """

    def __init__(self):
        cfg = get_config()
        self.temp_warn = cfg.temp_warn_c
        self.temp_crit = cfg.temp_crit_c
        self.temp_hyst = cfg.temp_hysteresis_c
        self.mem_warn = cfg.mem_warn_gb
        self.mem_crit = cfg.mem_crit_gb
        self.npu_util_warn = cfg.npu_util_warn_pct
        self.npu_util_idle = cfg.npu_util_idle_pct
        self.default_tokens = cfg.default_max_tokens
        self.default_ctx = cfg.default_context_len
        self.warn_ctx_scale = cfg.warn_context_scale
        self.warn_token_scale = cfg.warn_max_tokens_scale
        self.crit_ctx = cfg.crit_context_len
        self.stability_sec = cfg.stability_sec

        # 滞回状态追踪
        self._last_level: str = "normal"
        self._level_since: float = time.time()
        self._last_npu_freq: Optional[str] = None  # 追踪当前频档，避免重复写入

    # ── 主入口 ────────────────────────────────────────
    def evaluate(self, state: SystemState) -> Decision:
        """
        输入: 系统快照
        输出: 调控决策
        """
        d = Decision(level=state.level, context_len=self.default_ctx,
                     max_tokens=self.default_tokens)
        cfg = get_config()

        now = time.time()
        raw_level = state.level

        # ── 滞回稳定化 ──────────────────────────────────
        stable_level = self._apply_hysteresis(raw_level, now)
        d.level = stable_level

        # ── 模型档位动态选择(老师要求)───────────────────
        # 由 config.model_tiers 决定: level -> tier (full/mid/tiny)
        d.model_tier = cfg.model_tiers.get(stable_level, "full")

        # ── 按状态决策 ──────────────────────────────────
        if stable_level == "critical":
            d = self._decide_critical(state, d, cfg)
        elif stable_level == "warning":
            d = self._decide_warning(state, d, cfg)
        elif stable_level == "idle":
            d = self._decide_idle(state, d, cfg)
        else:
            d = self._decide_normal(state, d, cfg)

        d.timestamp = now
        return d

    # ── 滞回控制 ───────────────────────────────────────
    def _apply_hysteresis(self, raw_level: str, now: float) -> str:
        """
        防抖动：状态升级即时生效，降级需稳定 N 秒。
        温度方面用滞回值额外偏置。
        """
        # 升级：即时
        if self._level_order(raw_level) > self._level_order(self._last_level):
            self._last_level = raw_level
            self._level_since = now
            return raw_level

        # 同级别：刷新时间
        if raw_level == self._last_level:
            self._level_since = now
            return raw_level

        # 降级：需要持续稳定
        if now - self._level_since >= self.stability_sec:
            self._last_level = raw_level
            self._level_since = now
            return raw_level

        # 还没稳定，维持上一级
        return self._last_level

    @staticmethod
    def _level_order(level: str) -> int:
        return {"idle": 0, "normal": 1, "warning": 2, "critical": 3}.get(level, 1)

    # ── 各级决策 ──────────────────────────────────────
    def _decide_normal(self, state, d, cfg) -> Decision:
        d.backend = "npu"
        d.context_len = cfg.default_context_len
        d.max_tokens = cfg.default_max_tokens
        d.npu_freq_level = "high"
        return d

    def _decide_warning(self, state, d, cfg) -> Decision:
        """warning 状态下适度降级，仍优先用 NPU"""
        d.backend = "npu"
        d.context_len = int(cfg.default_context_len * cfg.warn_context_scale)
        d.max_tokens = int(cfg.default_max_tokens * cfg.warn_max_tokens_scale)

        # 根据 warn 的具体原因做差别处理
        if state.npu_util_pct > cfg.npu_util_warn_pct:
            # NPU 过载：降频一档，让后续请求排队自然冷却
            d.npu_freq_level = "mid"
        elif (state.cpu_temp_c or 0) > cfg.temp_warn_c or (state.npu_temp_c or 0) > cfg.temp_warn_c:
            # 温度告警：降频降温
            d.npu_freq_level = "mid"
            d.context_len = int(cfg.default_context_len * cfg.warn_context_scale * 0.6)
        else:
            d.npu_freq_level = "high"

        return d

    def _decide_critical(self, state, d, cfg) -> Decision:
        """critical: 切 CPU 推理，最小 context，温度再高就拒绝"""
        # 先判断是否要拒绝所有请求
        mem = state.mem_avail_gb
        cpu_temp = state.cpu_temp_c or 0

        if mem < 0.15:  # 可用内存 < 150MB，什么都跑不了
            d.can_infer = False
            d.backend = "refuse"
            d.refuse_reason = f"可用内存仅 {mem:.0f}MB，无法推理"
            return d

        if cpu_temp > 85.0:  # CPU 也过热，彻底暂停
            d.can_infer = False
            d.backend = "refuse"
            d.refuse_reason = f"CPU 温度 {cpu_temp:.1f}°C，暂停所有推理"
            return d

        # 可推理，但降到最低规格
        d.backend = "cpu"
        d.context_len = cfg.crit_context_len
        d.max_tokens = 50
        d.npu_freq_level = "low"
        return d

    def _decide_idle(self, state, d, cfg) -> Decision:
        """空闲状态: 恢复全规格，NPU 高频"""
        d.backend = "npu"
        d.context_len = cfg.default_context_len
        d.max_tokens = cfg.default_max_tokens
        d.npu_freq_level = "high"
        return d
