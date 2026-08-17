"""
RK3588 动态控制阈值配置
======================
所有阈值集中管理，不从代码里硬编码。
可以通过环境变量覆盖，也可以直接改这里的默认值。
"""

import os
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """动态控制器的所有阈值。

    优先级: 环境变量 > 这里的默认值
    """

    # ── 温度 (摄氏度) ───────────────────────────────────
    # 室内空调环境，RK3588 工作温度建议 0~80°C
    temp_warn_c: float = 65.0      # 超过则进入 warning
    temp_crit_c: float = 75.0      # 超过则进入 critical
    temp_hysteresis_c: float = 5.0 # 滞回：降到 crit-5 才退出 crit

    # ── 内存 (GB) ───────────────────────────────────────
    # 4GB 总量，系统占用约 0.5~1GB，模型约 1~2GB
    mem_warn_gb: float = 0.6     # 可用内存低于此值进入 warning
    mem_crit_gb: float = 0.3     # 可用内存低于此值进入 critical

    # ── NPU 利用率 (%) ──────────────────────────────────
    npu_util_warn_pct: float = 90.0    # 超过则进入 warning
    npu_util_idle_pct: float = 10.0    # 低于则判定为 idle

    # ── 推理参数调整 ────────────────────────────────────
    # normal 状态下的默认值
    default_max_tokens: int = 200
    default_context_len: int = 2048

    # warning 状态下自动调整 (缩放系数)
    warn_context_scale: float = 0.5    # context 缩到 50%
    warn_max_tokens_scale: float = 0.5 # max_tokens 缩到 50%

    # critical 状态
    crit_context_len: int = 512        # 最小 context

    # ── 模型档位动态选择 ────────────────────────────────
    # 老师要求: 根据 NPU/温度/内存 实时调节"模型的选择"。
    # 这里定义各状态推荐使用的模型档位名(对应后端里可切换的不同 .gguf/量化)。
    # level -> model tier 名。normal/idle 用大模型, warning 降级, critical 用最小。
    model_tiers: dict = None            # {level: name}, None 则走 _default_model_tiers
    # 每个档位对应的可选模型名(需与后端可加载的模型名一致), 供调度员了解
    model_tier_catalog: dict = None     # {tier: {model, note}}

    # ── 滞回控制 (防抖动) ───────────────────────────────
    # 退出 critical 需要满足所有条件持续 N 秒
    stability_sec: float = 30.0

    # ── 其他 ────────────────────────────────────────────
    # RKNN 频率档位 (MHz)，按你的 DTS 实际值调整
    npu_freq_high_mhz: int = 900
    npu_freq_mid_mhz: int = 600
    npu_freq_low_mhz: int = 300

    def __post_init__(self):
        """默认档位表 + 从环境变量覆盖"""
        def _default_tiers():
            return {
                "idle": "full",
                "normal": "full",
                "warning": "mid",
                "critical": "tiny",
            }
        def _default_catalog():
            return {
                "full":   {"model": "qwen2.5-1.5b-instruct-q4_k_m", "note": "正常/空闲: 全规格"},
                "mid":    {"model": "qwen2.5-1.5b-instruct-q2_k",   "note": "告警: 降量化省内存"},
                "tiny":   {"model": "qwen2.5-0.5b-instruct",         "note": "临界: 最小模型"},
            }
        if self.model_tiers is None:
            self.model_tiers = _default_tiers()
        if self.model_tier_catalog is None:
            self.model_tier_catalog = _default_catalog()
        for field_name in self.__dataclass_fields__:
            env_key = f"RK3588_{field_name.upper()}"
            if env_key in os.environ:
                val = os.environ[env_key]
                ftype = type(getattr(self, field_name))
                if ftype is bool:
                    setattr(self, field_name, val.lower() in ("1", "true", "yes"))
                else:
                    setattr(self, field_name, ftype(val))


# ── 全局单例 ────────────────────────────────────────────
_config: ControllerConfig = None


def get_config() -> ControllerConfig:
    """获取全局配置单例"""
    global _config
    if _config is None:
        _config = ControllerConfig()
    return _config
