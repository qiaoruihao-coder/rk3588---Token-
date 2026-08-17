"""
Actuator — 调控执行器
====================
接收 Decision，执行实际的参数调整和后端切换。

只在值真正变化时才写入 sysfs/切换后端，避免无谓开销。
"""

import os
import time
from typing import Optional

from .evaluator import Decision
from .config import get_config

_SYSFS_DEVFREQ = "/sys/class/devfreq"


class Actuator:
    """
    调控执行器。

    用法:
        act = Actuator()
        act.apply(decision)  # 执行决策
        backend = act.get_backend()  # 获取当前推理后端
    """

    def __init__(self):
        cfg = get_config()
        self._npu_devfreq_path: Optional[str] = None
        self._freq_map = {
            "high": cfg.npu_freq_high_mhz,
            "mid": cfg.npu_freq_mid_mhz,
            "low": cfg.npu_freq_low_mhz,
        }
        self._current_backend: str = "npu"
        self._current_freq_level: Optional[str] = None
        self._current_ctx_len: int = cfg.default_context_len
        self._current_max_tokens: int = cfg.default_max_tokens

        # 找到 NPU devfreq 节点
        self._npu_devfreq_path = self._find_npu_devfreq()

    # ── 主入口 ────────────────────────────────────────
    def apply(self, d: Decision) -> dict:
        """
        执行决策，返回实际生效的参数摘要。

        Returns:
            {"backend": "npu", "context_len": 2048, "max_tokens": 200, ...}
        """
        result = {
            "backend": d.backend,
            "context_len": d.context_len,
            "max_tokens": d.max_tokens,
            "can_infer": d.can_infer,
            "level": d.level,
        }

        # 后端切换
        if d.backend != self._current_backend and d.can_infer:
            self._switch_backend(d.backend)
            result["backend_switched"] = True

        # NPU 频率 (只在用 NPU 时调整)
        if d.backend == "npu" and d.npu_freq_level:
            self._set_npu_freq(d.npu_freq_level)
            result["npu_freq_level"] = d.npu_freq_level

        # 推理参数 (上层传给后端用)
        self._current_ctx_len = d.context_len
        self._current_max_tokens = d.max_tokens
        result["context_len"] = d.context_len
        result["max_tokens"] = d.max_tokens

        return result

    # ── 后端切换 ──────────────────────────────────────
    def _switch_backend(self, target: str) -> None:
        """
        切换推理后端。
        cpu → npu: 激活 RKNN (如果模型已加载)
        npu → cpu: 标记切换，RKNN 模型保持在内存 (4GB 限制下不能双开)
        """
        # 标记当前后端，实际路由由 Controller 处理
        self._current_backend = target

    def get_backend(self) -> str:
        return self._current_backend

    # ── NPU 频率 ──────────────────────────────────────
    def _set_npu_freq(self, level: str) -> bool:
        """设置 NPU 频率档位。返回值: True 表示成功写入"""
        if level == self._current_freq_level:
            return True  # 无变化，跳过

        freq_mhz = self._freq_map.get(level)
        if freq_mhz is None or self._npu_devfreq_path is None:
            return False

        # 尝试设 max_freq (限制上限)
        max_freq_path = os.path.join(self._npu_devfreq_path, "max_freq")
        if os.path.isfile(max_freq_path):
            try:
                with open(max_freq_path, "w") as f:
                    f.write(str(freq_mhz * 1_000_000))
                self._current_freq_level = level
                return True
            except PermissionError:
                pass  # 没有 root 权限，静默

        # 尝试设 userspace governor 的目标频率
        gov_path = os.path.join(self._npu_devfreq_path, "governor")
        if os.path.isfile(gov_path):
            try:
                with open(gov_path, "w") as f:
                    f.write("userspace")
            except Exception:
                pass

        target_path = os.path.join(self._npu_devfreq_path, "userspace", "set_freq")
        if os.path.isfile(target_path):
            try:
                with open(target_path, "w") as f:
                    f.write(str(freq_mhz * 1_000_000))
                self._current_freq_level = level
                return True
            except PermissionError:
                pass

        return False

    # ── NPU devfreq 发现 ──────────────────────────────
    def _find_npu_devfreq(self) -> Optional[str]:
        if not os.path.isdir(_SYSFS_DEVFREQ):
            return None
        for name in os.listdir(_SYSFS_DEVFREQ):
            if "npu" in name.lower():
                return os.path.join(_SYSFS_DEVFREQ, name)
        return None

    # ── 当前参数 (给 Controller 查询) ─────────────────
    @property
    def context_len(self) -> int:
        return self._current_ctx_len

    @property
    def max_tokens(self) -> int:
        return self._current_max_tokens
