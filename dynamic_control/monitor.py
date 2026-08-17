"""
RK3588 系统指标采集
==================
请求触发模式: 每次调用 snapshot() 采集一次当前状态。
RK3588 温度 zone 编号因板子而异，这里遍历匹配；找不到就返回 None 不报错。
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional


# ── sysfs 路径 ───────────────────────────────────────────
_SYSFS_THERMAL = "/sys/class/thermal"
_SYSFS_DEVFREQ = "/sys/class/devfreq"
_PROC_MEMINFO = "/proc/meminfo"


# ── 数据结构 ─────────────────────────────────────────────
@dataclass
class SystemState:
    """一次采集的系统快照"""

    # NPU
    npu_util_pct: float = 0.0
    npu_freq_mhz: int = 0
    npu_temp_c: Optional[float] = None

    # CPU
    cpu_temp_c: Optional[float] = None
    cpu_big_freq_mhz: int = 0         # A76 集群
    cpu_little_freq_mhz: int = 0      # A55 集群

    # 内存 (单位 GB)
    mem_total_gb: float = 0.0
    mem_avail_gb: float = 0.0
    mem_used_pct: float = 0.0

    timestamp: float = field(default_factory=time.time)

    # ── 综合判定 ──────────────────────────────────────
    @property
    def level(self) -> str:
        """综合状态等级: idle / normal / warning / critical"""
        from .config import ControllerConfig
        cfg = ControllerConfig()

        # crit 优先判断
        if self.mem_avail_gb < cfg.mem_crit_gb:
            return "critical"
        if self.cpu_temp_c and self.cpu_temp_c > cfg.temp_crit_c:
            return "critical"
        if self.npu_temp_c and self.npu_temp_c > cfg.temp_crit_c:
            return "critical"

        # warning
        if self.mem_avail_gb < cfg.mem_warn_gb:
            return "warning"
        if self.cpu_temp_c and self.cpu_temp_c > cfg.temp_warn_c:
            return "warning"
        if self.npu_temp_c and self.npu_temp_c > cfg.temp_warn_c:
            return "warning"
        if self.npu_util_pct > cfg.npu_util_warn_pct:
            return "warning"

        # idle
        if self.npu_util_pct < cfg.npu_util_idle_pct:
            return "idle"

        return "normal"


# ── 采集函数 ─────────────────────────────────────────────

class Monitor:
    """
    系统指标采集器。

    用法:
        mon = Monitor()
        state = mon.snapshot()   # 请求触发
        print(state.level)       # 'normal'
    """

    def __init__(self):
        self._npu_zone: Optional[str] = None   # 缓存 thermal zone 名
        self._cpu_zone: Optional[str] = None
        self._npu_devfreq: Optional[str] = None
        self._thermal_zones_scanned = False

    # ── 主入口 ────────────────────────────────────────
    def snapshot(self) -> SystemState:
        """采集一次完整系统快照"""
        s = SystemState()

        # NPU 指标 (RKNN API)
        self._read_npu_rknn(s)

        # 温度 (sysfs)
        if not self._thermal_zones_scanned:
            self._scan_thermal_zones()
        s.npu_temp_c = self._read_temp(self._npu_zone)
        s.cpu_temp_c = self._read_temp(self._cpu_zone)

        # NPU 频率 (devfreq)
        s.npu_freq_mhz = self._read_npu_freq()

        # CPU 频率
        s.cpu_big_freq_mhz = self._read_cpu_freq("policy0")    # little
        s.cpu_little_freq_mhz = self._read_cpu_freq("policy0")
        # RK3588: policy0 通常是 little, policy2 是大核; 尝试读
        s.cpu_big_freq_mhz = self._read_cpu_freq("policy2")
        s.cpu_little_freq_mhz = self._read_cpu_freq("policy0")

        # 内存
        self._read_memory(s)

        return s

    # ── NPU 使用率 (RKNN API) ─────────────────────────
    def _read_npu_rknn(self, s: SystemState) -> None:
        """通过 RKNN toolkit 查询 NPU 使用率。需要 RKNN 已初始化。"""
        try:
            from rknn.api import RKNN
            # 尝试获取全局 RKNN 实例
            rknn = RKNN._instance if hasattr(RKNN, '_instance') else None
            if rknn is None:
                return

            # eval_perf 返回 dict，包含 NPU 各核负载
            perf = rknn.eval_perf(is_print=False)
            if isinstance(perf, dict) and len(perf) > 0:
                loads = [v for k, v in perf.items() if 'load' in k.lower()]
                if loads:
                    s.npu_util_pct = sum(loads) / len(loads)
                else:
                    # 取第一个数值字段作为利用率的近似
                    s.npu_util_pct = float(list(perf.values())[0])
        except Exception:
            pass  # RKNN 未初始化或无权限，静默

    # ── thermal zone 扫描 ─────────────────────────────
    def _scan_thermal_zones(self) -> None:
        """扫描 /sys/class/thermal/ 找到 CPU 和 NPU 的 zone"""
        if not os.path.isdir(_SYSFS_THERMAL):
            self._thermal_zones_scanned = True
            return

        for name in os.listdir(_SYSFS_THERMAL):
            type_path = os.path.join(_SYSFS_THERMAL, name, "type")
            if not os.path.isfile(type_path):
                continue
            try:
                with open(type_path) as f:
                    ttype = f.read().strip()
            except Exception:
                continue

            if ttype in ("cpu", "cpu-thermal", "soc-thermal"):
                self._cpu_zone = name
            elif ttype in ("npu", "npu-thermal", "gpu", "gpu-thermal"):
                self._npu_zone = name

        # fallback: 如果没找到 npu zone，用 thermal_zone1
        if self._npu_zone is None:
            for name in os.listdir(_SYSFS_THERMAL):
                if name.startswith("thermal_zone1"):
                    self._npu_zone = name
                    break
        if self._cpu_zone is None:
            for name in os.listdir(_SYSFS_THERMAL):
                if name.startswith("thermal_zone0"):
                    self._cpu_zone = name
                    break

        self._thermal_zones_scanned = True

    def _read_temp(self, zone_name: Optional[str]) -> Optional[float]:
        """读取温度 (m°C → °C)。失败返回 None"""
        if zone_name is None:
            return None
        path = os.path.join(_SYSFS_THERMAL, zone_name, "temp")
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return None

    # ── NPU 频率 ──────────────────────────────────────
    def _read_npu_freq(self) -> int:
        """通过 devfreq 读 NPU 频率 (Hz → MHz)。失败返回 0"""
        if self._npu_devfreq is None:
            self._npu_devfreq = self._find_npu_devfreq()

        if self._npu_devfreq is None:
            return 0

        path = os.path.join(_SYSFS_DEVFREQ, self._npu_devfreq, "cur_freq")
        if not os.path.isfile(path):
            return 0
        try:
            with open(path) as f:
                return int(float(f.read().strip()) / 1_000_000)
        except Exception:
            return 0

    def _find_npu_devfreq(self) -> Optional[str]:
        """在 /sys/class/devfreq/ 中找 NPU 设备"""
        if not os.path.isdir(_SYSFS_DEVFREQ):
            return None
        for name in os.listdir(_SYSFS_DEVFREQ):
            if "npu" in name.lower():
                return name
        return None

    # ── CPU 频率 ──────────────────────────────────────
    def _read_cpu_freq(self, policy: str) -> int:
        """读 CPU 频率 (KHz → MHz)"""
        path = f"/sys/devices/system/cpu/cpufreq/{policy}/scaling_cur_freq"
        if not os.path.isfile(path):
            return 0
        try:
            with open(path) as f:
                return int(float(f.read().strip()) / 1000)
        except Exception:
            return 0

    # ── 内存 ──────────────────────────────────────────
    def _read_memory(self, s: SystemState) -> None:
        """读 /proc/meminfo"""
        meminfo = {}
        try:
            with open(_PROC_MEMINFO) as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = parts[1].strip()
        except Exception:
            return

        total_kb = self._parse_kb(meminfo.get("MemTotal", "0"))
        avail_kb = self._parse_kb(meminfo.get("MemAvailable", "0"))
        # MemAvailable 可能不存在(老内核), fallback: MemFree + Buffers + Cached
        if avail_kb == 0:
            free_kb = self._parse_kb(meminfo.get("MemFree", "0"))
            buf_kb = self._parse_kb(meminfo.get("Buffers", "0"))
            cache_kb = self._parse_kb(meminfo.get("Cached", "0"))
            avail_kb = free_kb + buf_kb + cache_kb

        s.mem_total_gb = total_kb / 1_048_576
        s.mem_avail_gb = avail_kb / 1_048_576
        if total_kb > 0:
            s.mem_used_pct = (total_kb - avail_kb) / total_kb * 100

    @staticmethod
    def _parse_kb(val: str) -> int:
        """解析 '1024 kB' → 1024"""
        return int(val.replace("kB", "").strip()) if val else 0

    # ── 便捷方法 ──────────────────────────────────────
    def quick_check(self) -> str:
        """快速检查，只返回等级字符串 (给调度员用的轻量接口)"""
        s = self.snapshot()
        return s.level

    def summary(self) -> dict:
        """返回 dict 摘要 (给 API 返回用)"""
        s = self.snapshot()
        return {
            "level": s.level,
            "npu_util_pct": s.npu_util_pct,
            "npu_freq_mhz": s.npu_freq_mhz,
            "npu_temp_c": s.npu_temp_c,
            "cpu_temp_c": s.cpu_temp_c,
            "cpu_big_freq_mhz": s.cpu_big_freq_mhz,
            "cpu_little_freq_mhz": s.cpu_little_freq_mhz,
            "mem_avail_gb": round(s.mem_avail_gb, 2),
            "mem_used_pct": round(s.mem_used_pct, 1),
            "timestamp": s.timestamp,
        }
