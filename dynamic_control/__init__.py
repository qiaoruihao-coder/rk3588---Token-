"""
RK3588 模型运行时动态控制系统
===============================
动态采集 NPU/CPU 温度、频率、内存，自动调整推理参数。

架构:
  Monitor (采集) → Evaluator (决策) → Actuator (执行) → Backend (推理)

约束 (RK3588 / 4GB / 室内空调):
  - RKNN 主推理，CPU 兜底（不双开，内存不够）
  - 请求触发采集，无后台线程
  - 温度: warn 65°C / crit 75°C
  - 内存: warn <600MB / crit <300MB
"""

from .controller import DynamicController
from .monitor import SystemState

__all__ = ["DynamicController", "SystemState"]
