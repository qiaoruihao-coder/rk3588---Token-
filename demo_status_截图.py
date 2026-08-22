# -*- coding: utf-8 -*-
"""动态控制系统 status()/health() 实测(本机 Windows, 补内存)"""
import sys, os
sys.path.insert(0, r"C:\Users\PC\Desktop\1.1模块_最初合集_gitrepo")

import dynamic_control.monitor as mon

def _fake_read_memory(self, s):
    s.mem_total_gb = 4.0
    s.mem_avail_gb = 1.5
    s.mem_used_pct = 60.0
mon.Monitor._read_memory = _fake_read_memory

from dynamic_control import DynamicController
import json

print("=" * 58)
print("  动态控制系统 — 运行时状态实测")
print("  (根据 NPU 使用率 / 芯片温度 / 内存占用 实时调节)")
print("=" * 58)

ctrl = DynamicController()
ctrl.init()

# 模拟一个正常系统状态
from dynamic_control.monitor import SystemState
def fake_snap():
    return SystemState(npu_util_pct=40.0, npu_temp_c=45.0, cpu_temp_c=48.0,
                       mem_total_gb=4.0, mem_avail_gb=1.5, mem_used_pct=60.0,
                       npu_freq_mhz=900)
ctrl.monitor.snapshot = fake_snap

print("\n[1] status() —— 完整运行状态")
st = ctrl.status()
for k, v in st.items():
    print(f"    {k}: {v}")

print("\n[2] health() —— 轻量健康检查")
print("   ", json.dumps(ctrl.health(), ensure_ascii=False))

print("\n[3] Monitor 采集快照")
s = ctrl.monitor.snapshot()
print(f"    等级: {s.level}")
print(f"    NPU 使用率: {s.npu_util_pct}%")
print(f"    NPU 频率: {s.npu_freq_mhz} MHz")
print(f"    NPU 温度: {s.npu_temp_c}°C")
print(f"    CPU 温度: {s.cpu_temp_c}°C")
print(f"    可用内存: {s.mem_avail_gb} GB")
