"""
DynamicController 演示脚本
==========================
模拟不同负载下的动态控制行为。

运行: python demo_dynamic_control.py
"""

import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_control.monitor import Monitor, SystemState
from dynamic_control.evaluator import Evaluator, Decision
from dynamic_control.config import get_config


def simulate_state(**overrides) -> SystemState:
    """快捷构造模拟状态"""
    defaults = {
        "npu_util_pct": 40, "npu_freq_mhz": 900,
        "cpu_temp_c": 45, "npu_temp_c": 48,
        "cpu_big_freq_mhz": 1800, "cpu_little_freq_mhz": 408,
        "mem_total_gb": 4.0, "mem_avail_gb": 1.5, "mem_used_pct": 62.5,
    }
    defaults.update(overrides)
    return SystemState(**defaults)


def main():
    cfg = get_config()
    ev = Evaluator()
    mon = Monitor()  # 用于采集真实状态 (如果有)

    print("=" * 60)
    print("  RK3588 Dynamic Controller — Demo")
    print(f"  Config: warn={cfg.temp_warn_c}°C, crit={cfg.temp_crit_c}°C")
    print(f"          mem_warn={cfg.mem_warn_gb}GB, mem_crit={cfg.mem_crit_gb}GB")
    print("=" * 60)

    # ── 场景1: 正常状态 ──────────────────────────────────
    print("\n" + "─" * 40)
    print("场景 1: 正常负载 (NPU 40%, 温度 45°C, 可用 1.5GB)")
    print("─" * 40)
    s1 = simulate_state()
    d1 = ev.evaluate(s1)
    print(f"  系统等级: {s1.level}")
    print(f"  决策: backend={d1.backend}, ctx={d1.context_len}, "
          f"tokens={d1.max_tokens}, freq={d1.npu_freq_level}")
    print(f"  预期: NPU 后端 + 全规格 (ctx=2048, tokens=200)")

    # ── 场景2: NPU 过载 ──────────────────────────────────
    print("\n" + "─" * 40)
    print("场景 2: NPU 过载 (NPU 95%, 温度 55°C)")
    print("─" * 40)
    s2 = simulate_state(npu_util_pct=95, npu_temp_c=55)
    d2 = ev.evaluate(s2)
    print(f"  系统等级: {s2.level}")
    print(f"  决策: backend={d2.backend}, ctx={d2.context_len}, "
          f"tokens={d2.max_tokens}, freq={d2.npu_freq_level}")
    print(f"  预期: NPU 后端 + 半规格 + 降频")

    # ── 场景3: 温度告警 ──────────────────────────────────
    print("\n" + "─" * 40)
    print("场景 3: 温度告警 (CPU 68°C, NPU 66°C)")
    print("─" * 40)
    s3 = simulate_state(cpu_temp_c=68, npu_temp_c=66)
    d3 = ev.evaluate(s3)
    print(f"  系统等级: {s3.level}")
    print(f"  决策: backend={d3.backend}, ctx={d3.context_len}, "
          f"tokens={d3.max_tokens}, freq={d3.npu_freq_level}")
    print(f"  预期: 仍用 NPU 但大幅缩 context")

    # ── 场景4: 内存临界 ──────────────────────────────────
    print("\n" + "─" * 40)
    print("场景 4: 内存不足 (可用 0.2GB)")
    print("─" * 40)
    s4 = simulate_state(mem_avail_gb=0.2, mem_used_pct=95)
    d4 = ev.evaluate(s4)
    print(f"  系统等级: {s4.level}")
    print(f"  决策: backend={d4.backend}, ctx={d4.context_len}, "
          f"tokens={d4.max_tokens}, can_infer={d4.can_infer}")
    if d4.refuse_reason:
        print(f"  拒绝原因: {d4.refuse_reason}")
    print(f"  预期: 切 CPU + 极小 context，或拒绝")

    # ── 场景5: 极端过热 ──────────────────────────────────
    print("\n" + "─" * 40)
    print("场景 5: CPU 过热 (CPU 86°C)")
    print("─" * 40)
    s5 = simulate_state(cpu_temp_c=86)
    d5 = ev.evaluate(s5)
    print(f"  系统等级: {s5.level}")
    print(f"  决策: backend={d5.backend}, can_infer={d5.can_infer}")
    if d5.refuse_reason:
        print(f"  拒绝原因: {d5.refuse_reason}")
    print(f"  预期: 拒绝所有请求")

    # ── 场景6: 温度恢复 (滞回测试) ────────────────────────
    print("\n" + "─" * 40)
    print("场景 6: 滞回测试 — 温度从临界恢复到正常")
    print("─" * 40)

    s_crit = simulate_state(cpu_temp_c=80)
    d_crit = ev.evaluate(s_crit)
    print(f"  80°C → level={d_crit.level} (预期 critical)")

    # 立即降到 40°C (应该被滞回阻止)
    s_ok = simulate_state(cpu_temp_c=40)
    d_blocked = ev.evaluate(s_ok)
    print(f"  立即 40°C → level={d_blocked.level} "
          f"(预期 critical, 因滞回阻止)")

    # 模拟等待足够时间后恢复
    ev._level_since = time.time() - ev.stability_sec - 1
    d_ok = ev.evaluate(s_ok)
    print(f"  等待 {ev.stability_sec}s 后 40°C → level={d_ok.level} "
          f"(预期 normal)")

    # ── 真实系统状态 (如果在 RK3588 上运行) ──────────────
    print("\n" + "─" * 40)
    print("场景 7: 当前真实系统状态")
    print("─" * 40)
    try:
        real = mon.summary()
        print(f"  等级: {real['level']}")
        print(f"  NPU: {real['npu_util_pct']}% @ {real['npu_freq_mhz']}MHz, "
              f"{real['npu_temp_c']}°C")
        print(f"  CPU: {real['cpu_temp_c']}°C, "
              f"big={real['cpu_big_freq_mhz']}MHz, "
              f"little={real['cpu_little_freq_mhz']}MHz")
        print(f"  内存: {real['mem_avail_gb']}GB 可用 "
              f"({real['mem_used_pct']}% 已用)")
    except Exception as e:
        print(f"  [非 RK3588 环境，跳过]: {e}")

    # ── 决策对照表 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  决策对照表")
    print("=" * 60)
    print(f"  {'状态':<12} {'后端':<8} {'Context':<10} {'Tokens':<8} {'NPU频率':<10} {'推理':<8}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")

    scenarios = [
        simulate_state(npu_util_pct=40, cpu_temp_c=45, mem_avail_gb=1.5),
        simulate_state(npu_util_pct=95, cpu_temp_c=55, mem_avail_gb=1.0),
        simulate_state(npu_util_pct=50, cpu_temp_c=68, mem_avail_gb=0.8),
        simulate_state(npu_util_pct=20, cpu_temp_c=78, mem_avail_gb=0.4),
        simulate_state(npu_util_pct=10, cpu_temp_c=86, mem_avail_gb=0.15),
        simulate_state(npu_util_pct=5, cpu_temp_c=40, mem_avail_gb=2.5),
    ]

    for s in scenarios:
        d = ev.evaluate(s)
        print(f"  {s.level:<12} {d.backend:<8} {d.context_len:<10} {d.max_tokens:<8} "
              f"{d.npu_freq_level or '─':<10} {'✓' if d.can_infer else '✗':<8}")

    print("\n  注: 实际阈值可通过环境变量调整")
    print("      export RK3588_TEMP_WARN_C=60")
    print("      export RK3588_MEM_WARN_GB=0.8")
    print("=" * 60)


if __name__ == "__main__":
    main()
