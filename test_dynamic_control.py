"""
DynamicController 集成测试
==========================
测试 Monitor、Evaluator、Actuator 的协同工作。

运行: python test_dynamic_control.py
"""

import sys
import os
import time
import json

# 添加父目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_control.monitor import Monitor, SystemState
from dynamic_control.evaluator import Evaluator, Decision
from dynamic_control.actuator import Actuator
from dynamic_control.config import get_config, ControllerConfig


def test_monitor_snapshot():
    """Monitor: 采集系统快照"""
    print("\n" + "=" * 55)
    print("Test 1: Monitor.snapshot()")
    print("=" * 55)

    mon = Monitor()
    s = mon.snapshot()

    assert isinstance(s, SystemState), "Should return SystemState"
    assert s.level in ("idle", "normal", "warning", "critical"), \
        f"Invalid level: {s.level}"
    assert s.mem_total_gb >= 0, "Memory total should be >= 0"
    assert s.timestamp > 0, "Timestamp should be set"

    print(f"  Level: {s.level}")
    print(f"  NPU util: {s.npu_util_pct}%")
    print(f"  NPU freq: {s.npu_freq_mhz} MHz")
    print(f"  NPU temp: {s.npu_temp_c}°C")
    print(f"  CPU temp: {s.cpu_temp_c}°C")
    print(f"  CPU big freq: {s.cpu_big_freq_mhz} MHz")
    print(f"  CPU little freq: {s.cpu_little_freq_mhz} MHz")
    print(f"  Memory: {s.mem_avail_gb:.2f} GB available / {s.mem_total_gb:.2f} GB total")
    print("  PASS")

    return s


def test_monitor_summary():
    """Monitor: summary() 返回 dict"""
    print("\n" + "=" * 55)
    print("Test 2: Monitor.summary()")
    print("=" * 55)

    mon = Monitor()
    summary = mon.summary()

    assert isinstance(summary, dict), "Summary should be dict"
    assert "level" in summary
    assert "mem_avail_gb" in summary
    assert "timestamp" in summary

    print(f"  Summary: {json.dumps(summary, indent=2, default=str)}")
    print("  PASS")


def test_evaluator_decision_for_levels():
    """Evaluator: 各级决策逻辑"""
    print("\n" + "=" * 55)
    print("Test 3: Evaluator decision for each level")
    print("=" * 55)

    ev = Evaluator()
    cfg = get_config()

    # 模拟不同状态
    test_cases = [
        # (state, expected_backend)
        ("normal", SystemState(
            npu_util_pct=50, npu_freq_mhz=900,
            cpu_temp_c=45, npu_temp_c=50,
            mem_avail_gb=2.0, mem_total_gb=4.0, mem_used_pct=50
        ), "npu"),
        ("warning_npu_overload", SystemState(
            npu_util_pct=95, npu_freq_mhz=900,
            cpu_temp_c=50, npu_temp_c=55,
            mem_avail_gb=1.0, mem_total_gb=4.0, mem_used_pct=75
        ), "npu"),
        ("warning_temp", SystemState(
            npu_util_pct=50, npu_freq_mhz=600,
            cpu_temp_c=68, npu_temp_c=66,
            mem_avail_gb=1.0, mem_total_gb=4.0, mem_used_pct=75
        ), "npu"),
        ("critical_mem", SystemState(
            npu_util_pct=30, npu_freq_mhz=300,
            cpu_temp_c=50, npu_temp_c=55,
            mem_avail_gb=0.2, mem_total_gb=4.0, mem_used_pct=95
        ), "cpu"),
        ("critical_overheat", SystemState(
            npu_util_pct=20, npu_freq_mhz=300,
            cpu_temp_c=78, npu_temp_c=79,
            mem_avail_gb=0.8, mem_total_gb=4.0, mem_used_pct=80
        ), "cpu"),
    ]

    for name, state, expected_backend in test_cases:
        d = ev.evaluate(state)
        actual_backend = d.backend
        status = "PASS" if actual_backend == expected_backend else "FAIL"
        print(f"  [{status}] {name}: {state.level} → backend={actual_backend} "
              f"(expected={expected_backend}), "
              f"ctx={d.context_len}, tokens={d.max_tokens}, "
              f"freq={d.npu_freq_level}")


def test_evaluator_hysteresis():
    """Evaluator: 滞回控制"""
    print("\n" + "=" * 55)
    print("Test 4: Evaluator hysteresis (anti-flapping)")
    print("=" * 55)

    ev = Evaluator()

    # 从 normal 直接到 critical (升级应立即生效)
    s_crit = SystemState(
        npu_util_pct=20, cpu_temp_c=80,  # > crit threshold
        mem_avail_gb=1.0, mem_total_gb=4.0, mem_used_pct=75
    )
    d1 = ev.evaluate(s_crit)
    assert d1.level == "critical", f"Upgrade should be instant, got {d1.level}"
    assert d1.backend == "cpu", f"Critical should switch to CPU, got {d1.backend}"
    print(f"  Normal → Critical: instant, backend={d1.backend} ✓")

    # 立即回到 normal (降级应该被滞后阻止)
    s_norm = SystemState(
        npu_util_pct=30, cpu_temp_c=40,
        mem_avail_gb=2.0, mem_total_gb=4.0, mem_used_pct=50
    )
    d2 = ev.evaluate(s_norm)
    assert d2.level == "critical", \
        f"Downgrade should be blocked by hysteresis, got {d2.level}"
    print(f"  Critical → Normal (immediate): blocked, stays {d2.level} ✓")

    # 等待 stability_sec 后再次降级
    ev._level_since = time.time() - ev.stability_sec - 1
    d3 = ev.evaluate(s_norm)
    assert d3.level == "normal", \
        f"After cooldown should downgrade, got {d3.level}"
    print(f"  After {ev.stability_sec}s → Normal: downgraded ✓")


def test_actuator_apply():
    """Actuator: 执行决策"""
    print("\n" + "=" * 55)
    print("Test 5: Actuator.apply()")
    print("=" * 55)

    act = Actuator()

    # normal 决策
    d_normal = Decision(
        backend="npu", max_tokens=200, context_len=2048,
        npu_freq_level="high", can_infer=True
    )
    r = act.apply(d_normal)
    assert r["backend"] == "npu"
    assert r["can_infer"] is True
    print(f"  Normal decision applied: {r}")

    # critical 决策 (拒绝)
    d_refuse = Decision(
        backend="refuse", max_tokens=0, context_len=0,
        can_infer=False, refuse_reason="内存不足"
    )
    r2 = act.apply(d_refuse)
    assert r2["can_infer"] is False
    print(f"  Refuse decision applied: {r2}")

    # warning 决策 (降级)
    d_warn = Decision(
        backend="npu", max_tokens=100, context_len=1024,
        npu_freq_level="mid", can_infer=True
    )
    r3 = act.apply(d_warn)
    assert r3["context_len"] == 1024
    assert r3["max_tokens"] == 100
    print(f"  Warning decision applied: {r3}")


def test_config_env_override():
    """Config: 环境变量覆盖"""
    print("\n" + "=" * 55)
    print("Test 6: Config environment override")
    print("=" * 55)

    os.environ["RK3588_TEMP_WARN_C"] = "70.0"

    # 重新获取 config (这会读环境变量)
    import importlib
    from dynamic_control import config as cfg_module
    importlib.reload(cfg_module)
    cfg = cfg_module.get_config()

    assert cfg.temp_warn_c == 70.0, \
        f"Env override should set temp_warn to 70, got {cfg.temp_warn_c}"
    print(f"  temp_warn_c = {cfg.temp_warn_c} (from env) ✓")

    # 恢复
    del os.environ["RK3588_TEMP_WARN_C"]


def test_controller_lifecycle():
    """Controller: 创建和初始化"""
    print("\n" + "=" * 55)
    print("Test 7: DynamicController lifecycle")
    print("=" * 55)

    from dynamic_control.controller import DynamicController

    ctrl = DynamicController()
    ok = ctrl.init()

    # 即使 RKNN 未加载，也应该至少有一个后端
    status = ctrl.status()
    assert "system" in status
    assert "backend" in status
    assert "stats" in status
    print(f"  Init OK: {ok}")
    print(f"  Status: {json.dumps(status, indent=2, default=str)}")

    # health check
    health = ctrl.health()
    assert "level" in health
    print(f"  Health: {json.dumps(health, indent=2, default=str)}")
    print("  PASS")


def test_decision_output_format():
    """Decision: 输出格式验证"""
    print("\n" + "=" * 55)
    print("Test 8: Decision output validation")
    print("=" * 55)

    d = Decision()
    assert d.backend in ("npu", "cpu", "refuse")
    assert d.max_tokens >= 0
    assert d.context_len >= 0
    assert d.level in ("idle", "normal", "warning", "critical")
    assert isinstance(d.can_infer, bool)
    print(f"  Default decision: backend={d.backend}, ctx={d.context_len}, "
          f"tokens={d.max_tokens}, level={d.level}")
    print("  PASS")


def main():
    print("=" * 55)
    print("Dynamic Controller — Integration Tests")
    print("=" * 55)

    results = []
    tests = [
        test_monitor_snapshot,
        test_monitor_summary,
        test_evaluator_decision_for_levels,
        test_evaluator_hysteresis,
        test_actuator_apply,
        test_config_env_override,
        test_controller_lifecycle,
        test_decision_output_format,
    ]

    for test in tests:
        try:
            test()
            results.append((test.__name__, "PASS"))
        except Exception as e:
            results.append((test.__name__, f"FAIL: {e}"))
            print(f"\n  *** FAILED: {e}")

    print("\n" + "=" * 55)
    print("Summary")
    print("=" * 55)
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    for name, result in results:
        print(f"  [{result[:4]}] {name}")
    print(f"\n  {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
