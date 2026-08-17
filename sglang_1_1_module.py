"""
1.1 Module: Output Token Control (RK3588 Edition)
==================================================
Authors: Qiao Ruihao, Zhang Zhuoming

Force LLM to output only what you want - no verbose text.
RK3588 native: uses RKNN NPU + dynamic control + CPU fallback.

Memory target: < 1GB (Qwen2.5:1.5b int8 on NPU, 4GB total)
Speed target: TTFT < 1s on NPU, < 3s on CPU fallback

API (unchanged — compatible with scheduler 2.1):
  generate_json(prompt, schema)           -> dict
  generate_json_pydantic(prompt, model)   -> dict  (Pydantic, Zhang)
  generate_label(prompt, labels)          -> str
  generate_value(prompt, schema)          -> dict
  generate_diff(prompt, old_code)         -> str
"""

import json
import re
import time
import os
import sys

# ── Internal: import dynamic controller ───────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from dynamic_control.controller import DynamicController

# ── Global controller singleton ────────────────────────────
_ctrl: DynamicController = None
_MODEL_PATH = os.environ.get(
    "RK3588_MODEL_PATH",
    os.path.join(_current_dir, "qwen2.5-1.5b.rknn")
)


def _get_ctrl() -> DynamicController:
    """Singleton: init once, reuse across all calls."""
    global _ctrl
    if _ctrl is None:
        _ctrl = DynamicController(_MODEL_PATH)
        ok = _ctrl.init()
        if not ok:
            print("[WARN] No backend available — requests may fail")
    return _ctrl


# ============================================================
# JSON constraint
# ============================================================
def generate_json(prompt: str, schema: dict, retries: int = 3) -> dict:
    """
    Force model to output a valid JSON object matching the schema.
    Auto-retries on failure. Uses dynamic control (NPU / CPU fallback).

    Example:
        schema = {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}
        generate_json("output Zhangsan info", schema)
        -> {'name': 'Zhangsan', 'age': 20}
    """
    ctrl = _get_ctrl()
    schema_str = json.dumps(schema)

    for attempt in range(1, retries + 1):
        try:
            result = ctrl.generate_json(prompt, schema, retries=1)
            if "required" in schema:
                for field in schema["required"]:
                    if field not in result:
                        raise KeyError(f"Missing required field: {field}")
            return result
        except RuntimeError as e:
            # "推理被拒绝" — don't retry, propagate
            if "拒绝" in str(e):
                raise
            if attempt >= retries:
                raise RuntimeError(
                    f"generate_json failed after {retries} retries: {e}"
                )
        except (json.JSONDecodeError, KeyError) as e:
            if attempt >= retries:
                raise RuntimeError(
                    f"generate_json failed after {retries} retries: {e}"
                )

    raise RuntimeError("Unreachable")


# ============================================================
# Pydantic JSON constraint (Zhang Zhuoming)
# ============================================================
def generate_json_pydantic(prompt: str, model_class, retries: int = 3) -> dict:
    """
    Force model output via Pydantic model. Type-safe with auto-validation.

    Args:
        prompt: user input
        model_class: Pydantic BaseModel subclass
        retries: max retry attempts

    Returns:
        dict: validated result

    Example:
        >>> from pydantic import BaseModel
        >>> class Student(BaseModel):
        ...     name: str
        ...     age: int
        >>> generate_json_pydantic("Output Zhangsan info", Student)
        {'name': 'Zhangsan', 'age': 20}
    """
    ctrl = _get_ctrl()

    for attempt in range(1, retries + 1):
        try:
            result = ctrl.generate_json_pydantic(prompt, model_class, retries=1)
            return result
        except RuntimeError as e:
            if "拒绝" in str(e):
                raise
            if attempt >= retries:
                raise RuntimeError(
                    f"generate_json_pydantic failed after {retries} retries: {e}"
                )


# ============================================================
# Label constraint
# ============================================================
def generate_label(prompt: str, labels: list[str], retries: int = 3) -> str:
    """
    Force model to output exactly one of the given labels.

    Example:
        generate_label("product has crack, quality?", ["qualified", "defective"])
        -> 'defective'
    """
    ctrl = _get_ctrl()

    for attempt in range(1, retries + 1):
        try:
            result = ctrl.generate_label(prompt, labels, retries=1)
            if result in labels:
                return result
            raise ValueError(f"Output '{result}' not in {labels}")
        except RuntimeError as e:
            if "拒绝" in str(e):
                raise
            if attempt >= retries:
                raise RuntimeError(f"Label generation failed: {e}")
        except ValueError:
            if attempt >= retries:
                raise


# ============================================================
# Value constraint
# ============================================================
def generate_value(prompt: str, schema: dict = None, retries: int = 3) -> dict:
    """
    Force structured numeric output.

    Example:
        generate_value("crack 12.5mm2 on product, judge quality")
        -> {'area_mm2': 12.5, 'quality': 'defective'}
    """
    if schema is None:
        schema = {
            "type": "object",
            "properties": {
                "area_mm2": {"type": "number"},
                "quality": {"type": "string", "enum": ["qualified", "defective"]}
            },
            "required": ["area_mm2", "quality"]
        }
    return generate_json(prompt, schema, retries)


# ============================================================
# Diff output
# ============================================================
def generate_diff(prompt: str, old_code: str = "") -> str:
    """
    Generate unified diff output.

    Example:
        generate_diff("rename 'old_name' to 'new_name'",
                       old_code="old_name: str = 'hello'")
        -> '--- a/code.py\\n+++ b/code.py\\n...'
    """
    ctrl = _get_ctrl()

    if old_code:
        full_prompt = (
            f"Original code:\n```\n{old_code}\n```\n\n"
            f"Change requested: {prompt}\n\n"
            f"Output ONLY a unified diff:\n"
            f"--- a/code.py\n+++ b/code.py\n@@ -1,1 +1,1 @@\n-old line\n+new line"
        )
    else:
        full_prompt = (
            f"Output a unified diff for this change: {prompt}\n"
            f"--- a/file\n+++ b/file\n@@"
        )

    raw = ctrl.generate_diff(prompt, old_code)

    # Clean up
    raw = re.sub(r'^```(?:diff)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)

    # Extract diff portion
    pattern = r"---[^\n]*\n\+\+\+[^\n]*\n@@[^\n]*@@(?:\n[ \-\+][^\n]*)*"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        return match.group()

    idx = raw.find("---")
    return raw[idx:] if idx >= 0 else raw[:300]


# ============================================================
# System status (for scheduler)
# ============================================================
def get_system_status() -> dict:
    """Return current system status. Call before sending requests."""
    ctrl = _get_ctrl()
    return ctrl.status()


def get_health() -> dict:
    """Lightweight health check."""
    ctrl = _get_ctrl()
    return ctrl.health()


# ============================================================
# Batch test
# ============================================================
def batch_test(n: int = 10):
    """Run all 5 functions n times, report pass rate."""
    print("=" * 55)
    print(f"1.1 Batch Test: {n} runs per function (RK3588)")
    print(f"Model: {_MODEL_PATH}")
    print("=" * 55)

    ctrl = _get_ctrl()
    health = ctrl.health()
    print(f"  Backend: {health['backend']} | Level: {health['level']}")
    print(f"  NPU available: {health['npu_avail']} | "
          f"Memory: {health['mem_avail_gb']:.2f} GB")
    print()

    start = time.time()

    schema_student = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "major": {"type": "string"}
        },
        "required": ["name", "age", "major"]
    }
    labels_qc = ["qualified", "defective"]

    results = {
        "generate_json": 0,
        "generate_label": 0,
        "generate_value": 0,
        "generate_diff": 0
    }

    for i in range(1, n + 1):
        # test 1: json
        try:
            r = generate_json(
                f"Output student info: Zhangsan, age 20, CS major (run {i})",
                schema_student, retries=1
            )
            if r.get("name"):
                results["generate_json"] += 1
        except Exception:
            pass

        # test 2: label
        try:
            r = generate_label(
                f"Camera found crack on product surface. Quality? (run {i})",
                labels_qc, retries=1
            )
            if r in labels_qc:
                results["generate_label"] += 1
        except Exception:
            pass

        # test 3: value
        try:
            r = generate_value(
                f"Crack detected, area 12.5 mm2 (run {i})", retries=1
            )
            if r.get("area_mm2"):
                results["generate_value"] += 1
        except Exception:
            pass

        # test 4: diff
        try:
            r = generate_diff(
                f"rename x to y (run {i})", old_code="x = 1\nprint(x)"
            )
            if r.startswith("---"):
                results["generate_diff"] += 1
        except Exception:
            pass

        if i % 3 == 0:
            print(f"  {i}/{n} done...")

    elapsed = time.time() - start
    print(f"\nResults ({elapsed:.1f}s total, ~{elapsed/n:.1f}s/req):")
    for name, count in results.items():
        pct = count / n * 100
        bar = "▓" * count + "░" * (n - count)
        print(f"  {name:20s} [{bar}] {pct:.0f}% ({count}/{n})")

    print("=" * 55)
    return results


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    from pydantic import BaseModel

    class Student(BaseModel):
        name: str
        age: int
        major: str

    print("=" * 55)
    print("1.1 Module: RK3588 Edition")
    print("Authors: Qiao Ruihao + Zhang Zhuoming")
    print(f"Model: {_MODEL_PATH}")
    print("=" * 55)

    # Show system status
    ctrl = _get_ctrl()
    health = ctrl.health()
    print(f"\n  Backend: {health['backend']} | Level: {health['level']}")
    print(f"  NPU: {'OK' if health['npu_avail'] else 'unavailable'} | "
          f"CPU: {'OK' if health['cpu_avail'] else 'unavailable'}")
    print(f"  Memory available: {health['mem_avail_gb']:.2f} GB")

    print("\n[1] generate_json:")
    r = generate_json(
        "Output student info: Zhangsan, age 20, CS major",
        {"type": "object",
         "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "major": {"type": "string"}},
         "required": ["name", "age", "major"]}
    )
    print(f"  -> {r}")

    print("\n[2] generate_json_pydantic (Zhang):")
    r_pyd = generate_json_pydantic("Output student Zhangsan, age 20, CS major", Student)
    print(f"  -> {r_pyd}")

    print("\n[3] generate_label:")
    r2 = generate_label("Camera found crack on product surface. Quality?", ["qualified", "defective"])
    print(f"  -> {r2}")

    print("\n[4] generate_value (industrial):")
    r3 = generate_value("Crack 12.5mm2 on bearing surface")
    print(f"  -> {r3}")

    print("\n[5] generate_diff:")
    r4 = generate_diff("rename x to y", old_code="x = 1\nprint(x)")
    print(f"  -> {r4[:200]}")

    print("\n[6] System status:")
    status = get_system_status()
    print(f"  Level: {status['decision_level']} | Backend: {status['backend']}")
    print(f"  Params: ctx={status['params']['context_len']}, tokens={status['params']['max_tokens']}")
    print(f"  Stats: {status['stats']}")

    print("\n")
    batch_test(10)
