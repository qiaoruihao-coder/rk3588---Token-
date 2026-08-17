"""
1.1 module: Output Token Control (v2 - with retry)
Authors: Qiao Ruihao, Zhang Zhuoming
=====================================
Force LLM to output only what you want - no verbose text.
All functions tested on Qwen2.5-1.5B, 8GB VRAM.

API:
  generate_json(prompt, schema, retries=3)   -> dict  (outlines)
  generate_json_pydantic(prompt, model)      -> dict  (Pydantic, Zhang)
  generate_value(prompt, schema, retries=3)  -> dict
  generate_label(prompt, labels, retries=3)  -> str
  generate_diff(prompt, old_code)            -> str
  batch_test(n)                              -> run all tests n times
"""

import os
os.environ["CUDA_HOST_COMPILER"] = "/usr/bin/gcc-14"

import re
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import models as ol_models, generate
from pydantic import BaseModel, ValidationError

MODEL_PATH = "/mnt/c/Users/PC/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

_model = None
_tokenizer = None
_ol_model = None


def load_model():
    """Singleton: load once, reuse."""
    global _model, _tokenizer, _ol_model
    if _model is None:
        print("Loading Qwen2.5-1.5B model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True,
            dtype=torch.float16, device_map="auto"
        )
        _ol_model = ol_models.Transformers(_model, _tokenizer)
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"Model loaded! GPU memory: {mem:.1f} GB")
    return _ol_model, _tokenizer, _model


# ============================================================
# Retry wrapper (Day 3)
# ============================================================
def _with_retry(func, max_retries=3, func_name="unknown"):
    """Auto-retry wrapper. If func fails, retry up to max_retries times."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except (json.JSONDecodeError, KeyError, TypeError, RuntimeError) as e:
            last_error = e
            if attempt < max_retries:
                print(f"  [{func_name}] Retry {attempt}/{max_retries}: {type(e).__name__}")
            else:
                print(f"  [{func_name}] FAILED after {max_retries} attempts")
    raise RuntimeError(f"{func_name} failed after {max_retries} retries. Last error: {last_error}")


# ============================================================
# JSON Schema constraint
# ============================================================
def generate_json(prompt: str, schema: dict, retries: int = 3) -> dict:
    """
    Force model to output a valid JSON object matching the schema.
    Auto-retries on failure.

    Example:
        schema = {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}
        generate_json("output Zhangsan info", schema)
        -> {'name': 'Zhangsan', 'age': 20}
    """
    ol_model, _, _ = load_model()
    schema_str = json.dumps(schema)

    def _try():
        gen = generate.json(ol_model, schema_str)
        result = gen(prompt)
        parsed = json.loads(result) if isinstance(result, str) else result
        # Validate all required fields are present
        if "required" in schema:
            for field in schema["required"]:
                if field not in parsed:
                    raise KeyError(f"Missing required field: {field}")
        return parsed

    return _with_retry(_try, retries, "generate_json")


# ============================================================
# Pydantic JSON constraint (Zhang Zhuoming)
# ============================================================
def generate_json_pydantic(prompt: str, model_class, retries: int = 3) -> dict:
    """
    Force model output via Pydantic model. Type-safe with auto-validation.

    Unlike generate_json() which takes a raw dict schema, this takes a
    Pydantic BaseModel subclass. Schema is auto-generated, output is
    validated against the Pydantic class.

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
    ol_model, _, _ = load_model()

    # Auto-generate JSON Schema from Pydantic model
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema)

    def _try():
        gen = generate.json(ol_model, schema_str)
        result = gen(prompt)
        parsed = json.loads(result) if isinstance(result, str) else result
        # Pydantic validation: catches type errors dict schema might miss
        model_class(**parsed)
        return parsed

    return _with_retry(_try, retries, "generate_json_pydantic")


# ============================================================
# Label constraint
# ============================================================
def generate_label(prompt: str, labels: list[str], retries: int = 3) -> str:
    """
    Force model to output exactly one of the given labels.
    Auto-retries on failure.

    Example:
        generate_label("product has crack, quality?", ["qualified", "defective"])
        -> 'defective'
    """
    ol_model, _, _ = load_model()

    def _try():
        gen = generate.choice(ol_model, labels)
        result = gen(prompt)
        if result not in labels:
            raise ValueError(f"Output '{result}' not in allowed labels {labels}")
        return result

    return _with_retry(_try, retries, "generate_label")


# ============================================================
# Value constraint
# ============================================================
def generate_value(prompt: str, schema: dict = None, retries: int = 3) -> dict:
    """
    Force numeric/structured output by wrapping in JSON schema.

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
    Ask model to produce a unified diff. Uses prompt engineering
    then regex-extracts the diff portion from the response.

    Example:
        generate_diff("change age from int to str", old_code="age: int = 20")
        -> '--- a/code.py\n+++ b/code.py\n...'
    """
    _, tokenizer, model = load_model()

    if old_code:
        full_prompt = (
            f"Original code:\n{old_code}\n\n"
            f"Change requested: {prompt}\n\n"
            f"Output ONLY a unified diff:\n"
            f"--- a/code.py\n+++ b/code.py\n@@ -1,1 +1,1 @@\n-old line\n+new line"
        )
    else:
        full_prompt = f"Output a unified diff for this change: {prompt}\n--- a/file\n+++ b/file\n@@"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=120, temperature=0, do_sample=False)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)

    pattern = r"---[^\n]*\n\+\+\+[^\n]*\n@@[^\n]*@@(?:\n[ \-\+][^\n]*)*"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        return match.group()
    idx = raw.find("---")
    return raw[idx:] if idx >= 0 else raw[:300]


# ============================================================
# Batch test (Day 3: run 10 times, count pass/fail)
# ============================================================
def batch_test(n: int = 10):
    """Run all 4 functions n times, report pass rate."""
    print("=" * 55)
    print(f"1.1 Batch Test: {n} runs per function")
    print("=" * 55)
    start = time.time()

    schema_student = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "major": {"type": "string"}},
        "required": ["name", "age", "major"]
    }
    labels_qc = ["qualified", "defective"]

    results = {"generate_json": 0, "generate_json_pydantic": 0, "generate_label": 0, "generate_value": 0, "generate_diff": 0}

    for i in range(1, n + 1):
        # test 1: json
        try:
            r = generate_json(
                f"Output student info: Zhangsan, age 20, CS major (run {i})", schema_student, retries=1
            )
            if r.get("name"):
                results["generate_json"] += 1
        except Exception:
            pass

        # test 2: pydantic
        try:
            r = generate_json_pydantic(
                f"Output student info: Zhangsan, age 20, CS major (run {i})", Student, retries=1
            )
            if r.get("name"):
                results["generate_json_pydantic"] += 1
        except Exception:
            pass

        # test 3: label
        try:
            r = generate_label(
                f"Camera found crack on product surface. Quality? (run {i})", labels_qc, retries=1
            )
            if r in labels_qc:
                results["generate_label"] += 1
        except Exception:
            pass

        # test 3: value
        try:
            r = generate_value(f"Crack detected, area 12.5 mm2 (run {i})", retries=1)
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
    print(f"\nResults ({elapsed:.1f}s total):")
    for name, count in results.items():
        pct = count / n * 100
        bar = "▓" * (count) + "░" * (n - count)
        print(f"  {name:20s} [{bar}] {pct:.0f}% ({count}/{n})")

    print("=" * 55)
    return results


# ============================================================
# Demo (quick)
# ============================================================
if __name__ == "__main__":
    from pydantic import BaseModel

    class Student(BaseModel):
        name: str
        age: int
        major: str

    print("=" * 55)
    print("1.1 Module Demo (v3 with Pydantic)")
    print("Qiao Ruihao + Zhang Zhuoming")
    print("=" * 55)

    # Quick single-run demo
    print("\n[1] generate_json (dict schema):")
    r = generate_json(
        "Output student info: Zhangsan, age 20, CS major",
        {"type": "object",
         "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "major": {"type": "string"}},
         "required": ["name", "age", "major"]}
    )
    print(f"  -> {r}")

    print("\n[2] generate_json_pydantic (Pydantic model, Zhang):")
    r_pyd = generate_json_pydantic("Output student info: Zhangsan, age 20, CS major", Student)
    print(f"  -> {r_pyd}")
    print(f"  -> Type safe: Student(**{r_pyd}) -> OK")

    print("\n[3] generate_label:")
    r2 = generate_label(
        "Camera found crack on product surface. Quality judgement:",
        ["qualified", "defective"]
    )
    print(f"  -> {r2}")

    print("\n[4] generate_value:")
    r3 = generate_value("Crack detected, area 12.5 mm2 on bearing surface")
    print(f"  -> {r3}")

    print("\n[5] generate_diff:")
    r4 = generate_diff(
        "rename 'old_name' to 'new_name'",
        old_code="old_name: str = 'hello'\nprint(old_name)"
    )
    print(f"  -> {r4[:200]}")

    # Batch test: 10 runs
    print("\n")
    batch_test(10)
