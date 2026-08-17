"""
1.1 Module: Output Token Control (Ollama Edition)
==================================================
Memory target: <= 1.5GB (Qwen2.5:1.5b quantized via Ollama)
Speed target: TTFT reduced 75% vs unconstrained

API:
  generate_json(prompt, schema)           -> dict
  generate_json_pydantic(prompt, model)   -> dict  (Pydantic, Zhang)
  generate_label(prompt, labels)          -> str
  generate_value(prompt, schema)          -> dict
  generate_diff(prompt, old_code)         -> str
"""

import json, re, time, psutil, subprocess
from openai import OpenAI
from pydantic import BaseModel, ValidationError

# Ollama's OpenAI-compatible endpoint
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:1.5b"


def get_ollama_memory() -> float:
    """Get Ollama process memory in GB."""
    for proc in psutil.process_iter(['name', 'memory_info']):
        if 'ollama' in proc.info['name'].lower():
            return proc.info['memory_info'].rss / 1e9
    return 0.0


# ============================================================
# JSON constraint (via Ollama JSON mode)
# ============================================================
def generate_json(prompt: str, schema: dict, retries: int = 3) -> dict:
    """
    Force JSON output via Ollama's native JSON mode.
    """
    schema_str = json.dumps(schema)
    system_msg = (
        f"You are a JSON-only assistant. Always output valid JSON matching this schema:\n"
        f"{schema_str}\n"
        f"Do NOT include markdown, explanations, or any text outside the JSON object."
    )

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=200
            )
            text = resp.choices[0].message.content.strip()
            # Remove markdown fences if present
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)

            # Validate required fields
            if "required" in schema:
                for field in schema["required"]:
                    if field not in parsed:
                        raise KeyError(f"Missing required field: {field}")
            return parsed

        except (json.JSONDecodeError, KeyError) as e:
            if attempt < retries:
                print(f"  Retry {attempt}/{retries}: {type(e).__name__}")
            else:
                raise RuntimeError(f"JSON generation failed after {retries} retries: {e}")


# ============================================================
# Pydantic JSON constraint (Zhang Zhuoming)
# ============================================================
def generate_json_pydantic(prompt: str, model_class, retries: int = 3) -> dict:
    """
    Force model output via Pydantic model. Type-safe with auto-validation.

    Unlike generate_json() which takes a raw dict schema, this takes a
    Pydantic BaseModel subclass. Schema is auto-generated.

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
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema)

    system_msg = (
        f"You are a JSON-only assistant. Output valid JSON matching this schema:\n"
        f"{schema_str}\n"
        f"Do NOT include markdown, explanations, or any text outside the JSON object."
    )

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=200
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            parsed = json.loads(text)
            # Pydantic validation
            model_class(**parsed)
            return parsed

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < retries:
                print(f"  Retry {attempt}/{retries}: {type(e).__name__}")
            else:
                raise RuntimeError(f"Pydantic JSON generation failed after {retries} retries: {e}")


# ============================================================
# Label constraint
# ============================================================
def generate_label(prompt: str, labels: list[str], retries: int = 3) -> str:
    """Force model to output exactly one of the given labels."""
    system_msg = (
        f"You are a classifier. Reply with EXACTLY ONE WORD from this list: {labels}.\n"
        f"Do NOT add any explanation, punctuation, or other text."
    )

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            text = resp.choices[0].message.content.strip().lower()
            # Find which label matches
            for label in labels:
                if label.lower() in text:
                    return label
            raise ValueError(f"Output '{text}' not in {labels}")
        except ValueError as e:
            if attempt < retries:
                print(f"  Retry {attempt}/{retries}: {e}")
            else:
                raise


# ============================================================
# Value constraint
# ============================================================
def generate_value(prompt: str, schema: dict = None, retries: int = 3) -> dict:
    """Force structured numeric output."""
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
# Diff output (Ollama doesn't support regex constraints, use prompt)
# ============================================================
def generate_diff(prompt: str, old_code: str = "") -> str:
    """Generate unified diff via prompt engineering."""
    if old_code:
        full_prompt = (
            f"Original code:\n```\n{old_code}\n```\n\n"
            f"Change requested: {prompt}\n\n"
            f"Reply with ONLY a unified diff (no markdown, no explanation):\n"
            f"--- a/code.py\n+++ b/code.py\n@@ -1,1 +1,1 @@\n-old line\n+new line"
        )
    else:
        full_prompt = f"Output ONLY a unified diff for: {prompt}\n--- a/file\n+++ b/file\n@@"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0, max_tokens=200
    )
    raw = resp.choices[0].message.content.strip()

    # Strip markdown fences
    raw = re.sub(r'^```(?:diff)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    # Extract diff
    pattern = r"---[^\n]*\n\+\+\+[^\n]*\n@@[^\n]*@@(?:\n[ \-\+][^\n]*)*"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        return match.group()
    idx = raw.find("---")
    return raw[idx:] if idx >= 0 else raw[:300]


# ============================================================
# Performance test
# ============================================================
def perf_test():
    """Memory + speed benchmark."""
    print("=" * 55)
    print("Ollama Edition: Performance Test")
    print("=" * 55)

    # Memory
    mem = get_ollama_memory()
    print(f"\n  Ollama process memory: {mem:.2f} GB")
    print(f"  Target <= 1.5GB: {'PASS' if mem <= 1.5 else 'FAIL'}")

    # Speed
    schema = {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}

    t0 = time.time()
    r = generate_json("Output student Zhangsan, age 20", schema)
    t = time.time() - t0
    print(f"\n  JSON generation: {t:.2f}s")
    print(f"  Result: {r}")

    # Label speed
    t0 = time.time()
    l = generate_label("Product has crack. Quality?", ["qualified", "defective"])
    t2 = time.time() - t0
    print(f"\n  Label generation: {t2:.2f}s")
    print(f"  Result: {l}")

    print("=" * 55)


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("1.1 Module: Ollama Edition")
    print(f"Model: {MODEL}")
    print("=" * 55)

    print("\n[1] generate_json:")
    r = generate_json(
        "Output student info: Zhangsan, age 20, CS major",
        {"type": "object",
         "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "major": {"type": "string"}},
         "required": ["name", "age", "major"]}
    )
    print(f"  -> {r}")

    print("\n[2] generate_label:")
    r2 = generate_label("Camera found crack. Quality?", ["qualified", "defective"])
    print(f"  -> {r2}")

    print("\n[3] generate_value (industrial):")
    r3 = generate_value("Crack 12.5mm2 on bearing surface")
    print(f"  -> {r3}")

    print("\n[4] generate_diff:")
    r4 = generate_diff("rename x to y", old_code="x = 1\nprint(x)")
    print(f"  -> {r4[:200]}")

    print("\n")
    perf_test()
