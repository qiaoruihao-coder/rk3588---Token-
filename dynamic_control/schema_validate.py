"""
JSON Schema 类型校验器
=========================
对 LLM/规则后端返回的 dict 做"字段存在 + 字段类型 + enum 合法值"校验，
防止 {"name": 123}(string 字段给了 int) 这类类型错乱被静默当成正确结果。

支持子集(覆盖 1.1 模块工业/交通常用 schema)：
  - type: string / integer / number / boolean / array / object / null
  - integer 接受承载为 float 的整值(20.0→20)，但排除 bool(True 是 int 子类)
  - enum: 值必须属于给定集合
  - required: 必填字段必须存在
  - 嵌套 properties / items 递归
"""
from typing import Any, Optional


class SchemaValidationError(ValueError):
    """校验失败，携带具体路径与原因。"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def validate_schema(data: Any, schema: Optional[dict], path: str = "$") -> None:
    """校验 data 是否符合 JSON Schema 子集。不符则抛 SchemaValidationError。"""
    if schema is None:
        return

    # ── 顶层/逐字段 type 校验 ──────────────────────
    type_ok, expected = _check_type(data, schema.get("type"))
    if not type_ok:
        raise SchemaValidationError(f"{path}: 应为 {expected}，实际得到 {_type_name(data)}")

    # ── enum ─────────────────────────────────────────
    if "enum" in schema and data not in schema["enum"]:
        raise SchemaValidationError(f"{path}: 值 {data!r} 不在 enum {schema['enum']}")

    # ── object: 递归 properties + required ───────────
    if isinstance(data, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in data:
                raise SchemaValidationError(f"{path}: 缺少必填字段 {req!r}")
        for key, val in data.items():
            if key in props:
                validate_schema(val, props[key], f"{path}.{key}")

    # ── array: 递归 items ────────────────────────────
    if isinstance(data, list):
        items = schema.get("items")
        if items:
            for i, it in enumerate(data):
                validate_schema(it, items, f"{path}[{i}]")


def _check_type(value: Any, t: Optional[str]) -> tuple:
    if t is None:
        return True, t
    if t == "string":
        return isinstance(value, str), "string"
    if t == "integer":
        # int(含 bool 除外的 int)；接收收纳在 float 的整值(20.0)
        if isinstance(value, bool):
            return False, "integer"
        if isinstance(value, int):
            return True, "integer"
        if isinstance(value, float) and value.is_integer():
            return True, "integer"
        return False, "integer"
    if t == "number":
        if isinstance(value, bool):
            return False, "number"
        return isinstance(value, (int, float)), "number"
    if t == "boolean":
        return isinstance(value, bool), "boolean"
    if t == "array":
        return isinstance(value, list), "array"
    if t == "object":
        return isinstance(value, dict), "object"
    if t == "null":
        return value is None, "null"
    return True, t  # 未知类型不拦截


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def coerce_int_float(value: Any) -> Any:
    """将承载为 float 的整值归一化为 int(如 20.0→20)。"""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def apply_coercion(data: Any, schema: Optional[dict]) -> Any:
    """递归把 20.0(integer 字段)归一化为 20，返回新结构(不改原数据)。"""
    if schema is None:
        return data
    if isinstance(data, dict):
        out = {}
        props = schema.get("properties", {})
        for key, val in data.items():
            ps = props.get(key, {}) if key in props else None
            out[key] = apply_coercion(val, ps)
        return out
    if isinstance(data, list):
        items = schema.get("items")
        return [apply_coercion(it, items) for it in data]
    if isinstance(data, float) and data.is_integer() and not isinstance(data, bool):
        return int(data)
    return data
