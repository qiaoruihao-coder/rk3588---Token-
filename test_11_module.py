"""
Unit Tests for 1.1 Module (WSL Edition + Ollama Edition)
========================================================
Run: pytest test_11_module.py -v
"""

import json
import pytest

# ================================================================
# Test data
# ================================================================
STUDENT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "major": {"type": "string"}},
    "required": ["name", "age", "major"]
}

INDUSTRIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "defect_type": {"type": "string", "enum": ["crack", "scratch", "dent", "none"]},
        "area_mm2": {"type": "number"},
        "quality": {"type": "string", "enum": ["qualified", "defective"]}
    },
    "required": ["defect_type", "area_mm2", "quality"]
}

from pydantic import BaseModel

class StudentModel(BaseModel):
    name: str
    age: int
    major: str

class IndustrialModel(BaseModel):
    defect_type: str
    area_mm2: float
    quality: str


TRAFFIC_SCHEMA = {
    "type": "object",
    "properties": {
        "vehicle_count": {"type": "integer"},
        "avg_speed_kmh": {"type": "number"},
        "status": {"type": "string", "enum": ["smooth", "slow", "congested", "blocked"]}
    },
    "required": ["vehicle_count", "avg_speed_kmh", "status"]
}


# ================================================================
# WSL Edition Tests
# ================================================================
class TestWSLEdition:
    """Tests for sglang_1_1_module.py (outlines + transformers)"""

    @pytest.fixture(autouse=True)
    def setup(self):
        import os
        os.environ["CUDA_HOST_COMPILER"] = "/usr/bin/gcc-14"
        import sys
        sys.path.insert(0, "/home/qiaoruihao")
        from sglang_1_1_module import load_model
        load_model()  # pre-load

    def test_generate_json_returns_dict(self):
        from sglang_1_1_module import generate_json
        result = generate_json("Output student Zhangsan, age 20, CS", STUDENT_SCHEMA)
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "major" in result

    def test_generate_json_all_required_fields(self):
        from sglang_1_1_module import generate_json
        result = generate_json("Output student Zhangsan, age 20, CS", STUDENT_SCHEMA)
        for field in STUDENT_SCHEMA["required"]:
            assert field in result, f"Missing required field: {field}"

    def test_generate_json_retry_on_failure(self):
        from sglang_1_1_module import generate_json
        # This should succeed even with a tricky prompt
        result = generate_json("Student: Zhangsan, 20 years old, Computer Science", STUDENT_SCHEMA, retries=3)
        assert isinstance(result, dict)

    def test_generate_json_pydantic_returns_dict(self):
        from sglang_1_1_module import generate_json_pydantic
        result = generate_json_pydantic("Output student Zhangsan, age 20, CS", StudentModel)
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result

    def test_generate_json_pydantic_validates_types(self):
        from sglang_1_1_module import generate_json_pydantic
        result = generate_json_pydantic("Output Zhangsan, 20, CS", StudentModel)
        # Pydantic validation ensures age is int, name is str
        StudentModel(**result)  # won't raise if types are wrong

    def test_generate_label_returns_valid_choice(self):
        from sglang_1_1_module import generate_label
        labels = ["qualified", "defective"]
        result = generate_label("Camera found crack on product. Quality?", labels)
        assert result in labels

    def test_generate_label_multiple_choices(self):
        from sglang_1_1_module import generate_label
        labels = ["smooth", "slow", "congested", "blocked"]
        result = generate_label("Intersection: 31 vehicles, 3km/h. Status?", labels)
        assert result in labels

    def test_generate_value_industrial(self):
        from sglang_1_1_module import generate_value
        result = generate_value("Crack detected, 12.5 mm2 on bearing surface")
        assert isinstance(result, dict)
        assert "area_mm2" in result
        assert "quality" in result
        assert result["quality"] in ["qualified", "defective"]

    def test_generate_diff_format(self):
        from sglang_1_1_module import generate_diff
        result = generate_diff("rename x to y", old_code="x = 1\nprint(x)")
        assert isinstance(result, str)
        assert result.startswith("---")
        assert "+++" in result
        assert "@@" in result

    def test_generate_diff_without_old_code(self):
        from sglang_1_1_module import generate_diff
        result = generate_diff("add a print statement")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_industrial_scenario(self):
        from sglang_1_1_module import generate_json
        result = generate_json(
            "Camera #3: 3.2mm crack on bearing. Output result.",
            INDUSTRIAL_SCHEMA
        )
        assert result["defect_type"] in ["crack", "scratch", "dent", "none"]
        assert isinstance(result["area_mm2"], (int, float))
        assert result["quality"] in ["qualified", "defective"]

    def test_traffic_scenario(self):
        from sglang_1_1_module import generate_json
        result = generate_json(
            "Intersection A: 5 vehicles, 45km/h. Output status.",
            TRAFFIC_SCHEMA
        )
        assert isinstance(result["vehicle_count"], int)
        assert isinstance(result["avg_speed_kmh"], (int, float))
        assert result["status"] in ["smooth", "slow", "congested", "blocked"]


# ================================================================
# Ollama Edition Tests
# ================================================================
class TestOllamaEdition:
    """Tests for sglang_1_1_ollama.py"""

    @pytest.fixture(autouse=True)
    def setup(self):
        import sys
        sys.path.insert(0, r"C:\Users\PC\Desktop\揭榜挂帅")

    def test_generate_json_returns_dict(self):
        from sglang_1_1_ollama import generate_json
        result = generate_json("Output student Zhangsan, age 20, CS", STUDENT_SCHEMA)
        assert isinstance(result, dict)
        assert "name" in result

    def test_generate_json_all_fields(self):
        from sglang_1_1_ollama import generate_json
        result = generate_json("Student Zhangsan, 20, Computer Science", STUDENT_SCHEMA)
        for field in STUDENT_SCHEMA["required"]:
            assert field in result, f"Missing: {field}"

    def test_generate_json_pydantic_ollama(self):
        from sglang_1_1_ollama import generate_json_pydantic
        result = generate_json_pydantic("Output Zhangsan, 20, CS", StudentModel)
        assert isinstance(result, dict)
        assert "name" in result
        StudentModel(**result)  # Pydantic validation

    def test_generate_label_returns_valid(self):
        from sglang_1_1_ollama import generate_label
        labels = ["qualified", "defective"]
        result = generate_label("Product has crack. Quality?", labels)
        assert result in labels

    def test_generate_value_industrial(self):
        from sglang_1_1_ollama import generate_value
        result = generate_value("Crack 12.5mm2 on bearing")
        assert "area_mm2" in result
        assert "quality" in result

    def test_generate_diff_format(self):
        from sglang_1_1_ollama import generate_diff
        result = generate_diff("rename x to y", old_code="x = 1")
        assert result.startswith("---")
        assert "+++" in result

    def test_retry_mechanism(self):
        from sglang_1_1_ollama import generate_json
        # Should succeed within retries
        result = generate_json("Output Zhangsan info", STUDENT_SCHEMA, retries=3)
        assert isinstance(result, dict)

    def test_ollama_memory_under_limit(self):
        import psutil
        mems = []
        for proc in psutil.process_iter(['name', 'memory_info']):
            if 'ollama' in proc.info['name'].lower():
                mems.append(proc.info['memory_info'].rss / 1e9)
        total = sum(mems)
        assert total <= 2.0, f"Ollama memory {total:.2f}GB exceeds 2GB limit"


# ================================================================
# Common tests (run against whichever edition is available)
# ================================================================
class TestSchemaValidation:
    """Tests that don't need model inference - pure logic"""

    def test_schema_has_required_field(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        assert "x" in schema["required"]

    def test_json_schema_is_valid(self):
        schema = STUDENT_SCHEMA
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_label_list_not_empty(self):
        labels = ["qualified", "defective"]
        assert len(labels) > 0
        assert all(isinstance(l, str) for l in labels)

    def test_diff_pattern_matches(self):
        import re
        diff = "--- a/test.py\n+++ b/test.py\n@@ -1,3 +1,4 @@\n unchanged\n-old\n+new"
        pattern = r"---[^\n]*\n\+\+\+[^\n]*\n@@[^\n]*@@(?:\n[ \-\+][^\n]*)*"
        assert re.search(pattern, diff, re.DOTALL) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
