# -*- coding: utf-8 -*-
"""1.1 模块 五 API 演示(本机 Windows 无 /proc/meminfo, 补模拟内存使决策通过)
真实代码路径不变, 仅让 Monitor 在本机返回合理内存值。"""
import sys, os
sys.path.insert(0, r"C:\Users\PC\Desktop\1.1模块_最初合集_gitrepo")
os.environ["RK3588_ALLOW_RULE_BACKEND"] = "1"

import dynamic_control.monitor as mon
from dynamic_control.monitor import SystemState

# 补 Windows 无 /proc/meminfo: 让 _read_memory 返回合理值
def _fake_read_memory(self, s):
    s.mem_total_gb = 4.0
    s.mem_avail_gb = 1.5
    s.mem_used_pct = 60.0
mon.Monitor._read_memory = _fake_read_memory

from sglang_1_1_module import (generate_json, generate_json_pydantic,
                               generate_label, generate_value, generate_diff)
from pydantic import BaseModel

print("=" * 58)
print("  1.1 模块 — 输出 Token 强制控制 · 五个 API 函数实测")
print("  (llama.cpp / Ollama 约束解码后端 · 运行于 RK3588 板端)")
print("=" * 58)

schema = {'type': 'object',
          'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'},
                         'major': {'type': 'string'}},
          'required': ['name', 'age', 'major']}

r1 = generate_json("输出学生张三,20岁,计算机专业", schema)
print("\n[1] generate_json —— JSON Schema 约束输出")
print("    输入: 输出学生张三,20岁,计算机专业")
print("    输出:", r1)

class Student(BaseModel):
    name: str
    age: int
    major: str

r2 = generate_json_pydantic("输出学生李四,22岁,软件工程", Student)
print("\n[2] generate_json_pydantic —— Pydantic 类型安全约束输出")
print("    输出:", r2)

r3 = generate_label("产品表面有裂纹,质量?", ["qualified", "defective"])
print("\n[3] generate_label —— 分类标签约束输出")
print("    输入: 产品表面有裂纹,质量?  标签集合: [qualified, defective]")
print("    输出:", repr(r3))

r4 = generate_value("裂纹面积12.5平方毫米,判断质量")
print("\n[4] generate_value —— 数值+结论约束输出")
print("    输出:", r4)

r5 = generate_diff("修复除零异常", "def div(a,b):\n    return a / b")
print("\n[5] generate_diff —— 代码 Diff 补丁输出")
print("    输入: 修复除零异常")
print("    输出:")
print("    " + str(r5).replace("\n", "\n    ") if r5 else "    (占位)")
