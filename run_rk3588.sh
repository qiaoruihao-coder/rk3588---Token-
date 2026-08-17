#!/usr/bin/env bash
# ============================================================
# RK3588 一键启动脚本
# ============================================================
# 用法: bash run_rk3588.sh [rknn模型路径]
#
# 前置条件:
#   1. RK3588 已刷 Linux (Armbian / Ubuntu ARM)
#   2. pip install psutil pydantic rknn-toolkit-lite2
#   3. .rknn 模型已转换并放在本目录
#
# 默认模型: qwen2.5-1.5b.rknn
# ============================================================
set -e

MODEL="${1:-qwen2.5-1.5b.rknn}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  RK3588 动态控制 — 一键启动"
echo "  模型: $MODEL"
echo "========================================"

# 检查模型文件
if [ ! -f "$MODEL" ]; then
    echo ""
    echo "  [ERROR] 模型文件不存在: $MODEL"
    echo "  请先运行模型转换，或指定路径:"
    echo "    bash run_rk3588.sh /path/to/model.rknn"
    echo ""
    exit 1
fi

# 检查 NPU 驱动
if ! lsmod | grep -q rknpu 2>/dev/null; then
    echo "  [WARN] rknpu 驱动未加载，尝试加载..."
    sudo modprobe rknpu 2>/dev/null || echo "  请手动加载: sudo modprobe rknpu"
fi

# 启动演示
echo ""
echo "  [1] 验证系统指标采集..."
python3 -c "
from dynamic_control.monitor import Monitor
m = Monitor()
s = m.snapshot()
print(f'  NPU temp: {s.npu_temp_c}°C  |  CPU temp: {s.cpu_temp_c}°C')
print(f'  NPU freq: {s.npu_freq_mhz}MHz  |  Memory: {s.mem_avail_gb:.2f}GB free')
"

echo ""
echo "  [2] 初始化动态控制器..."
python3 -c "
from dynamic_control import DynamicController
ctrl = DynamicController('$MODEL')
ok = ctrl.init()
if ok:
    print('  控制器就绪，后端: ' + ctrl.health()['backend'])
else:
    print('  [WARN] 无可用后端，将拒绝所有推理请求')
"

echo ""
echo "  [3] 运行测试推理..."
python3 -c "
from dynamic_control import DynamicController
import json

ctrl = DynamicController('$MODEL')
ctrl.init()

schema = {
    'type': 'object',
    'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'major': {'type': 'string'}},
    'required': ['name', 'age', 'major']
}

result = ctrl.generate_json('输出学生张三, 20岁, 计算机专业', schema)
print(f'  结果: {json.dumps(result, ensure_ascii=False)}')
print(f'  状态: {ctrl.health()[\"level\"]} | 后端: {ctrl.health()[\"backend\"]}')
"

echo ""
echo "  [4] 运行演示脚本..."
python3 demo_dynamic_control.py

echo ""
echo "========================================"
echo "  启动完成。"
echo "  调度员接入示例:"
echo ""
echo "  from dynamic_control import DynamicController"
echo "  ctrl = DynamicController('$MODEL')"
echo "  ctrl.init()"
echo "  result = ctrl.generate_json(prompt, schema)"
echo "========================================"
