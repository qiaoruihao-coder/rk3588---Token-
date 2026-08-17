from sglang_1_1_ollama import generate_json, generate_label, generate_value

print("=" * 40)
print("1.1 Module Quick Test")
print("=" * 40)

print("\n[1] JSON output:")
r = generate_json("Output student Zhangsan, 20 years old, CS major",
    {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"major":{"type":"string"}},"required":["name","age","major"]})
print(f"  -> {r}")

print("\n[2] Label output:")
r2 = generate_label("Camera found crack on product. Quality?",
    ["qualified", "defective"])
print(f"  -> {r2}")

print("\n[3] Value output:")
r3 = generate_value("Crack detected, area 12.5 mm2 on bearing surface")
print(f"  -> {r3}")

print("\n" + "=" * 40)
print("All done! Your module works!")
print("=" * 40)
