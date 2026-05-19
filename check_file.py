filepath = r"C:\Users\hoda\Desktop\نام_فایل_EIS.idf"

with open(filepath, "r", encoding="latin-1", errors="replace") as f:
    lines = f.read().splitlines()

# Method type
for line in lines[:50]:
    if "method" in line.lower():
        print(f"Method: {line}")
        break

# Find first numeric data line
print("\nFirst 10 data lines:")
count = 0
for i, line in enumerate(lines):
    parts = line.strip().split()
    if len(parts) >= 3:
        try:
            nums = [float(p) for p in parts[:5]]
            print(f"Line {i+1}: {nums}")
            count += 1
            if count >= 10:
                break
        except:
            continue