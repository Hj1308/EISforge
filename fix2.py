import os

f = "app.py"
t = open(f, "r", encoding="utf-8", errors="replace").read()
m = t

fixes = [
    ("Metal-Free (B₄C, N-doped C, CNT)", "Carbon Material (N-doped C, CNT, Graphene)"),
    ("Metal-Free (B₄C, N-doped C, CNT)", "Carbon Material (N-doped C, CNT, Graphene)"),
    ("Metal-Free", "Carbon Material"),
]

for old, new in fixes:
    m = m.replace(old, new)

if m != t:
    open(f, "w", encoding="utf-8").write(m)
    print("FIXED: app.py")
else:
    print("No changes found")
print("Done!")
