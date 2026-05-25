import os

R = [
    ("metal_free", "carbon_material"),
    ("B4C, N-doped Carbon, CNT, rGO", "N-doped Carbon, CNT, rGO, graphene-based"),
    ("B4C, N-doped C, CNT", "N-doped C, CNT, graphene-based"),
    ("B4C-specific", "carbon_material-specific"),
    ("B4C", "carbon_material"),
]

F = [
    "eisforge/analysis/cv_analyzer.py",
    "eisforge/analysis/lsv_analyzer.py",
    "eisforge/analysis/eis_cv_correlator.py",
    "eisforge/analysis/batch_analyzer.py",
    "app.py",
    "app_simple.py",
]

for fp in F:
    if not os.path.exists(fp):
        print("SKIP: " + fp)
        continue
    t = open(fp, "r", encoding="utf-8", errors="replace").read()
    m = t
    for o, n in R:
        m = m.replace(o, n)
    if m != t:
        open(fp, "w", encoding="utf-8").write(m)
        print("FIXED: " + fp)
    else:
        print("OK: " + fp)

print("Done!")