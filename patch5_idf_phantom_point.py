# patch5_idf_phantom_point.py
# Fix: _read_data_block reads 5 extra lines past n_points, swallowing a row
# from the NEXT data object in multi-object Ivium primary_data sections.
# This injects a phantom point (e.g. f=0.921 Hz, Z'=1.0) that creates the
# criss-cross "loop to origin" artifact in the Nyquist plot.
# Fix: stop as soon as n_points valid rows have been collected.
import shutil, sys

PATH = r"eisforge/parsers/autolab_parser.py"

OLD = """        for line in lines[data_start: data_start + max_lines + 5]:
            stripped = line.strip()
            if not stripped:
                continue"""

NEW = """        for line in lines[data_start: data_start + max_lines + 5]:
            if n_points and len(rows) >= n_points:
                break  # multi-object IDF: never read into the next data object
            stripped = line.strip()
            if not stripped:
                continue"""

s = open(PATH, encoding="utf-8").read()
if NEW in s:
    print("Already patched. Nothing to do.")
    sys.exit(0)
if OLD not in s:
    print("ERROR: OLD block not found — file differs. Aborting, no changes made.")
    sys.exit(1)
shutil.copy(PATH, PATH + ".bak")
open(PATH, "w", encoding="utf-8").write(s.replace(OLD, NEW, 1))
print("Patched OK:", PATH, "(backup:", PATH + ".bak)")
