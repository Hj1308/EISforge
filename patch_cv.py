import re

filepath = "eisforge/analysis/cv_analyzer.py"

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    code = f.read()

old_import = "import numpy as np"
new_import = "import numpy as np\nfrom eisforge.analysis.ecsa_calculator import AutoECSA"

if "ecsa_calculator" not in code:
    code = code.replace(old_import, new_import, 1)
    print("Import added")
else:
    print("Import already exists")

old_block = "# Current densities"
new_block = """# Auto-ECSA (if not manually set)
        if self.ecsa <= 0:
            try:
                ecsa_calc = AutoECSA(
                    catalyst_type=self.catalyst_type,
                    catalyst_loading_mg=self.catalyst_loading if self.catalyst_loading > 0 else None
                )
                ecsa_result = ecsa_calc.calculate(potential, current_ma, self.scan_rate)
                self.ecsa = ecsa_result.ecsa_cm2
                logger.info(f"Auto-ECSA: {self.ecsa:.4f} cm2 via {ecsa_result.method}")
            except Exception as ex:
                logger.warning(f"Auto-ECSA failed: {ex}")

        # Current densities"""

if "Auto-ECSA" not in code:
    code = code.replace(old_block, new_block, 1)
    print("Auto-ECSA block added")
else:
    print("Auto-ECSA block already exists")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("Done!")
