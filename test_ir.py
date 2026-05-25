from eisforge.analysis.cv_analyzer import CVAnalyzer
import numpy as np

pot = np.array([-0.2, 0.0, 0.5, 1.0, 0.5, 0.0, -0.2])
cur = np.array([-0.1, 0.0, 0.5, 3.5, 2.5, 1.0, -0.1])

ana = CVAnalyzer(scan_rate=50, electrode_area=1.0)
result = ana.analyze(pot, cur, r_s_ohms=25.23)
print(result.ir_compensated)
print(f"E_onset = {result.e_onset:.4f} V")