# EISforge — Example Data

This folder contains example files to help you get started with EISforge.

## Synthetic EIS Data (CSV)

The file `synthetic_randles.csv` contains a simulated Randles circuit EIS spectrum:

- **Circuit model**: Rs + Rct//(Cdl)  
- **Parameters**: Rs = 10 Ω, Rct = 150 Ω, Cdl = 50 µF  
- **Frequency range**: 100 kHz → 10 mHz (60 logarithmically spaced points)  

### Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("examples/synthetic_randles.csv")

# Nyquist plot
plt.figure(figsize=(6, 5))
plt.plot(df["Zreal"], -df["Zimag"], "o-", markersize=4)
plt.xlabel("Z' / Ω")
plt.ylabel("-Z'' / Ω")
plt.title("Nyquist Plot — Synthetic Randles Circuit")
plt.axis("equal")
plt.tight_layout()
plt.show()
```

## Supported File Formats

| Format | Extension | Parser |
|--------|-----------|--------|
| Generic CSV | `.csv` | `eisforge.parsers.csv_parser` |
| Gamry EXPLAIN | `.DTA` | `eisforge.parsers.gamry_parser` |
| Autolab NOVA | `.csv` (Autolab) | `eisforge.parsers.autolab_parser` |
| BioLogic EC-Lab | `.mpr`, `.mpt` | `eisforge.parsers.biologic_parser` |

## Planned (Roadmap)

- Zahner Thales `.ism` parser (v0.3)
- Batch processing of multiple files
