# tests/data — real instrument files

This directory contains **real experimental files** used to test the
instrument parsers end-to-end. These files are not generated
programmatically; they are exported from the physical instrument and
provided by the project author.

## sample_eis.idf

- Format: Ivium `.idf` (raw instrument export).
- Consumed by `tests/test_parsers.py::TestIviumIDF` via `IviumIDFParser`.
- The tests assert structural properties only: expected columns, a
  non-empty result, monotonic (all ascending or all descending)
  frequencies, and no NaNs in the frequency/impedance columns.
- The whole test class is guarded by
  `pytest.mark.skipif(not DATA.exists())`, so the suite passes on machines
  without the file (e.g. CI).
- Electrolyte: HClO4.
- Provenance: measured on a material studied for desulfurization. It is
  unrelated to any published or in-review work by the author, and is
  released here solely as test and example data.
- The timestamp embedded in the file is not reliable — the instrument
  clock was not set, so the recorded date does not reflect when the
  measurement was made. The file carries a January 2005 date, which reads
  as a vendor demo file rather than real data; that date is an artifact
  and should be ignored.
- **Kramers-Kronig:** this spectrum FAILS the built-in K-K validation
  (max residual ~43.8% via the Voigt-circuit fallback). This is expected:
  the spectrum has pseudo-inductive character in the high-frequency tail,
  and inductive loops violate the K-K causality assumptions. The file is
  still a valid parser and CNLS fitting test case — a failed K-K check is
  an informative scientific result, not a sign of a broken tool. If you
  see a red K-K warning in the app for this file, that is the expected
  behaviour.

## What is intentionally not stored here

Electrode area, catalyst loading, reactant concentration, and any other
experimental condition are part of the author's experimental record. They
are not derived, assumed, or committed in this repository — do not look
for (or add) such values here, because they do not exist in this file.
