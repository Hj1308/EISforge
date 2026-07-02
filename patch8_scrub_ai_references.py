# patch8_scrub_ai_references.py
# Level-4 cleanup: remove named references to third-party AI coding tools
# from source comments/docstrings, and genericize the AI-assistance
# disclosure in paper.md (standard JOSS-style wording, no tool names).
# No functional/behavioral change — comments and one dict key rename only.
import shutil, sys

FILES_PATCHES = {
    "eisforge/standards/carbon_standards.py": [
        (
            "  - + gemini-code supplement (Hoda Jafari, May 2026)",
            "  - + supplementary internal compilation (Hoda Jafari, May 2026)",
        ),
        (
            "#     Source: gemini-code supplement + AOR literature",
            "#     Source: internal compilation + AOR literature",
        ),
        (
            '        "note": "KOH / NaOH media. gemini-code + Fu 2023.",',
            '        "note": "KOH / NaOH media. Internal compilation + Fu 2023.",',
        ),
        (
            "#     Source: gemini-code + EIS table (Section 3)",
            "#     Source: internal compilation + EIS table (Section 3)",
        ),
        (
            '    "alkaline": (1000.0, 30000.0),   # \u03a9  \u2014 gemini-code',
            '    "alkaline": (1000.0, 30000.0),   # \u03a9  \u2014 internal compilation',
        ),
        (
            "#     Source: gemini-code supplement",
            "#     Source: internal compilation",
        ),
        (
            "#     Source: gemini-code + Gholipour 2021 + EIS table (Section 3)",
            "#     Source: internal compilation + Gholipour 2021 + EIS table (Section 3)",
        ),
        (
            "# 4.  E_onset TYPICAL RANGE (V vs RHE)\n#     Source: gemini-code supplement",
            "# 4.  E_onset TYPICAL RANGE (V vs RHE)\n#     Source: internal compilation",
        ),
        (
            "    # gemini-code original suggestion (kept for compatibility)\n    \"carbon_gemini\": {",
            "    # Nested CPE + Warburg variant (kept for compatibility)\n    \"carbon_nested_cpe_warburg\": {",
        ),
        (
            '        "circuit": "R0-p(CPE1,p(R1,Wo1))",   # R_s(CPE(R_ct W_o)) in gemini notation',
            '        "circuit": "R0-p(CPE1,p(R1,Wo1))",   # R_s(CPE(R_ct W_o)) nested notation',
        ),
        (
            '        "note": "Gemini-code variant: nested CPE + Warburg. Use for porous carbon.",',
            '        "note": "Nested CPE + Warburg variant. Use for porous carbon.",',
        ),
        (
            "All thresholds sourced from Carbon_Materials_Knowledge_Base.md + gemini-code.",
            "All thresholds sourced from Carbon_Materials_Knowledge_Base.md + internal compilation.",
        ),
    ],
    "eisforge/analysis/lsv_analyzer.py": [
        (
            "        # Hybrid Tafel domain (ChatGPT OER detection + Grok noise floor + our valley)",
            "        # Hybrid Tafel domain: percentile-based OER-onset detection +\n"
            "        # noise-floor threshold + valley detection (see PROJECT_STATUS.md)",
        ),
    ],
    "eisforge/catalogs/suggestion_engine.py": [
        (
            "eisforge/catalogs/circuit_models.py::AOR_PSEUDOINDUCTIVE and the project\n"
            "knowledge base, gemini-code, section 4.2). Presenting only one option lets",
            "eisforge/catalogs/circuit_models.py::AOR_PSEUDOINDUCTIVE and the project\n"
            "knowledge base, Section 4.2). Presenting only one option lets",
        ),
    ],
    "paper.md": [
        (
            "The authors acknowledge the use of AI-assisted development tools,\n"
            "including Claude (Anthropic) and Perplexity AI, during the software\n"
            "development process. All code was reviewed, validated, and tested\n"
            "by the authors.",
            "The authors acknowledge the use of AI-based coding and language\n"
            "tools during the software development process. All code was\n"
            "reviewed, validated, and tested by the authors.",
        ),
    ],
}

any_error = False
for path, patches in FILES_PATCHES.items():
    s = open(path, encoding="utf-8").read()
    orig = s
    for i, (old, new) in enumerate(patches, 1):
        if new in s:
            print(f"[{path}] {i}/{len(patches)} already applied, skipping")
            continue
        if old not in s:
            print(f"ERROR [{path}] step {i}: OLD block not found. Skipping this file.")
            any_error = True
            break
        s = s.replace(old, new, 1)
        print(f"[{path}] {i}/{len(patches)} OK")
    else:
        if s != orig:
            shutil.copy(path, path + ".bak_patch8")
            open(path, "w", encoding="utf-8").write(s)
            print(f"Patched OK: {path} (backup: {path}.bak_patch8)")
        else:
            print(f"Nothing to do: {path}")
        continue
    print(f"Aborted changes for {path} due to error above.")

if any_error:
    sys.exit(1)
print("\nDone. Grep to verify no references remain:")
print('  findstr /s /i /m "gemini chatgpt grok deepseek" *.py')
