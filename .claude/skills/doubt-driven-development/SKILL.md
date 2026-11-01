---
name: doubt-driven-development
description: >
  Cross-examine every non-trivial EISForge decision with a fresh-context,
  adversarial reviewer before it stands. Use before writing a patch that
  changes detection or fitting logic (onset detection, Tafel domain selection,
  peak finding, circuit fitting, K-K, DRT, model ranking), before claiming a
  number is "correct", and whenever a sign convention, unit, or potential
  scale is involved.
---

# Doubt-Driven Development (EISForge edition)

Adapted from `doubt-driven-development` in
[addyosmani/agent-skills](https://github.com/addyosmani/agent-skills)
(MIT, © 2025 Addy Osmani; see `../THIRD_PARTY_NOTICES.md`). Shortened and
rewritten for electrochemical analysis code; the external-CLI cross-model step
was dropped and replaced by "ask the human".

A confident answer is not a correct one. Long sessions turn assumptions into
"facts" (anodic = positive, potential is vs RHE, current is in A not mA). This
skill makes a reviewer whose job is to **disprove** the decision look at it
while changing course is still cheap.

It complements, and never replaces, the MANDATORY WORKFLOW in `AGENTS.md`:
real-data verification still comes first. Doubt checks the reasoning; real
data checks the numbers.

## When to use

A decision is non-trivial when at least one is true:

- It adds or changes branching logic in detection or fitting code.
- It depends on a sign convention, unit, reference electrode, or scan direction.
- It changes a value a user will publish (E_onset, Tafel slope, Rct, ECSA, ...).
- It changes an optimizer setting, bound, weighting, or model-ranking rule.
- It touches a parser for an instrument format (`.idf`, `.dta`, `.mpt`, `.mpr`).

Do not use it for renames, formatting, docs typos, or running tests.

## The cycle

```
- [ ] 1 CLAIM     write the claim and why it matters (2-3 lines)
- [ ] 2 EXTRACT   artifact + contract only, reasoning stripped
- [ ] 3 DOUBT     fresh-context reviewer with the adversarial prompt
- [ ] 4 RECONCILE classify every finding against the artifact text
- [ ] 5 STOP      trivial findings, 3 cycles, or the human says go
```

### 1. CLAIM

```
CLAIM: The hybrid onset detector returns the same E_onset for a forward
       scan stored with negative anodic current as for its sign-flipped copy.
WHY IT MATTERS: Ivium files can store anodic current as negative; a wrong
       E_onset is a wrong number in a paper.
```

If you cannot write it this compactly, you have a hunch, not a decision.

### 2. EXTRACT

Give the reviewer the **artifact** and the **contract**, not the journey.

- Artifact: the function or the diff, not the whole module.
- Contract: what it must satisfy: input conventions (sign, units, reference
  scale), expected behaviour on edge cases, and any known-good reference
  behaviour, stated **qualitatively** (see "Private data" below).
- Strip your reasoning, keep the evidence. If a caller, guard, or test
  contradicts the claim, include it.

### 3. DOUBT

Run a fresh-context subagent (in Claude Code: the Agent tool) with this prompt,
verbatim, followed by the artifact and the contract:

```
Adversarial review. Find what is wrong with this artifact.
Assume the author is overconfident. Look for:
- Unstated assumptions about sign convention, units, reference electrode,
  scan direction, or cycle selection
- Edge cases: flat or noisy current, no peak, reversed arrays, NaN/inf,
  single-point segments, metal-free catalysts with no metal-like peak
- Numerical issues: scale-dependent tolerances, ill-conditioning,
  bounds the optimizer can hit, wrong objective in model ranking
- Ways the contract could be violated
- Existing conventions in this codebase it might break
Do NOT validate. Do NOT summarize. Find issues, or state explicitly that
you cannot find any after thorough examination.

ARTIFACT: <paste>
CONTRACT: <paste>
```

Never pass the CLAIM; it biases the reviewer toward agreement.

A second opinion from a different model can catch blind spots the first one
shares with you. Do not run any external CLI yourself; offer it to the human
in one line ("want a second opinion from another model?") and let them decide.

### 4. RECONCILE

The reviewer's output is data, not a verdict. Re-read the artifact for each
finding and classify it, first match wins:

1. **Contract misread**: the contract was unclear. Fix the contract, re-run.
2. **Valid and actionable**: change the artifact, then re-verify on real data.
3. **Valid trade-off**: real, but not worth fixing now. Write it down in the
   patch's `PATCHES.md` entry as a known limitation.
4. **Noise**: correct under context the reviewer lacked. Ask whether that
   context belongs in the contract.

A finding that says "this would give a different number on real data" is
settled only by running the real data, not by argument.

### 5. STOP

Stop when the next cycle finds only trivial or already-considered issues,
after 3 cycles, or when the human says go. Three cycles with substantive
findings means the artifact is not ready: tell the human; do not loop a
fourth time. If the artifact is too large to review in 3 cycles, split it.

## Private data

The author's measurement files are unpublished. Contracts and reviewer
prompts carry only qualitative facts ("E_onset must not move when the current
sign is flipped", "Rs from the fit should agree with the high-frequency
intercept"), never real numbers from those files. Synthetic data with a known
answer may be pasted freely.

## Red flags

- Running a reviewer for a one-line rename.
- Treating the reviewer's output as authoritative without re-reading the code.
- Asking "is this good?" instead of "find issues".
- Passing the CLAIM, or stripping the contract.
- Two or more cycles with substantive findings and zero classified actionable:
  that is validation, not doubt. Stop and tell the human.
- Skipping doubt because the patch "obviously works" on synthetic data.
