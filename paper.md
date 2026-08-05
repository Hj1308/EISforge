---
title: "EISForge: An Integrated Open-Source Framework for Electrochemical
  Characterisation — EIS, CV, LSV, and ECSA Analysis with Physics-Informed
  Machine Learning"
tags:
  - Python
  - electrochemical impedance spectroscopy
  - cyclic voltammetry
  - linear sweep voltammetry
  - ECSA
  - equivalent circuit fitting
  - electrocatalysis
  - machine learning
  - Kramers-Kronig
authors:
  - name: Hoda Jafari
    orcid: 0000-0003-1478-3715
    affiliation: 1
  - name: Morteza Tabatabaeipour
    orcid: 0000-0002-0819-6592
    affiliation: 2
affiliations:
  - name: Faculty of Chemical Process Development,
           Chemistry and Chemical Engineering Research
           Center of Iran (CCERCI)
    index: 1
  - name: School of Engineering, Faculty of Computing,
           Engineering and the Built Environment,
           Ulster University, United Kingdom
    index: 2
date: 14 June 2026
bibliography: paper.bib
---

## Summary

EISForge is an open-source Python framework that provides a unified,
instrument-agnostic pipeline for electrochemical characterisation.
Researchers in electrocatalysis routinely acquire Electrochemical
Impedance Spectroscopy (EIS), cyclic voltammetry (CV), and linear sweep
voltammetry (LSV) data in the same experiment, yet no single open-source
tool handles all three coherently. EISForge addresses this by integrating
Complex Nonlinear Least Squares (CNLS) equivalent circuit fitting,
Kramers-Kronig validation, CV/LSV analysis, electrochemically active
surface area (ECSA) calculation via three independent methods, batch
statistical reproducibility for replicate measurements, and a
literature-guided knowledge layer providing parameter priors for 28
electrochemical systems, distilled from a survey of over 200 papers on
alcohol oxidation and carbon-supported catalysts.
The framework additionally provides the architecture of a
Physics-Informed Transformer model (EIS-GPT) for circuit classification
and parameter regression. EISForge supports Ivium (.idf), Gamry (.dta),
and generic CSV file formats, and exposes a Streamlit web interface for
interactive, no-code analysis.

## Statement of Need

EIS, CV, and LSV are the three pillars of electrochemical catalyst
characterisation, yet the open-source ecosystem treats them as separate
problems. `impedance.py` [@Murbach2020] provides CNLS fitting but no
CV, LSV, or ECSA support. `PyEIS` offers limited circuit definitions.
Commercial packages — ZView, Metrohm Nova, BioLogic EC-Lab — are
proprietary and inaccessible to many research groups, particularly in
low- and middle-income countries. No existing open-source tool combines
EIS fitting with Kramers-Kronig validation, CV/LSV analysis, ECSA
calculation, statistical reproducibility for replicate measurements,
a literature knowledge base, and a machine learning pathway in a single
framework.

EISForge fills this gap. It is designed for researchers studying
electrocatalysts in alcohol oxidation reactions (AOR), hydrogen evolution
(HER), oxygen reduction (ORR), and water splitting — who routinely
acquire multi-technique data from different instruments and need
reproducible, publication-quality analysis without commercial software.
The software was developed and validated on a real Ivium EIS spectrum
shipped with the repository (tests/data/sample_eis.idf). Fitting an
R-p(R,CPE) circuit yields R0 = 51.6 Ω, R1 = 6.55 × 10⁴ Ω,
CPE1_0 = 1.35 × 10⁻⁵, CPE1_1 = 0.896, with a reduced χ² of 0.00067
under ordinary least squares and 0.00076 with Huber weighting. The fit
converges to identical parameters from widely separated initial guesses.

## Functionality

### EIS Analysis

EISForge implements CNLS fitting via `scipy.optimize.least_squares`
with Levenberg-Marquardt and Trust Region Reflective strategies.
Supported circuit elements include resistors (R), capacitors (C),
constant phase elements (CPE), and Warburg diffusion (W), composable
into arbitrary series and parallel combinations using a string notation
(e.g. `"R0-p(R1,CPE1)-W"`). Every fit is accompanied by a
Kramers-Kronig consistency check using the linear K-K method
[@Schonleber2014], with a Voigt-circuit fallback for spectra with
fewer than 10 frequency points. On the real Ivium spectrum shipped with the repository
(tests/data/sample_eis.idf), the R-p(R,CPE) fit reaches a reduced
χ² of 0.00067 under ordinary least squares and 0.00076 with Huber
weighting.

### CV and LSV Analysis

The CV module extracts onset potential (E_onset), forward/backward peak
current ratio (I_f/I_b), and geometric and ECSA-normalised current
densities — key metrics for assessing alcohol oxidation and ORR
activity. The LSV module computes Tafel slope, overpotential at
10 mA cm⁻², and mass activity, following benchmarking protocols
established for oxygen evolution electrocatalysts [@McCrory2013].
Both modules produce publication-quality figures with automatic
annotation of key parameters.

### ECSA Calculation

Three ECSA methods are implemented to cover the full range of
catalyst chemistries: (A) H-UPD integration for Pt and Pd catalysts
(Q_ref = 210 µC cm⁻²) [@Pozio2002]; (B) CO stripping for PtRu and
PtSn alloys, where H-UPD is not applicable due to overlapping
Ru/Sn oxide features; and (C) double-layer capacitance (C_dl) for
carbon-based, nitrogen-doped, and metal-free catalysts [@McCrory2013].
All three methods report specific ECSA (cm² mg⁻¹) and include
diagnostic warnings when integration windows contain Faradaic features.

### Statistical Reproducibility

The `BatchAnalyzer` module processes n ≥ 3 replicate measurements and
reports mean ± standard deviation for all fitted EIS parameters,
CV metrics, and LSV metrics. Relative standard deviation (RSD%) flags
are raised automatically when RSD > 10%, prompting the user to inspect
for outlier spectra. This functionality directly addresses the
reproducibility crisis in electrocatalysis benchmarking [@Morales2021].

### Literature Knowledge Base

A curated knowledge layer covering 28 electrochemical systems provides
literature-guided initial parameter estimates for CNLS fitting,
indexed by catalytic system (AOR, HER, ORR), catalyst composition,
and electrolyte. This substantially reduces convergence failures
compared to arbitrary initial guesses, particularly for multi-element
circuits with correlated parameters.

### EIS-GPT — Physics-Informed Transformer

EISForge includes the architecture of a Physics-Informed Transformer
in which each frequency point is treated as a single token, enabling
the model to learn the global spectral shape rather than point-wise
features. A novel physics-informed loss function combines spectral
reconstruction loss with three regularisation terms: Kramers-Kronig
consistency, passivity (Z_real > 0), and high-frequency limit
enforcement (Z → R_s as f → ∞) [@Vaswani2017]. The architecture
is fully implemented and tested; training on synthetic spectra
is planned for v0.4.

## Acknowledgements

The authors acknowledge the use of AI-based coding and language
tools during the software development process. All code was
reviewed, validated, and tested by the authors.

## References
