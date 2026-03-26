# Getting Started

Core workflows: data preparation, running PMF, choosing factors, and a complete end-to-end analysis.

## Basic PMF

The simplest possible PMF workflow: generate synthetic data with known sources, run ACLS factorization, and inspect how well the solver recovers the truth. Learn the basics of the Q statistic, explained variance, and factor matching.

```{marimo} 01_basic_example.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Basic PMF Example
```

## Data Preparation

Real environmental data is messy: missing values, values below detection limits, outliers, and mixed measurement methods. This notebook walks through cleaning, quality control, and uncertainty estimation before running PMF.

```{marimo} 03_data_preparation.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Data Preparation
```

## Factor Selection

Choosing the number of factors is critical and non-obvious. This notebook compares three methods: Q/Qexp curves, RMSE inflection, and the Diagnostic Discrepancy Principle. Learn to interpret disagreement between metrics.

```{marimo} 02_factor_selection.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch Factor Selection
```

## Complete Workflow

End-to-end source apportionment: load data, assess quality, estimate uncertainty, choose factor count, run ACLS, perform DISP and FPEAK analysis, and interpret profiles. A real example with all diagnostics.

```{marimo} 04_complete_workflow.py
:height: 900px
:click-to-load: overlay
:load-button-text: Launch Complete Workflow
```

## DISP Analysis

Factor profiles can be locally uncertain even when the overall fit is good. Displacement testing (DISP) probes the Q-surface by pinning each profile element and measuring how far it can move before fit degrades. Learn to assess rotational stability and factor uniqueness.

```{marimo} 12_disp_demo.py
:height: 800px
:click-to-load: overlay
:load-button-text: Launch DISP Demo
```
