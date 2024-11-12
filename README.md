# pyRankMCDA

## Introduction

**pyRankMCDA** is a Python library designed for rank aggregation in multi-criteria decision analysis (MCDA). It provides implementations of classical and modern rank aggregation methods, allowing users to combine multiple rankings into a single consensus ranking. This is particularly useful in fields like decision science, information retrieval, and any domain where synthesizing different ordered lists is necessary.

## Features

- **Multiple Rank Aggregation Methods**:
  - Borda Method
  - Copeland Method
  - Footrule Rank Aggregation
  - Fast Footrule Rank Aggregation
  - Kemeny-Young Method
  - Fast Kemeny-Young
  - Median Rank Aggregation
  - PageRank Algorithm
  - Plackett-Luce Model Aggregation
  - Reciprocal Rank Fusion
  - Schulze Method

- **Distance and Correlation Metrics**:
  - Cayley Distance 
  - Footrule Distance
  - Kendall Tau Distance
  - Kendall Tau Correlation
  - Spearman Rank Correlation

- **Visualization Tools**:
  - Heatmaps of rankings
  - Radar charts comparing rankings from different methods
  - Multidimensional Scaling (MDS) plots for visualizing distances between ranking methods

## Usage
1. Install

```bash
pip install pyRankMCDA
```

2. Basic Example

```python
import numpy as np
from pyRankMCDA import rank_aggregation

# Example rankings from different methods

ranks = np.array([
    [1, 2, 3],
    [2, 1, 3],
    [3, 2, 1]
])

# Initialize rank aggregation object
ra = rank_aggregation(ranks)

# Run Borda method
borda_rank = ra.borda_method(verbose = True)
```

3. Running Multiple Methods

```python
# Define the methods to run
methods = ['bd', 'cp', 'fky', 'md', 'pg']

# Run selected methods
df = ra.run_methods(methods)
```

4. Visualization

```python
# Plot heatmap of rankings
ra.plot_ranks_heatmap(df)

# Plot radar chart of rankings
ra.plot_ranks_radar(df)
```

5. Computing Metrics

```python
# Calculate distance and correlation metrics
d_matrix = ra.metrics(df)

# Plot metric comparisons
ra.metrics_plot(d_matrix)
```

6. Try it in **Colab**:

- Example: ([ Colab Demo ](https://colab.research.google.com/drive/1qtS4kRMN_NdG0yer8UcN196bWnTM-dlI?usp=sharing))


7. Others
- [3MOAHP](https://github.com/Valdecy/Method_3MOAHP) - Inconsistency Reduction Technique for AHP and Fuzzy-AHP Methods
- [pyDecision](https://github.com/Valdecy/pyDecision) - A library for many MCDA methods
- [pyMissingAHP](https://github.com/Valdecy/pyMissingAHP) - A Method to Infer AHP Missing Pairwise Comparisons
- [ELECTRE-Tree](https://github.com/Valdecy/ELECTRE-Tree) - Algorithm to infer the ELECTRE Tri-B method parameters
- [Ranking-Trees](https://github.com/Valdecy/Ranking-Trees) - Algorithm to infer the ELECTRE II, III, IV and PROMETHEE I, II, III, IV method parameters
- [EC-PROMETHEE](https://github.com/Valdecy/ec_promethee) -  A Committee Approach for Outranking Problems
- [MCDM Scheduler](https://github.com/Valdecy/mcdm_scheduler) -  A MCDM approach for Scheduling Problems
