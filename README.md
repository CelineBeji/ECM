# ECM

The Expectation-Causality-Maximization (ECM) is a algorithm that estimates the parameters of causal populations, from which the Individual Treatment Effect can be inferred. A more description is provided in the paper [Estimating Individual Treatment Effects through Causal Populations Identification](https://pages.github.com/).

This repository contains the code source use to comprae ECM with baselin methods on synthetic and real-world datasets.

## Requirements

- This project was designed for use with Python 3.7. 
- To run models you will need to install `sklearn`, `numpy`.
- to run results you will need to install `pandas`, `os`.

## Datasets

- Synthetical datasets: Two dimensional features vector distributed as a mixture of four Gaussian distributions.
- IHDP: We also use the Infant Health and Development Program (IHDP) semi-synthetic dataset compiled by J.Hill 2011 for causal effect estimation.

## Run all experiments

To reproduce all experiments of the paper just type: `python run_all_experiments.py`


