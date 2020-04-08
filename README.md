# ECM

The Expectation-Causality-Maximization (ECM) is a algorithm that estimates the parameters of causal populations, from which the Individual Treatment Effect (ITE) can be inferred. A more description is provided in the paper [Estimating Individual Treatment Effects through Causal Populations Identification].

This repository contains the code source use to comprae ECM with baselin methods on synthetic and real-world datasets.

## Requirements

- This project was designed for use with Python 3.7. 
- To run models you will need to install `sklearn`, `numpy`.
- to run results you will need to install `pandas`, `os`.

## Datasets

- 
- IHDP: We also use the IHDP semi-synthetic dataset compiled by J.Hill for causal effect estimation described on her paper [Bayesian Nonparametric Modeling for Causal Inference](https://www.researchgate.net/profile/Jennifer_Hill3/publication/236588890_Bayesian_Nonparametric_Modeling_for_Causal_Inference/links/0deec5187f94192f12000000.pdf).

## Run all experiments

To reproduce all experiments of the paper just type: `python all_experiments.py`


