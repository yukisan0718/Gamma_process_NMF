Gamma process NMF (GaP-NMF)
====

## Overview
Implementation of a Bayesian inference algorithm for music, namely the gamma process-NMF (GaP-NMF).

The "Gamma_process_NMF.py" has been implemented to estimate the number of basements decomposed from music. This algorithm, using a Bayesian inference (non-parametric) approach, was proposed by M. D. Hoffman, et.al. [1]. Most of multiplicative update rules for NMF [2] need to be set the number of basements in advance, however, the GaP-NMF can estimate the appropriate number of basements deterministically.

## Requirement
soundfile 0.10.3

matplotlib 3.1.0

numpy 1.18.1

scipy 1.4.1

scikit-learn 0.21.2

museval 0.3.0 (only for evaluation metrics)

## Dataset preparation
You can apply this algotithm to any audio signal you want. An example of a short piano play has been prepared for the demonstration.

## References
[1] M. D. Hoffman, D. M. Blei, and P. R. Cook: 'Bayesian Nonparametric Matrix Factorization for Recorded Music', In Proceedings of the International Conference on Machine Learning (ICML), pp.641–648, (2010)

[2] D. D. Lee and H. S. Seung: 'Algorithms for Non-negative Matrix Factorization', The Neural Information Processing Systems (NIPS), Vol.13, pp556–562, (2001)