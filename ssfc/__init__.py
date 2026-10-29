"""SSFC: Standardized Sequential Feature Compensation."""
from .core import SSFC, AutoAlphaSSFC, pca_feature_impact, sigmoid
from .datasets import AVAILABLE, load_dataset, load_many
from .evaluation import (ARMS, ARM_A, ARM_B, ARM_C, ARM_D, cluster_arms, make_arms,
                         make_transformers, n_to_remove, paired_summary, plot_arms,
                         relative_contrast, run_benchmark, run_comparison, verdict_counts, arm_table)

__version__ = "1.0.0"
