"""
Utilities for structured MLP pruning based on channel norms.

This package contains helpers to score hidden units and rebuild a VisionTransformerDiffPruning
model after pruning channels from the MLP layers.
"""

from .mlp_pruning import prune_model_mlp_channels, summarize_pruning  # noqa: F401
