"""Kept for the scripts here: the graph library now lives in the package (src/gap_attractor/analysis/graph.py).

    uv run --extra graph python scripts/graph/<script>.py ...
"""
from gap_attractor.analysis.graph import GRAPH_DIR, CoreGraph, bfs, csr, log  # noqa: F401
