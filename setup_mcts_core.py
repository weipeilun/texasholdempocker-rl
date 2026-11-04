"""Build the C search-tree node used by rl/splendor_mcts.py:

    python setup_mcts_core.py build_ext --inplace

Without the compiled module the search falls back to the numpy implementation (same results, slower).
-ffp-contract=off keeps every floating-point operation separately rounded, as numpy does, so the
C node gives bit-identical results to the numpy reference.
"""
from setuptools import Extension, setup

setup(
    name='splendor-mcts-core',
    ext_modules=[Extension('rl._mcts_core', ['rl/_mcts_core.c'],
                           extra_compile_args=['-O3', '-ffp-contract=off', '-fno-fast-math'])],
)
