import sys
import types
import os
import shutil
import numpy as np


def test_reduce_with_flatter_missing_executable():
    """Standalone test (does not require pytest) that verifies a clear RuntimeError
    is raised when the 'flatter' executable cannot be found."""

    # Insert a minimal fake `fpylll` into sys.modules so the function's import succeeds
    fake = types.SimpleNamespace()

    class FakeIntegerMatrix:
        def __init__(self, mat):
            self._mat = mat

        @staticmethod
        def from_matrix(mat):
            return FakeIntegerMatrix(mat)

        def __str__(self):
            # Return a simple FPLLL-like textual matrix representation
            # This test only needs the import to succeed up to the executable check.
            return "[1 0]\n[0 1]\n"

    fake.IntegerMatrix = FakeIntegerMatrix
    fake.BKZ = None
    fake.GSO = None

    sys.modules["fpylll"] = fake

    # Ensure FLATTER_BIN is unset and that shutil.which will report not found
    old_flat = os.environ.pop("FLATTER_BIN", None)
    old_which = shutil.which
    shutil.which = lambda x: None

    try:
        import importlib.util
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / 'cool' / 'reduction.py'
        spec = importlib.util.spec_from_file_location('cool.reduction', str(p))
        reduction = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reduction)

        try:
            reduction.reduce_with_flatter(np.array([[1, 0], [0, 1]]))
            raise AssertionError("Expected RuntimeError due to missing 'flatter' executable")
        except RuntimeError as e:
            msg = str(e)
            assert "flatter" in msg and "executable not found" in msg
            print("TEST PASSED: Missing 'flatter' raises RuntimeError with helpful message")
    finally:
        # restore environment
        if old_flat is not None:
            os.environ['FLATTER_BIN'] = old_flat
        shutil.which = old_which
        sys.modules.pop('fpylll', None)


if __name__ == '__main__':
    test_reduce_with_flatter_missing_executable()
