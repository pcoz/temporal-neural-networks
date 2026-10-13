"""`import tnn` must work without scikit-learn (only clustered discovery needs it)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_import_without_sklearn():
    code = ("import sys; sys.modules['sklearn'] = None; "      # make sklearn unimportable
            "sys.path.insert(0, %r); import tnn; "
            "from tnn import TemporalNeuron, NeuronType; print('ok')" % ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
    print("  PASS  import tnn without scikit-learn")


if __name__ == "__main__":
    test_import_without_sklearn()
