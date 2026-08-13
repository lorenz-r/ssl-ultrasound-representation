"""zea picks a Keras backend at import time and defaults to tensorflow, which
is not installed in this project's environment. torch already is (it's the
training dependency), so default to it — set before anything imports zea.
An explicit environment override still wins (setdefault).
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
