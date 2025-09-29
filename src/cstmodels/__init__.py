import warnings
import keras

# Check which backend is active
backend = keras.backend.backend()

if backend != 'tensorflow':
    warnings.warn(
        f"[cstmodels] Detected Keras backend '{backend}', "
        "but this package currently supports only TensorFlow.",
        UserWarning
    )

from .layers import *
from .models import *

from .layers import __all__ as _layers_all
from .models import __all__ as _models_all

__all__ = _layers_all + _models_all