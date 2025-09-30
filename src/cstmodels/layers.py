
import keras
from keras import layers
from keras import ops


# Dimension of each attention head in the MHSA of SetConv2D
CST15_MHSA_HEAD_DIM = 64


@keras.utils.register_keras_serializable()
class SmartReshape2D(layers.Layer):
    """
    Reshapes 4D or 5D tensors to handle an explicit set dimension.

    This layer is useful when working with data that may or may not have a set dimension
    (e.g., `(batch * set_size, H, W, C)` vs. `(batch, set_size, H, W, C)`).
    It automatically infers the correct shape and reshapes the input tensor accordingly.
    
    Args:
        **kwargs: Keyword arguments for the Layer base class.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def build(self, input_shape):
        """
        This method simply marks the layer as built.

        Args:
            input_shape (shapelike): Shape of the input to the layer.
        """
        self.built = True

    def call(self, x, set_size=None):
        """
        Main logic for the SmartReshape2D layer.

        If the input tensor has 5 dimensions, it is assumed to be in the format
        `(batch, set_size, H, W, C)` and is reshaped to
        `(batch * set_size, H, W, C)`.

        If the input tensor has 4 dimensions, it is assumed to be in the format
        `(batch * set_size, H, W, C)` and is reshaped to
        `(batch, set_size, H, W, C)`, where `set_size` is provided as an argument.

        Args:
            x (tensor): Input tensor of shape `(batch * set_size, H, W, C)`
                        or `(batch, set_size, H, W, C)`.
            set_size (scalar): Size of the set dimension (required if input is 4D, optional if 5D)
              or `None`.

        Returns:
            x (tensor): The reshaped tensor.
            set_size (scalar): The set size.
        """
        tensor_shape = ops.shape(x)
        height = tensor_shape[-3]
        width = tensor_shape[-2]
        channels = tensor_shape[-1]
        n_dims = ops.ndim(x)

        if n_dims == 5:
            # Input is already in (batch, set_size, height, width, channels) format
            # -> Flatten the set dimension
            target_shape = (-1, height, width, channels)
            set_size = ops.shape(x)[1] # Extract set_size from the input shape
        else:
            # Input is in (batch * set_size, height, width, channels) format
            # -> Reshape to include set dimension
            target_shape = (-1, set_size, height, width, channels)

        x = ops.reshape(x, target_shape)

        return x, set_size
    
    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            config (dict): Configuration of the layer for serialization.
        """
        config = super().get_config()
        return config
    

@keras.utils.register_keras_serializable()
class SetConv2D(layers.Layer):
    """
    Implementation of the SetConv2D layer. For more details see Chinello & Boracchi (2025).

    Args:
        filters (int): Number of output filters in the convolution.
        kernel_size (int | tuple): Size of the convolution kernel.
        activation (string | None): Activation function to use.
        mhsa_dropout (float): Dropout rate for the MHSA layer.
        padding (string): Padding mode for convolution (`same` or `valid`).
        strides (int | tuple): Stride size for convolution.
        **kwargs: Additional keyword arguments for the Layer base class.
    """
    def __init__(
            self,
            filters,
            kernel_size,
            activation=None,
            mhsa_dropout=.0,
            padding='same',
            strides=1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.mhsa_dropout = mhsa_dropout
        self.padding = padding
        self.strides = strides

        if not (activation is None or isinstance(activation, str)):
            raise ValueError("Activation must be a string or None")
        self.activation = activation if activation else 'linear'

        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            activation=None,
            padding=self.padding,
            strides=self.strides,
            name='conv'
        )
        self.gap = layers.GlobalAveragePooling2D()
        self.mha = layers.MultiHeadAttention(
            num_heads=max(1, self.filters // CST15_MHSA_HEAD_DIM),
            key_dim=min(self.filters, CST15_MHSA_HEAD_DIM),
            dropout=self.mhsa_dropout,
            name ='mhsa'
        )
        self.activ = layers.Activation(activation=self.activation)

    def build(self, input_shape):
        """
        This method simply marks the layer as built.

        Args:
            input_shape (shapelike): Shape of the input to the layer.
        """
        self.built = True

    def call(self, X, set_size):
        """
        Main logic for the SetConv2D layer.

        Args:
            X (tensor): Input tensor of shape `(batch * set_size, H, W, C)`.
            set_size (scalar): Size of the set dimension.

        Returns:
            X (tensor): Output tensor after applying SetConv2D operations.
        """
        # 1. Convolution
        X = self.conv(X)

        # 2. Compute channel descriptors via GAP
        Y = self.gap(X)

        # 3. Compute bias adjustments via MHSA
        Y = ops.reshape(Y, [-1, set_size, self.filters])
        Y = self.mha(Y, Y)
        Y = ops.reshape(Y, [-1, self.filters])

        # 4. Add dynamic bias adjustments to the output of 1.
        X = X + ops.expand_dims(ops.expand_dims(Y, axis=1), axis=1)

        # 5. Activation
        X = self.activ(X)

        return X

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            config (dict): Configuration of the layer for serialization.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': self.activation,
            'mhsa_dropout': self.mhsa_dropout,
            'padding': self.padding,
            'strides': self.strides,
        })

        return config
    

__all__ = [
    'SmartReshape2D',
    'SetConv2D'
]