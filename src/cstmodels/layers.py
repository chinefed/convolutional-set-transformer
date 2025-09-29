
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
    (e.g., (batch * set_size, height, width, channels) vs. (batch, set_size, height, width, channels)).
    It automatically infers the correct shape and reshapes the input tensor accordingly.
    Methods
    -------
    build(input_shape):
        Standard Keras build method. No weights are created in this layer.
    call(x, set_size=None):
        Reshapes the input tensor `x` based on its number of dimensions.
        If `x` is 5D, flattens the set dimension into the batch dimension.
        If `x` is 4D, expands the tensor to include the set dimension using the provided `set_size`.
        Returns the reshaped tensor and the set size.
    get_config():
        Returns the configuration of the layer for serialization.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def build(self, input_shape):
        self.built = True

    def call(self, x, set_size=None):
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
        config = super().get_config()
        return config
    

@keras.utils.register_keras_serializable()
class SetConv2D(layers.Layer):
    """
    Implementation of the SetConv2D layer.

    Args:
        filters (int): Number of output filters in the convolution.
        kernel_size (int or tuple): Size of the convolution kernel.
        activation (str or None, optional): Activation function to use. Defaults to 'linear' if None.
        mhsa_dropout (float, optional): Dropout rate for the MHSA layer. Defaults to 0.0.
        padding (str, optional): Padding mode for convolution ('same' or 'valid'). Defaults to 'same'.
        strides (int or tuple, optional): Stride size for convolution. Defaults to 1.
        **kwargs: Additional keyword arguments for the Layer base class.

    Methods:
        build(input_shape):
            Builds the layer (no weights to initialize).
        call(X, set_size):
            Main logic for the SetConv2D layer. Receives input tensor and set size.
        get_config():
            Returns the configuration of the layer for serialization.
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
        self.built = True

    def call(self, X, set_size):
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