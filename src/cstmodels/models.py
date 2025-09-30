import keras
from keras import layers

from .layers import SmartReshape2D, SetConv2D


CST15_URL = 'https://zenodo.org/records/17227254/files/CST15.keras?download=1'


def _build_CST15():
    """
    Builds the CST15 model architecture.
    """
    input_layer = layers.Input(shape=(None, None, None, 3))
    x, l = SmartReshape2D(name='reshape_input')(input_layer)
    x = keras.applications.imagenet_utils.preprocess_input(x, mode='torch')

    # Block 1
    for i in range(2):
        x = layers.Conv2D(
            64, 3, activation='relu6', padding='same', name=f'block1_conv{i+1}'
        )(x)
    x = layers.MaxPooling2D(name='block1_pool')(x)

    # Block 2
    for i in range(2):
        x = layers.Conv2D(
            128, 3, activation='relu6', padding='same', name=f'block2_conv{i+1}'
        )(x)
    x = layers.MaxPooling2D(name='block2_pool')(x)

    # Block 3 (SetConv2D)
    for i in range(2):
        x = SetConv2D(
            256, 3, activation='relu6', padding='same', name=f'block3_setconv{i+1}'
        )(x, set_size=l)
    x = layers.MaxPooling2D(name='block3_pool')(x)

    # Block 4 (SetConv2D)
    for i in range(4):
        x = SetConv2D(
            512, 3, activation='relu6', padding='same', name=f'block4_setconv{i+1}'
        )(x, set_size=l)
    x = layers.MaxPooling2D(name='block4_pool')(x)

    # Block 5 (SetConv2D)
    for i in range(4):
        x = SetConv2D(
            512, 3, activation='relu6', padding='same', name=f'block5_setconv{i+1}'
        )(x, set_size=l)
    x = layers.MaxPooling2D(name='block5_pool')(x)

    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    output_layer = layers.Dense(1000, activation='softmax', name='output')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer, name='CST15')

    return model
    

def CST15(pretrained=True):
    """
    Loads or builds the CST15 model. In both cases, the model is compiled
    with Adam optimizer and Categorical Crossentropy loss.

    Args:
        pretrained (bool): If `True`, loads CST15 pretrained on ImageNet.
            If `False`, builds a new CST15 model from scratch.

    Returns:
        model (KerasModel): The CST15 model instance.
    """
    if not isinstance(pretrained, bool):
        raise ValueError("Pretrained must be a boolean value")
    
    if pretrained:
        # Load the pretrained model
        path = keras.utils.get_file('CST15.keras', origin=CST15_URL)
        model = keras.saving.load_model(path)
        return model
    
    # Build a new model
    model = _build_CST15()
    model.compile(
        optimizer='adam',
        loss='CategoricalCrossentropy',
        metrics=['accuracy']
    )

    return model


__all__ = ['CST15']