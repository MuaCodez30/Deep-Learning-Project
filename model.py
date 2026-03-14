"""
model.py
--------
CNN feature extractor definitions using pretrained Keras backbones.
Both models freeze all backbone weights and attach a single global pooling layer
to produce a fixed-length feature vector per input frame.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2S
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D


def build_mobilenetv2_extractor(pooling="max", input_shape=(224, 224, 3)):
    """
    Build a MobileNetV2 feature extractor pretrained on ImageNet.

    Args:
        pooling:      'max' or 'avg' global pooling after the backbone.
        input_shape:  Expected (H, W, C) — must match FRAME_SIZE used in preprocessing.

    Returns:
        A Keras Model with output shape (batch, 1280).
    """
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False   # Freeze backbone — used only as a fixed feature extractor
    x = GlobalMaxPooling2D()(base.output) if pooling != "avg" else GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=x)


def build_efficientnetv2_extractor(pooling="max", input_shape=(224, 224, 3)):
    """
    Build an EfficientNetV2-S feature extractor pretrained on ImageNet.
    Heavier than MobileNetV2 but generally achieves higher accuracy.

    Args:
        pooling:      'max' or 'avg' global pooling after the backbone.
        input_shape:  Expected (H, W, C) — must match FRAME_SIZE used in preprocessing.

    Returns:
        A Keras Model with output shape (batch, 1280).
    """
    base = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False   # Freeze backbone — used only as a fixed feature extractor
    x = GlobalMaxPooling2D()(base.output) if pooling != "avg" else GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=x)
