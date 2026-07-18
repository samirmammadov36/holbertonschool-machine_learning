#!/usr/bin/env python3
"""Builds a neural network using the Keras Sequential API."""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Build a Keras Sequential neural network.

    Args:
        nx: Number of input features.
        layers: List containing the number of nodes in each layer.
        activations: List containing each layer's activation function.
        lambtha: L2 regularization parameter.
        keep_prob: Probability that a node is retained during dropout.

    Returns:
        The constructed Keras Sequential model.
    """
    model = K.models.Sequential()

    for index, nodes in enumerate(layers):
        if index == 0:
            model.add(
                K.layers.Dense(
                    units=nodes,
                    activation=activations[index],
                    kernel_regularizer=K.regularizers.l2(lambtha),
                    input_shape=(nx,)
                )
            )
        else:
            model.add(
                K.layers.Dense(
                    units=nodes,
                    activation=activations[index],
                    kernel_regularizer=K.regularizers.l2(lambtha)
                )
            )

        if index < len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model

