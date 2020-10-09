import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def search_layer(inputs, name, exclude=None):
    """search layers by name in inpus.
       inputs: a layer or the output of a layer；
       name: the name of the layer you want to search。
    """
    if exclude is None:
        exclude = set()

    if isinstance(inputs, tf.keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude:
        return None
    else:
        exclude.add(layer)
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude)
                if layer is not None:
                    return layer


def adversarial_training(model, embedding_name, epsilon=1):
    """Add adversal perturbations to Embedding.
       model: the model to add adversal perturbations.
       embedding_name: the name of the embedding layer which to add perturbations.
    """
    if model.train_function is None:  # if no train func
        model._make_train_function()  # make train func 
    old_train_function = model.train_function 

    # search embedding layer
    embedding_layer = None
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # get gradients of embedding
    embeddings = embedding_layer.embeddings  # Embedding Matrix
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding Gradients
    gradients = K.zeros_like(embeddings) + gradients[0]  # transfer to dense tensor

    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name=embedding_name+'gradients',
    )  

    def train_function(inputs):  # define train func
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # calc perturbations
        K.set_value(embeddings, K.eval(embeddings) + delta)  # set perturbations
        outputs = old_train_function(inputs)  # gradients decent
        K.set_value(embeddings, K.eval(embeddings) - delta)  # del perturbations
        return outputs

    model.train_function = train_function  # cover raw train func