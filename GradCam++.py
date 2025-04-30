from tensorflow import keras
from keras import layers, models
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

net = models.load_model('/content/drive/MyDrive/Colab Notebooks/CNN_functional_Train_Test.h5')

target_layer = net.get_layer("block5_conv3")

grad_model = models.Model(
    inputs=[net.inputs],
    outputs=[target_layer.output, net.output]
)


def compute_gradcam_plus(input_image, model, target_class):
    with tf.GradientTape(persistent=True) as tape:
        conv_outputs, predictions = model(input_image)
        loss = predictions[:, target_class]



    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients could not be calculated. Check the target layer and input configurations.")

    second_grads = tape.gradient(grads, conv_outputs)
    if second_grads is None:
        print("Second-order gradients could not be calculated. Using Grad-CAM instead.")
        second_grads = tf.zeros_like(grads)

    
    conv_outputs = tf.squeeze(conv_outputs).numpy()
    grads = tf.squeeze(grads).numpy()
    second_grads = tf.squeeze(second_grads).numpy()


    grads_power = np.power(grads, 2)
    second_grads_power = np.power(second_grads, 3)
    denominator = 2 * grads + second_grads_power
    denominator = np.where(denominator != 0, denominator, np.ones_like(denominator))
    alpha = grads_power / denominator