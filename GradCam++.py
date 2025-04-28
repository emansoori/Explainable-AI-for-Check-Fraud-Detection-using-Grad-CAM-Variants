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