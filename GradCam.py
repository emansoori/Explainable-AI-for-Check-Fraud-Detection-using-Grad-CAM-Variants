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

def compute_gradcam(input_image, model, target_class):
    print("Input image shape:", input_image.shape)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        predictions = tf.convert_to_tensor(predictions)
        print("Predictions shape:", predictions.shape)
        loss = predictions[0, target_class]