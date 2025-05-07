import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt

net = models.load_model('/content/drive/MyDrive/Colab Notebooks/CNN_functional_Train_Test.h5')
target_layer = net.get_layer("block5_conv3")
grad_model = models.Model(inputs=[net.input], outputs=[target_layer.output, net.output])
