import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt

net = models.load_model('/content/drive/MyDrive/Colab Notebooks/CNN_functional_Train_Test.h5')
target_layer = net.get_layer("block5_conv3")
grad_model = models.Model(inputs=[net.input], outputs=[target_layer.output, net.output])


def compute_xgradcam(input_image, model, target_class):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = model(input_image)
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient computation failed. Check the target layer or input.")
    
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradient computation failed. Check the target layer or input.")
    
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = tf.squeeze(conv_outputs).numpy()

    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)  
    cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[1]))
    cam /= cam.max() if cam.max() != 0 else 1
    return cam


img_path = '/content/drive/MyDrive/Colab Notebooks/Test10.Original.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_resized = cv2.resize(img, (224, 224))
img_normalized = np.expand_dims(img_resized / 255.0, axis=0)


preds = net.predict(img_normalized)
class_idx = np.argmax(preds[0])
cam = compute_xgradcam(img_normalized, grad_model, class_idx)

