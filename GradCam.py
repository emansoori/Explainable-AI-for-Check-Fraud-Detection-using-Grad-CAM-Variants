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

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    # Compute Grad-CAM weights
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # Apply ReLU and normalization
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    return cam

# Load and preprocess input image
img_path = '/content/drive/MyDrive/Colab Notebooks/9515.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_resized = cv2.resize(img, (224, 224))
img_normalized = np.expand_dims(img_resized / 255.0, axis=0)

# Compute Grad-CAM
preds = net.predict(img_normalized)
class_idx = np.argmax(preds[0])
cam = compute_gradcam(img_normalized, grad_model, class_idx)

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

img_uint8 = np.uint8(img)
superimposed_img = cv2.addWeighted(img_uint8, 0.5, heatmap, 0.5, 0)