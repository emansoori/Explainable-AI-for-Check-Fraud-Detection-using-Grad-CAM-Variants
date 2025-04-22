import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def change_brightness(image, value=50):
    bright = np.clip(image + value, 0, 255)
    return bright

def add_gaussian_noise(image, mean=0, std=30):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def custom_preprocess(image):
    image = change_brightness(image, value=50)
    image = add_gaussian_noise(image, mean=0, std=10)
    return image

image = cv2.imread(r'Final-Colab\5601.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

aug = ImageDataGenerator(
    rotation_range=5, 
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    fill_mode="nearest"
)

ImageGen = aug.flow(
    np.array([image]), 
    batch_size=1, 
    save_to_dir='out', 
    save_format='jpg', 
    save_prefix='cv'
)

total = 0
for img_batch in ImageGen:
    total += 1
    if total == 10: 
        break

for i in range(5):
    noisy_image = add_gaussian_noise(image, mean=0, std=10 * (i + 1))
    bright_image = change_brightness(image, value=50 * (i + 1))
    
    cv2.imwrite(f'out/noisy_image_{i+1}.jpg', noisy_image)
    cv2.imwrite(f'out/bright_image_{i+1}.jpg', bright_image)
