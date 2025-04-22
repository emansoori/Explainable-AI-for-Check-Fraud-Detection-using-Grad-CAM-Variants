import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


feature_vector = []
labels = []

for i, address in enumerate(glob.glob('/content/drive/MyDrive/Colab Notebooks/Check_Dataset/*/*')):
    img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224,224))
    img = img/255.0


    feature_vector.append(img)
    labels.append(address.split("/")[-2])

    if i % 100 == 0:
        print(f"{i} samples processed")


feature_vector = np.array(feature_vector)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)
print(labels)



base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(100, activation="relu")(x)
x = layers.Dropout(0.6)(x)
output_layer = layers.Dense(2, activation="softmax")(x)

net = models.Model(inputs=base_model.input, outputs=output_layer)

indices = np.arange(len(feature_vector))
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    feature_vector, labels, indices, test_size=0.2, random_state=42
)

print(f"Test data indices: {test_idx}")

net.compile(optimizer='SGD', loss="categorical_crossentropy", metrics=["accuracy"])

H = net.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test), epochs=5)

print(net.summary())

net.save('/content/drive/MyDrive/Colab Notebooks/CNN_functional_Train_Test.h5')

plt.plot(H.history["accuracy"], label="train accuracy")
plt.plot(H.history["val_accuracy"], label="test accuracy")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Check Dataset Classification')

plt.show()



y_pred = net.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))



net = models.load_model('/content/drive/MyDrive/Colab Notebooks/CNN_functional_Train_Test.h5')