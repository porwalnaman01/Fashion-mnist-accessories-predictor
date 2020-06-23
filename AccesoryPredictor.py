import tensorflow as tf
from tensorflow import keras
import numpy as np



fashion_minist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels ) = fashion_minist.load_data()

# now we normalize all our images in range 0 to 1 before feeding them into neural network

train_images = train_images/255
test_images = test_images/255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

prediction_model = tf.keras.Sequential([model, keras.layers.Softmax()])
predictions = prediction_model.predict(test_images)
print(class_names[np.argmax(predictions[0])])






