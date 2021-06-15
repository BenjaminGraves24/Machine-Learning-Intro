import tensorflow as tf
import numpy as np
from tensorflow import keras


# Define the Neural Network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# Compile the Neural Network
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide Data inputs 
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Training the neural network
model.fit(xs, ys, epochs=500)

# Printing the results
print(model.predict([10.0]))
