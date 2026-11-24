
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1,28,28,1)/255.0
X_test = X_test.reshape(-1,28,28,1)/255.0

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and evaluate
model.fit(X_train, y_train, epochs=2, batch_size=128)
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc*100)

# Predict digits
pred = model.predict(X_test[:5]).argmax(axis=1)

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {pred[i]}")
    plt.axis('off')
plt.show()

# Visualize feature maps
feature_model = Model(model.input, model.layers[0].output)
maps = feature_model.predict(X_test[:1])

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(maps[0,:,:,i], cmap='gray')
    plt.axis('off')
plt.show()
