
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, utils

# Load dataset
data = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
X = data.images.reshape(-1, data.images.shape[1], data.images.shape[2], 1)
y = utils.to_categorical(data.target)

# Split dataset
x1, x2, y1, y2 = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(y.shape[1], activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x1, y1, epochs=15, batch_size=32,
                    validation_data=(x2, y2))

# Test accuracy
loss, acc = model.evaluate(x2, y2, verbose=0)
print("Test Accuracy:", round(acc * 100, 2), "%")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
