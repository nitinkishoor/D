
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Training text
text = "This is RNN training Program"

# Character encoding
chars = sorted(set(text))
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}

# Prepare training data
n = 3
seq = [[c2i[c] for c in text[i:i+n]]
       for i in range(len(text)-n)]

x = tf.one_hot(seq, len(chars))
y = tf.one_hot([c2i[text[i+n]]
                for i in range(len(text)-n)], len(chars))

# Create RNN model
model = Sequential([
    SimpleRNN(50, input_shape=(n, len(chars))),
    Dense(len(chars), activation='softmax')
])

# Train model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=100, verbose=0)

# Generate text
s = "This"
for i in range(50):
    a = tf.one_hot([[c2i[c] for c in s[-n:]]], len(chars))
    s += i2c[np.argmax(model.predict(a, verbose=0))]

print("Generated Text:")
print(s)
