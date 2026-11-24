
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import pad_sequences

# Load IMDB dataset
(x, y), (a, b) = imdb.load_data(num_words=1000)

# Pad sequences
x = pad_sequences(x, maxlen=200)
a = pad_sequences(a, maxlen=200)

# Build LSTM model
model = Sequential([
    Embedding(1000, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=64, epochs=3,
          validation_data=(a, b))

# Test accuracy
print("Test Accuracy:",
      model.evaluate(a, b, verbose=0)[1] * 100)
