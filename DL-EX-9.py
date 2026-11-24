
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

x=np.random.rand(500,128,128,3)
y=np.random.randint(0,5,500)

m=models.Sequential([
    layers.Input((128,128,3)),
    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(5,activation='softmax')
])

m.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
h=m.fit(x,y,epochs=10,validation_split=.2)

print("Accuracy:",m.evaluate(x,y,verbose=0)[1]*100)
plt.plot(h.history['accuracy'],label='Training')
plt.plot(h.history['val_accuracy'],label='Validation')
plt.title("Flower Classification Accuracy")
plt.legend()
plt.show()

m.save("flower_model.keras")
print("Flower Classification Completed")
