
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

(x,_),_=tf.keras.datasets.mnist.load_data()
x=((x-127.5)/127.5).reshape(-1,28,28,1)
ds=tf.data.Dataset.from_tensor_slices(x).shuffle(60000).batch(256)

G=tf.keras.Sequential([
    layers.Dense(7*7*256,use_bias=False,input_shape=(100,)),
    layers.BatchNormalization(),layers.LeakyReLU(),
    layers.Reshape((7,7,256)),
    layers.Conv2DTranspose(128,5,padding='same',use_bias=False),
    layers.BatchNormalization(),layers.LeakyReLU(),
    layers.Conv2DTranspose(64,5,strides=2,padding='same',use_bias=False),
    layers.BatchNormalization(),layers.LeakyReLU(),
    layers.Conv2DTranspose(1,5,strides=2,padding='same',activation='tanh')
])

D=tf.keras.Sequential([
    layers.Conv2D(64,5,strides=2,padding='same',input_shape=(28,28,1)),
    layers.LeakyReLU(),layers.Dropout(.3),
    layers.Conv2D(128,5,strides=2,padding='same'),
    layers.LeakyReLU(),layers.Dropout(.3),
    layers.Flatten(),layers.Dense(1)
])

loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
go=tf.keras.optimizers.Adam(1e-4)
do=tf.keras.optimizers.Adam(1e-4)
seed=tf.random.normal([16,100])

@tf.function
def step(real):
    z=tf.random.normal([tf.shape(real)[0],100])
    with tf.GradientTape() as a,tf.GradientTape() as b:
        fake=G(z,training=True)
        r,f=D(real,training=True),D(fake,training=True)
        gl=loss(tf.ones_like(f),f)
        dl=loss(tf.ones_like(r),r)+loss(tf.zeros_like(f),f)
    go.apply_gradients(zip(a.gradient(gl,G.trainable_variables),G.trainable_variables))
    do.apply_gradients(zip(b.gradient(dl,D.trainable_variables),D.trainable_variables))

for e in range(20):
    for batch in ds: step(batch)
    img=G(seed,training=False)
    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(img[i,:,:,0]*127.5+127.5,cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Generated Images - Epoch {e+1}")
    plt.show()
    print("Epoch",e+1,"finished")

print("Image augmentation using GAN completed")
