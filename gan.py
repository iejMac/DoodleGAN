import numpy as np
import os 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from keras.optimizers import Adam
from keras.datasets import mnist 
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation

class DoodleGAN:
    def __init__(self, seed_size, image_shape, batch_size):
        self.seed_size = seed_size
        self.image_shape = image_shape

        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

        self.generator = self.create_generator()

        self.discriminator = self.create_discriminator()
        self.discriminator.compile(optimizer=self.discriminator_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.gan = self.create_gan()

        self.batch_size = batch_size

        self.upsampler = self.create_upsampling_model()

    def create_upsampling_model(self):
        model = Sequential()
        model.add(UpSampling2D(input_shape=(32,32,3)))
        model.add(UpSampling2D())
        model.add(UpSampling2D())
        model.compile()
        return model

    def create_generator(self):
        model = Sequential()

        model.add(Dense(4*4*256, input_dim=self.seed_size))
        model.add(Reshape((4,4,256)))
        model.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=3, kernel_size=(5, 5), strides=1, padding='same', activation='tanh'))

        print("Gen summary:")
        model.summary()

        return model


    def create_discriminator(self):
        model = Sequential()

        model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', input_shape=self.image_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        print("Disc summary:")
        model.summary()

        return model


    def create_gan(self):
        self.discriminator.trainable = False
        model = Sequential()

        model.add(self.generator)
        model.add(self.discriminator)

        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, epochs, const_seed):
        # Load data:
        (X_train, _), (_, _) = cifar10.load_data()

        X_train = (X_train.astype(np.float32) - 127.5)/127.5

        half_batch = int(self.batch_size/2)

        const_seed = np.random.normal(0, 1, (1, self.seed_size))

        for epoch in range(epochs):

          # Disc train:

          for i in range(2):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.seed_size))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

          # Gen train:

          noise = np.random.normal(0, 1, (self.batch_size, self.seed_size))
          valid_y = np.array([1] * self.batch_size)
          g_loss = self.gan.train_on_batch(noise, valid_y)

          print ("%d Dics loss: %f, Accuracy: %.2f%% | Generator loss: %f" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

          if epoch % 5 == 0:
            const_img = self.generator.predict(const_seed).reshape(1, 32, 32, 3)
            const_img = self.upsampler.predict(const_img)
            const_img = const_img.reshape(256, 256, 3)

            const_img = (const_img+1)/2

	    # Taking out save part:
            # plt.imsave(fname=f"glimpse{epoch/5}.png", arr=const_img, format='png')

            plt.imshow(const_img)
            plt.show()

    def save_plot(self, examples, epoch, n=1):
      examples = (examples + 1) / 2.0
      for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples)
      filename = 'generated_plot_e%03d.png' % (epoch+1)
      plt.savefig(filename)
      plt.close()

    def generate(self):
        seed = tf.random.normal([1, self.seed_size])
        img = self.generator(seed)
        return img

    def save_gan(self):
        tf.keras.models.save_model(self.generator, 'generator')
        tf.keras.models.save_model(self.discriminator, 'discriminator')
        return


doodle_gan = DoodleGAN(100, (32, 32, 3), 128)
const_seed = np.random.normal(0, 1, (1, doodle_gan.seed_size))

doodle_gan.train(40000, const_seed)



