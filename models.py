from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import keras.backend as K
import scipy
import logging
import matplotlib.pyplot as plt
import os

import numpy as np

from utils import *
from kh_tools import *


class ALOCC_Model():
    def __init__(self,
               input_height=28,input_width=28, output_height=28, output_width=28,
               attention_label=1, is_training=True,
               z_dim=100, gf_dim=16, df_dim=16, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               checkpoint_dir='checkpoint', log_dir='log', sample_dir='sample', r_alpha = 0.2,
               kb_work_on_patch=True, nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param sess: TensorFlow session.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param is_training: True if in training mode.
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16] 
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16] 
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: 'UCSD', 'mnist' or custom defined name.
        :param dataset_address: path to dataset folder. e.g. './dataset/mnist'.
        :param input_fname_pattern: Glob pattern of filename of input images e.g. '*'.
        :param checkpoint_dir: path to saved checkpoint(s) directory.
        :param log_dir: log directory for training, can be later viewed in TensorBoard.
        :param sample_dir: Directory address which save some samples [.]
        :param r_alpha: Refinement parameter, trade-off hyperparameter for the G network loss to reconstruct input images. [0.2]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not, only applies to UCSD dataset [True]
        :param nd_patch_size:  Input patch size, only applies to UCSD dataset.
        :param n_stride: PatchBased data preprocessing stride, only applies to UCSD dataset.
        :param n_fetch_data: Fetch size of Data, only applies to UCSD dataset. 
        """

        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.is_training = is_training

        self.r_alpha = r_alpha

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.dataset_name = dataset_name
        self.dataset_address= dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.attention_label = attention_label
        if self.is_training:
          logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

        if self.dataset_name == 'mnist':
          (X_train, y_train), (_, _) = mnist.load_data()
          specific_idx = np.where(y_train == self.attention_label)[0]
          self.data = X_train[specific_idx].reshape(-1, 28, 28, 1)
          self.c_dim = 1
        else:
          assert('Error in loading dataset')

        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_generator(self, input_shape):
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """
        image = Input(shape=input_shape)
        # Encoder.
        x = Conv2D(filters=self.df_dim * 2, kernel_size = 5, strides=2, padding='same', name='g_encoder_h0_conv')(image)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 4, kernel_size = 5, strides=2, padding='same', name='g_encoder_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self.df_dim * 8, kernel_size = 5, strides=2, padding='same', name='g_encoder_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Decoder.
        # TODO: need a flexable solution to select output_padding and padding.
        x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, dilation_rate=1, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, dilation_rate=1, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, dilation_rate=1, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)
        return Model(image, x, name='R')

    def build_discriminator(self, input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """

        image = Input(shape=input_shape)
        x = Conv2D(filters=self.df_dim, kernel_size = 5, strides=2, padding='same', name='d_h0_conv')(image)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*2, kernel_size = 5, strides=2, padding='same', name='d_h1_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*4, kernel_size = 5, strides=2, padding='same', name='d_h2_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters=self.df_dim*8, kernel_size = 5, strides=2, padding='same', name='d_h3_conv')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)
        x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

        return Model(image, x, name='D')

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        # Original image input for generator/R network.
        self.inputs = Input(shape=image_dims, name='real_images')
        inputs = self.inputs

        # Image input to the generator/R.
        self.z = Input(shape=image_dims, name='z')

        # Construct generator/R network.
        self.generator = self.build_generator(image_dims)
        self.G = self.generator(self.z)

        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)
        self.D = self.discriminator(inputs)

        # Construct discriminator/D network takes real image as input.
        # D - sigmoid output.
        self.discriminator = self.build_discriminator(image_dims)
        self.D_ = self.discriminator(self.G)
        
        # Model to train D to discrimate real images.
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')
        print('\n\rdiscriminator')
        self.discriminator.summary()

        self.discriminator.trainable = False
        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.model_train_R = Model(self.z, [self.G, self.D_])
        self.model_train_R.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[self.r_alpha, 1],
            optimizer=optimizer)

        # z->R->D, train D to discriminate generated as "fake" images.
        self.discriminator.trainable = True
        self.generator.trainable = False
        self.model_train_D = Model(self.z, self.D_)
        self.model_train_D.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.discriminator.trainable = False
        self.generator.trainable = True # TODO: set to False after debugging.
        self.sampler = self.generator
        self.sampler.compile(optimizer=optimizer, loss='mse')
        print('\n\rmodel_train_R')
        self.model_train_R.summary()
        print('\n\rmodel_train_D')
        self.model_train_D.summary()
        print('\n\rsampler')
        self.sampler.summary()


    def train_auto_encoder(self, epochs, batch_size = 128, sample_interval=500):
        # Make log folder if not exist.
        log_dir = os.path.join(self.log_dir, self.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        if self.dataset_name == 'mnist':
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.sample_dir, exist_ok=True)
        scipy.misc.imsave('./{}/train_input_samples.jpg'.format(self.sample_dir), montage(sample_inputs[:,:,:,0]))

        counter = 1
        # TODO: Load previously train model if exists.
        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # if could_load:
        #     counter = checkpoint_counter
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")


        # Load traning data, add random noise.
        if self.dataset_name == 'mnist':
            sample_w_noise = get_noisy_data(self.data)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch,epochs))
            if self.dataset_name == 'mnist':
                # Number of batches computed by total number of target data / batch size.
                batch_idxs = len(self.data) // batch_size
             
            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                if self.dataset_name == 'mnist':
                    batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                    batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                if self.dataset_name == 'mnist':
                    # Update R network, minimize reconstruction loss. auto try batch_noise_images
                    g_r_loss = self.sampler.train_on_batch(batch_noise_images, batch_noise_images)
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> g_r_loss: {3:>0.3f}'.format(epoch, idx, batch_idxs, g_r_loss)
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    if self.dataset_name == 'mnist':
                        samples = self.sampler.predict(sample_inputs)
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                            './{}/ae_train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch)
    def train(self, epochs, batch_size = 128, sample_interval=500):
        # Make log folder if not exist.
        log_dir = os.path.join(self.log_dir, self.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        if self.dataset_name == 'mnist':
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.sample_dir, exist_ok=True)
        scipy.misc.imsave('./{}/train_input_samples.jpg'.format(self.sample_dir), montage(sample_inputs[:,:,:,0]))

        counter = 1
        # TODO: Load previously train model if exists.
        # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        # if could_load:
        #     counter = checkpoint_counter
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")


        # Load traning data, add random noise.
        if self.dataset_name == 'mnist':
            sample_w_noise = get_noisy_data(self.data)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch,epochs))
            if self.dataset_name == 'mnist':
                # Number of batches computed by total number of target data / batch size.
                batch_idxs = len(self.data) // batch_size
             
            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                if self.dataset_name == 'mnist':
                    batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                    batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = batch_images #np.array(batch_noise).astype(np.float32)
                if self.dataset_name == 'mnist':
                    # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                    d_loss_real = self.discriminator.train_on_batch(batch_images, ones)
                    d_loss_fake = self.model_train_D.train_on_batch(batch_images, zeros)

                    # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                    self.model_train_R.train_on_batch(batch_noise_images, [batch_noise_images, ones])
                    g_loss = self.model_train_R.train_on_batch(batch_noise_images, [batch_noise_images, ones])
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}'.format(epoch, idx, batch_idxs, d_loss_real+d_loss_fake, g_loss[0])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    if self.dataset_name == 'mnist':
                        samples = self.sampler.predict(sample_inputs)
                        manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        save_images(samples, [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

            # Save the checkpoint end of each epoch.
            self.save(epoch)

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.dataset_name,
            self.output_height, self.output_width)

    def save(self, step):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_name = 'ALOCC_Model_{}.h5'.format(step)
        self.model_train_D.save_weights(os.path.join(self.checkpoint_dir, model_name))


if __name__ == '__main__':
    model = ALOCC_Model(dataset_name='mnist', input_height=28,input_width=28)
    # model.train_auto_encoder(epochs=40, batch_size=128, sample_interval=500)
    model.train(epochs=40, batch_size=128, sample_interval=500)
