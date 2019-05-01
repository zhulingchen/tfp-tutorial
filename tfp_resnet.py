"""
Trains a ResNet on the CIFAR10 dataset using Keras and Tensorflow Probability.

ResNet v1:
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

ResNet v2:
[Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf)

Model parameter
-----------------------------------------------------------------------------------
          |             | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
Model     | n_res_block | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
          |    v1(v2)   | %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
-----------------------------------------------------------------------------------
ResNet20  |     3 (2)   | 92.16     | 91.25     | -----     | -----     | 35 (---)
ResNet32  |     5(NA)   | 92.46     | 92.49     | NA        | NA        | 50 ( NA)
ResNet44  |     7(NA)   | 92.50     | 92.83     | NA        | NA        | 70 ( NA)
ResNet56  |     9 (6)   | 92.71     | 93.03     | 93.01     | NA        | 90 (100)
ResNet110 |    18(12)   | 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
ResNet164 |    27(18)   | -----     | 94.07     | -----     | 94.54     | ---(---)
ResNet1001|   NA(111)   | -----     | 92.39     | -----     | 95.08+-.14| ---(---)
-----------------------------------------------------------------------------------
"""
from __future__ import print_function
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # set up tensorflow backend for keras
import keras
from keras.utils import plot_model
from keras.datasets import cifar10
import numpy as np


def lr_schedule(epoch):
    """
    Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def get_kernel_posterior_fn(kernel_posterior_scale_mean=-9.0,
                            kernel_posterior_scale_stddev=0.1,
                            kernel_posterior_scale_constraint=0.2):
    """
    Get the kernel posterior distribution

    # Arguments
        kernel_posterior_scale_mean (float): kernel posterior's scale mean.
        kernel_posterior_scale_stddev (float): the initial kernel posterior's scale stddev.
          ```
          q(W|x) ~ N(mu, var),
          log_var ~ N(kernel_posterior_scale_mean, kernel_posterior_scale_stddev)
          ````
        kernel_posterior_scale_constraint (float): the log value to constrain the log variance throughout training.
          i.e. log_var <= log(kernel_posterior_scale_constraint).

    # Returns
        kernel_posterior_fn: kernel posterior distribution
    """

    def _untransformed_scale_constraint(t):
        return tf.clip_by_value(t, -1000, tf.math.log(kernel_posterior_scale_constraint))

    kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(
        untransformed_scale_initializer=tf.random_normal_initializer(
            mean=kernel_posterior_scale_mean,
            stddev=kernel_posterior_scale_stddev),
        untransformed_scale_constraint=_untransformed_scale_constraint)
    return kernel_posterior_fn


def get_kernel_divergence_fn(train_size, w=1.0):
    """
    Get the kernel Kullback-Leibler divergence function

    # Arguments
        train_size (int): size of the training dataset for normalization
        w (float): weight to the function

    # Returns
        kernel_divergence_fn: kernel Kullback-Leibler divergence function
    """
    def kernel_divergence_fn(q, p, _):  # need the third ignorable argument
        kernel_divergence = tfp.distributions.kl_divergence(q, p) / tf.cast(train_size, tf.float32)
        return w * kernel_divergence
    return kernel_divergence_fn


def get_neg_log_likelihood_fn(bayesian=False):
    """
    Get the negative log-likelihood function
    # Arguments
        bayesian(bool): Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        a negative log-likelihood function
    """
    if bayesian:
        def neg_log_likelihood_bayesian(y_true, y_pred):
            labels_distribution = tfp.distributions.Categorical(logits=y_pred)
            log_likelihood = labels_distribution.log_prob(tf.argmax(input=y_true, axis=1))
            loss = -tf.reduce_mean(input_tensor=log_likelihood)
            return loss
        return neg_log_likelihood_bayesian
    else:
        def neg_log_likelihood(y_true, y_pred):
            y_pred_softmax = keras.layers.Activation('softmax')(y_pred)  # logits to softmax
            loss = keras.losses.categorical_crossentropy(y_true, y_pred_softmax)
            return loss
        return neg_log_likelihood


def get_categorical_accuracy_fn(y_true, y_pred):
    y_pred_softmax = keras.layers.Activation('softmax')(y_pred)  # logits to softmax
    acc = keras.metrics.categorical_accuracy(y_true, y_pred_softmax)
    return acc


class KLLossScheduler(tf.keras.callbacks.Callback):
    def __init__(self, update_per_batch=False, annealing_factor=50, verbose=0):
        self.update_per_batch = update_per_batch
        self.annealing_factor = annealing_factor
        self.verbose = verbose
    def on_batch_begin(self, batch, logs=None):
        if self.update_per_batch:
            idx_total_batch = self.epoch * int(np.ceil(self.params['samples'] / self.params['batch_size'])) + batch
            kl_weight = (idx_total_batch / self.annealing_factor) / (self.params['samples'] / self.params['batch_size'])
            kl_weight = np.minimum(kl_weight, 1.0)
            self.kl_weight = kl_weight
            if self.verbose > 0:
                print('\nBatch: {}, KL Divergence Loss Weight = {:.6f}'.format(batch+1, kl_weight))
            for l in self.model.layers:
                for id_w, w in enumerate(l.weights):
                    if 'kl_loss_weight' in w.name:
                        l_weights = l.get_weights()
                        l.set_weights([*l_weights[:id_w], kl_weight, *l_weights[id_w+1:]])
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if not self.update_per_batch:
            kl_weight = (epoch + 1) / self.annealing_factor
            kl_weight = np.minimum(kl_weight, 1.0)
            self.kl_weight = kl_weight
            if self.verbose > 0:
                print('\nEpoch: {}, KL Divergence Loss Weight = {:.6f}'.format(epoch+1, kl_weight))
            for l in self.model.layers:
                for id_w, w in enumerate(l.weights):
                    if 'kl_loss_weight' in w.name:
                        l_weights = l.get_weights()
                        l.set_weights([*l_weights[:id_w], kl_weight, *l_weights[id_w+1:]])
    def on_epoch_end(self, epoch, logs={}):
        print('KL Divergence Weight = {:.6f}, KL Divergence Loss = {:.4f}'.format(self.kl_weight,
                                                                                  sum(self.model.losses).eval(session=tf.keras.backend.get_session())))


def resnet_layer(inputs, train_size,
                 n_filter=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 bayesian=False):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        n_filter (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
        bayesian (bool): implement Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    if bayesian:
        # scale the KL divergence function to avoid the loss function being over-regularized
        conv = tfp.layers.Convolution2DFlipout(n_filter,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding='same',
                                               kernel_posterior_fn=get_kernel_posterior_fn(),
                                               kernel_divergence_fn=None)
        w = conv.add_weight(name=conv.name+'/kl_loss_weight', shape=(), initializer=tf.initializers.constant(1.0), trainable=False)
        conv.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)
    else:
        conv = keras.layers.Conv2D(n_filter,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=keras.regularizers.l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = keras.layers.BatchNormalization()(x)
        if activation is not None:
            x = keras.layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, n_res_block, train_size, n_class=10, bayesian=False):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        n_res_block (int): number of residual blocks
        n_class (int): number of classes (CIFAR10 has 10)
        bayesian (bool): implement Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        model (Model): Keras model instance
    """
    n_filter = 16

    inputs = keras.layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, train_size=train_size, bayesian=bayesian)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(n_res_block):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x, train_size=train_size,
                             n_filter=n_filter,
                             strides=strides,
                             bayesian=bayesian)
            y = resnet_layer(inputs=y, train_size=train_size,
                             n_filter=n_filter,
                             activation=None,
                             bayesian=bayesian)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, train_size=train_size,
                                 n_filter=n_filter,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 bayesian=bayesian)
            x = keras.layers.add([x, y])
            x = keras.layers.Activation('relu')(x)
        n_filter *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = keras.layers.AveragePooling2D(pool_size=8)(x)
    y = keras.layers.Flatten()(x)
    if bayesian:
        # scale the KL divergence function to avoid the loss function being over-regularized
        dense = tfp.layers.DenseFlipout(n_class,
                                        activation=None,
                                        kernel_posterior_fn=get_kernel_posterior_fn(),
                                        kernel_divergence_fn=None)
        w = dense.add_weight(name=dense.name+'/kl_loss_weight', shape=(), initializer=tf.initializers.constant(1.0), trainable=False)
        dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)
        logits = dense(y)
    else:
        logits = keras.layers.Dense(n_class,
                                    activation=None,
                                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=logits)

    return model


def resnet_v2(input_shape, n_res_block, train_size, n_class=10, bayesian=False):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        n_res_block (int): number of residual blocks
        n_class (int): number of classes (CIFAR10 has 10)
        bayesian (bool): implement Bayesian neural network (True) or point-estimate neural network (False)

    # Returns
        model (Model): Keras model instance
    """
    n_filter_in = 16

    inputs = keras.layers.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, train_size=train_size,
                     n_filter=n_filter_in,
                     conv_first=True,
                     bayesian=bayesian)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(n_res_block):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                n_filter_out = n_filter_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                n_filter_out = n_filter_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x, train_size=train_size,
                             n_filter=n_filter_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False,
                             bayesian=bayesian)
            y = resnet_layer(inputs=y, train_size=train_size,
                             n_filter=n_filter_in,
                             conv_first=False,
                             bayesian=bayesian)
            y = resnet_layer(inputs=y, train_size=train_size,
                             n_filter=n_filter_out,
                             kernel_size=1,
                             conv_first=False,
                             bayesian=bayesian)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, train_size=train_size,
                                 n_filter=n_filter_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 bayesian=bayesian)
            x = keras.layers.add([x, y])

        n_filter_in = n_filter_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=8)(x)
    y = keras.layers.Flatten()(x)
    if bayesian:
        # scale the KL divergence function to avoid the loss function being over-regularized
        dense = tfp.layers.DenseFlipout(n_class,
                                        activation=None,
                                        kernel_posterior_fn=get_kernel_posterior_fn(),
                                        kernel_divergence_fn=None)
        w = dense.add_weight(name=dense.name+'/kl_loss_weight', shape=(), initializer=tf.initializers.constant(1.0), trainable=False)
        dense.kernel_divergence_fn = get_kernel_divergence_fn(train_size, w)
        logits = dense(y)
    else:
        logits = keras.layers.Dense(n_class,
                                    activation=None,
                                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model = keras.Model(inputs=inputs, outputs=logits)

    return model


if __name__ == '__main__':

    # Bayesian mode setting
    bayesian = True
    if bayesian:
        keras = tf.keras

    n_mc_run = 20 if bayesian else 1

    # Training parameters
    batch_size = 128  # orig paper trained all networks with batch_size=128
    epochs = 200
    data_augmentation = False
    n_class = 10

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    n_res_block = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 2
    assert version in [1, 2], 'ResNet version must be 1 or 2.'

    # Computed depth from supplied model parameter n_res_block
    depth = n_res_block * 6 + 2 if version == 1 else n_res_block * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    if bayesian:
        model_type += '_Bayesian'

    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, n_class)
    y_test = keras.utils.to_categorical(y_test, n_class)

    if version == 1:
        model = resnet_v1(input_shape=input_shape,
                          n_res_block=n_res_block,
                          train_size=len(x_train),
                          bayesian=bayesian)
    else:
        model = resnet_v2(input_shape=input_shape,
                          n_res_block=n_res_block,
                          train_size=len(x_train),
                          bayesian=bayesian)

    model.compile(loss=get_neg_log_likelihood_fn(bayesian=bayesian),
                  optimizer=keras.optimizers.Adam(lr=lr_schedule(0)),
                  metrics=[get_categorical_accuracy_fn])
    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_vis_name = 'cifar10_%s_model.pdf' % model_type
    model_name = 'cifar10_%s_model.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_vis_filepath = os.path.join(save_dir, model_vis_name)
    model_filepath = os.path.join(save_dir, model_name)

    # Plot the model
    plot_model(model, to_file=model_vis_filepath, show_shapes=True)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_filepath,
                                                 monitor='val_get_categorical_accuracy_fn',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                   cooldown=0,
                                                   patience=5,
                                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    if bayesian:
        kl_loss_scheduler = KLLossScheduler(update_per_batch=True)
        callbacks += [kl_loss_scheduler]

    # Run training, with or without data augmentation.
    if not os.path.isfile(model_filepath):
        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      callbacks=callbacks)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = keras.preprocessing.image.ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                steps_per_epoch=len(x_train) / batch_size,
                                validation_data=(x_test, y_test),
                                epochs=epochs, verbose=1, workers=4,
                                callbacks=callbacks)
    model.load_weights(model_filepath)  # load the optimal model with the lowest validation loss

    # apply the model on test data
    y_pred_logits = [model.predict(x_test) for _ in range(n_mc_run)]
    y_pred_logits = np.concatenate([y[np.newaxis, :, :] for y in y_pred_logits], axis=0)
    y_pred_logits_mean = np.mean(y_pred_logits, axis=0)
    y_pred_logits_std = np.std(y_pred_logits, axis=0)

    y_pred_softmax = keras.layers.Activation('softmax')(keras.backend.variable(y_pred_logits_mean)).eval(session=keras.backend.get_session())
    print('Test accuracy: ', sum(np.equal(np.argmax(y_test, axis=-1), np.argmax(y_pred_softmax, axis=-1))) / len(y_test))