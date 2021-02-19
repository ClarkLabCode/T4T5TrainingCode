import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

import h5py as h
import numpy as np
import scipy.io as sio
from math import log, sqrt


def mult_by_noise(input_tensor, noise_std):
    if (noise_std is not None) and (noise_std != 0.0):
        target_mean = 1.0
        target_std = float(noise_std)
        target_var = target_std**2
        source_mean = log((target_mean**2)/sqrt(target_var + target_mean**2))
        source_std = sqrt(log(target_var/(target_mean**2) + 1))
        mult_noise = Lambda(lambda x: x * tf.math.exp(tf.random.normal(shape=K.shape(x),mean=source_mean, stddev=source_std)))
        return mult_noise(input_tensor)
    else:
        return input_tensor


def ln_model_flip(input_shape, filter_shape=(30, 3), num_filter=4, output_noise_std=None, predict_direction = False):

    image_in = Input(input_shape)

    assert (np.mod(num_filter, 2) == 0)
    num_free_filters = int(num_filter / 2) #For each freely parameterized filter there is a reversed filter

    pad_x = int((filter_shape[1] - 1))
    pad_t = int((filter_shape[0] - 1))

    #An LN model maps nicely no a Conv2D. First we define the output of a single arm
    input_conv = Conv2D(num_free_filters, filter_shape, name='conv1', activation='relu')
    conv1 = input_conv(image_in)

    #Now we define the output of paired, revered filters. To do this, we do a flip(convolve(flip(input))) operation
    reverseLayer2 = Lambda(lambda x: K.reverse(x, axes=2))
    reversedInput = reverseLayer2(image_in)
    conv2 = reverseLayer2(input_conv(reversedInput))

    #Add output noise
    noised_conv1 = mult_by_noise(conv1, output_noise_std)
    noised_conv2 = mult_by_noise(conv2, output_noise_std)

    #Subtract the filter pairs
    subtracted_layer = subtract([noised_conv1, noised_conv2])

    # Velocity estimate is the weighted sum of the LN model outputs
    # If we want to just the direction of the motion, we add a sigmoid nonlinearity
    final_activation = 'sigmoid' if predict_direction else None
    combine_filters = Conv2D(1, (1, 1), name='conv2', activation=final_activation,
                             use_bias=False)(subtracted_layer)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='ln_model_flip')

    return model, pad_x, pad_t


def lnln_model_flip(input_shape, filter_shape=(30, 3), num_filter=4, output_noise_std=None, predict_direction = False):

    # For the LNLN and Conductance models, the true spatial extent of the filters
    # will always be 3, but it's going to be hardcoded in the model. The spatial 
    # extent of the first LN filter is 1, so we're going to hardcode that here
    filter_shape[1] = 1

    assert (np.mod(num_filter, 2) == 0)
    num_free_filters = int(num_filter / 2)

    pad_x = int((filter_shape[1] - 1)) + 2
    pad_t = int((filter_shape[0] - 1))

    # We're going to split up the input into three separate channels: s1,s2 and s3
    # These correspond to the three spatial inputs
    image_in = Input(input_shape)

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(image_in)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(image_in)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(image_in)

    # Each input channel will be convolved with a corresponding filter "g"
    g1 = Conv2D(num_free_filters, filter_shape, name='g1', activation='relu')
    g2 = Conv2D(num_free_filters, filter_shape, name='g2', activation='relu')
    g3 = Conv2D(num_free_filters, filter_shape, name='g3', activation='relu')

    # The order of these filters is different for the normal filter and its corresponding reversed filter
    # gX_Y now refers to the result of applying the first convolutions where X is the location in space
    # and Y refers to the original or reversed arm.
    g2_both = g2(s2)

    g1_1 = g1(s1)
    g1_2 = g1(s3)

    g3_1 = g3(s3)
    g3_2 = g3(s1)

    expand_last = Lambda(lambda lam: K.expand_dims(lam, axis=-1))
    squeeze_last = Lambda(lambda lam: K.squeeze(lam, axis=-1))

    g2_both = expand_last(g2_both)
    g1_1 = expand_last(g1_1)
    g1_2 = expand_last(g1_2)
    g3_1 = expand_last(g3_1)
    g3_2 = expand_last(g3_2)

    # Combine the three outputs of the first LN filters into one tensor by concatenating in the 4th dimension
    # This lets us do a weighted sum in the next with a 1x1x1 filter convolution
    combined_g_1 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_1, g2_both, g3_1])
    combined_g_2 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_2, g2_both, g3_2])

    second_ln = Conv3D(1, (1, 1, 1), name='second_ln', activation='relu')

    h_1 = second_ln(combined_g_1)
    h_2 = second_ln(combined_g_2)

    noised_h_1 = mult_by_noise(h_1, output_noise_std)
    noised_h_2 = mult_by_noise(h_2, output_noise_std)

    subtracted_layer = squeeze_last(subtract([noised_h_1, noised_h_2]))

    final_activation = 'sigmoid' if predict_direction else None
    combine_filters = Conv2D(1, (1, 1), name='conv2', use_bias=False,
                             activation=final_activation)(subtracted_layer)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='lnln_model_flip')

    return model, pad_x, pad_t


def conductance_model_flip(input_shape, filter_shape=(30, 3), num_filter=4, output_noise_std=None,  predict_direction = False):
    # Conductance model looks like the LNLN model with a biophysical nonlinearity at the end

    filter_shape[1] = 1

    assert(np.mod(num_filter, 2) == 0)
    num_free_filters = int(num_filter/2)

    pad_x = int((filter_shape[1] - 1))+2
    pad_t = int((filter_shape[0] - 1))

    image_in = Input(input_shape)

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(image_in)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(image_in)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(image_in)

    g1 = Conv2D(num_free_filters, filter_shape, name='g1', activation='relu')
    g2 = Conv2D(num_free_filters, filter_shape, name='g2', activation='relu')
    g3 = Conv2D(num_free_filters, filter_shape, name='g3', activation='relu')

    g2_both = g2(s2)

    g1_1 = g1(s1)
    g1_2 = g1(s3)

    g3_1 = g3(s3)
    g3_2 = g3(s1)

    expand_last = Lambda(lambda lam: K.expand_dims(lam, axis=-1))
    squeeze_last = Lambda(lambda lam: K.squeeze(lam, axis=-1))

    g2_both = expand_last(g2_both)
    g1_1 = expand_last(g1_1)
    g1_2 = expand_last(g1_2)
    g3_1 = expand_last(g3_1)
    g3_2 = expand_last(g3_2)

    # The combined outputs of the first LN are weighted to create the numerator for the biophysical nonlinearity.
    # The denominator is 1 + the unweighted outputs of the first LN. The modeled membrane voltage vm_X = numerator/denominator
    numerator_in_1 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_1, g2_both, g3_1])
    numerator_in_2 = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1_2, g2_both, g3_2])

    numerator_comb = Conv3D(1, (1, 1, 1), strides=(1, 1, 1), name='create_numerator', use_bias=False)

    numerator_1 = numerator_comb(numerator_in_1)
    denominator_1 = Lambda(lambda inputs: 1 + inputs[0] + inputs[1] + inputs[2])([g1_1, g2_both, g3_1])
    vm_1 = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator_1, denominator_1])

    numerator_2 = numerator_comb(numerator_in_2)
    denominator_2 = Lambda(lambda inputs: 1 + inputs[0] + inputs[1] + inputs[2])([g1_2, g2_both, g3_2])
    vm_2 = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator_2, denominator_2])

    vm_1 = squeeze_last(vm_1)
    vm_2 = squeeze_last(vm_2)

    # Now we bias and rectify the voltages
    bias_layer = BiasLayer()
    vm_1_bias = bias_layer(vm_1)
    vm_2_bias = bias_layer(vm_2)

    vm_1_rect = Lambda(lambda lam: K.relu(lam))(vm_1_bias)
    vm_2_rect = Lambda(lambda lam: K.relu(lam))(vm_2_bias)

    noised_vm_1_rect = mult_by_noise(vm_1_rect, output_noise_std)
    noised_vm_2_rect = mult_by_noise(vm_2_rect, output_noise_std)

    vm = subtract([noised_vm_1_rect, noised_vm_2_rect])

    final_activation = 'sigmoid' if predict_direction else None
    combine_filters = Conv2D(1, (1, 1), name='conv2', use_bias=False,
                             activation=final_activation)(vm)

    # Create model
    model = Model(inputs=image_in, outputs=combine_filters, name='conductance_model_flip')

    return model, pad_x, pad_t


def load_data_rr(path):
    mat_contents = h.File(path, 'r')

    train_in = mat_contents['train_in'][:]
    train_out = mat_contents['train_out'][:]
    dev_in = mat_contents['dev_in'][:]
    dev_out = mat_contents['dev_out'][:]
    test_in = mat_contents['test_in'][:]
    test_out = mat_contents['test_out'][:]

    sample_freq = mat_contents['sampleFreq'][:]
    phase_step = mat_contents['phaseStep'][:]

    train_in = np.expand_dims(train_in, axis=3)
    dev_in = np.expand_dims(dev_in, axis=3)
    test_in = np.expand_dims(test_in, axis=3)

    train_out = np.expand_dims(train_out, axis=2)
    train_out = np.expand_dims(train_out, axis=3)
    dev_out = np.expand_dims(dev_out, axis=2)
    dev_out = np.expand_dims(dev_out, axis=3)
    test_out = np.expand_dims(test_out, axis=2)
    test_out = np.expand_dims(test_out, axis=3)

    mat_contents.close()

    return train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step


def r2(y_true, y_pred):
    # r2_value = 1 - K.mean(K.sum(K.square(y_pred - y_true), axis=[1, 2])/K.sum(K.square(y_true - K.mean(y_true)), axis=[1, 2]))
    r2_value = 1 - K.sum(K.square(y_pred - y_true))/K.sum(K.square(y_true - K.mean(y_true)))
    return r2_value

def r2_np(y_true, y_pred):
    r2_value = 1 - np.sum(np.square(y_pred - y_true))/np.sum(np.square(y_true - np.mean(y_true)))
    return r2_value

class BiasLayer(Layer):
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zero',
                                    trainable=True)
        super(BiasLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


