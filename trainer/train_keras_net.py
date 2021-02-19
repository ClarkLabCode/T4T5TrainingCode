from enum import Enum, auto
import argparse

class RunLocation(Enum):
    LOCAL = auto()
    GClOUD = auto()
    GCLOUD_DOCKER = auto()

    def __str__(self):
        return self.name

# Arguments: run_name, worker_id, 
parser = argparse.ArgumentParser()
parser.add_argument("--run_name",default='default')
parser.add_argument("--worker_id",default='0')
parser.add_argument("--job-dir")
parser.add_argument("--run_location", type=lambda loc: RunLocation[loc], choices=list(RunLocation))
args = parser.parse_args()

run_name = args.run_name
worker_id = args.worker_id
run_loc = args.run_location

if run_loc is RunLocation.GClOUD:
    from . import model_keras as md
else:
    import model_keras as md

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
import scipy.io as sio
import datetime
import os
import copy
import pickle


# define the input path
# data set location
gcloud_bucket_name = 'YOUR BUCKET NAME HERE'
gcloud_data_set_storage = 'gs://' + gcloud_bucket_name + '/data_sets'
gcloud_run_group_storage = 'gs://' + gcloud_bucket_name + '/run_outputs/' + run_name

local_scratch_path = 'staging'
local_training_data_path = 'data_sets'
local_output_path = 'outputs'

# load params
if run_loc in (RunLocation.GClOUD, RunLocation.GCLOUD_DOCKER):
    tf.io.gfile.copy(gcloud_run_group_storage + '/' + 'param_dict_' + worker_id + '.p', 'param_dict.p')
    param_dict = pickle.load(open( "param_dict.p", "rb" ))
else:
    param_dict = pickle.load(open(local_scratch_path + '/' +'param_dict_' + worker_id + '.p', "rb"))

#Copy data sets
data_set_names = ['xtPlot_ns241_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-25.mat',
                  'xtPlot_sineWaves_sl20_ll90_pe360_ps5_sf100_tt1_nt6080_hl0-2_vs100_df0-05.mat']

if run_loc in (RunLocation.GClOUD, RunLocation.GCLOUD_DOCKER):
    data_set_folder = 'data_sets'
    os.makedirs(data_set_folder)
    for name in data_set_names:
        tf.io.gfile.copy(gcloud_data_set_storage + '/' + name, data_set_folder + '/' + name)
else:
    data_set_folder = local_training_data_path

image_types = ['nat', 'sine']

# load in data set
all_train_ins = [0 for i in range(len(image_types))]
all_train_outs = [[0 for i in range(2)] for j in range(len(image_types))]
all_dev_ins = [0 for i in range(len(image_types))]
all_dev_outs = [[0 for i in range(2)] for j in range(len(image_types))]
all_test_ins = [0 for i in range(len(image_types))]
all_test_outs = [[0 for i in range(2)] for j in range(len(image_types))]
sample_freqs = [0 for i in range(len(image_types))]
phase_steps = [0 for i in range(len(image_types))]
for image_type_idx in range(len(image_types)):
    path = data_set_folder + '/' + data_set_names[image_type_idx]

    all_train_ins[image_type_idx], all_train_outs[image_type_idx],\
        all_dev_ins[image_type_idx], all_dev_outs[image_type_idx],\
        all_test_ins[image_type_idx], all_test_outs[image_type_idx],\
        sample_freqs[image_type_idx], phase_steps[image_type_idx] = md.load_data_rr(path)

# Detect hardware
try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None
  gpus = tf.config.experimental.list_logical_devices("GPU")
    
# Select appropriate distribution strategy
if tpu:
  tf.tpu.experimental.initialize_tpu_system(tpu)
  strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
elif len(gpus) > 1:
  strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
  print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
elif len(gpus) == 1:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on single GPU ', gpus[0].name)
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
  sess = tf.compat.v1.Session(config=config)
  tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
else:
  strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on CPU')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

total_runs = len(param_dict)
# fit all the models
train_begin = time.time()
for p_ind, p in enumerate(param_dict):
    tf.keras.backend.clear_session()
    run_begin = time.time()

    image_type_idx = image_types.index(p['image_type'])

    train_in_full = all_train_ins[image_type_idx]
    dev_in_full = all_dev_ins[image_type_idx]
    test_in_full = all_test_ins[image_type_idx]

    train_out_full = all_train_outs[image_type_idx]
    dev_out_full = all_dev_outs[image_type_idx]
    test_out_full = all_test_outs[image_type_idx]

    p['sample_freq'] = sample_freqs[image_type_idx]
    p['phase_step'] = phase_steps[image_type_idx]

    m, size_t, size_x, n_c = train_in_full.shape
    filter_indicies_t = int(np.ceil(p['filter_time']*p['sample_freq']))
    filter_indicies_x = int(np.ceil(p['filter_space']/p['phase_step']))

    
    # perform modifications to the data and answer sets
    train_in = train_in_full.astype(np.float32)
    dev_in = dev_in_full.astype(np.float32)
    test_in = test_in_full.astype(np.float32)

    train_out = train_out_full.astype(np.float32)
    dev_out = dev_out_full.astype(np.float32)
    test_out = test_out_full.astype(np.float32)

    # normalize images
    train_in = (train_in_full - np.mean(train_in_full, axis=(1,2), keepdims=True)) / np.std(train_in_full, axis=(1, 2), keepdims=True)
    dev_in = (dev_in_full - np.mean(dev_in_full, axis=(1,2), keepdims=True)) / np.std(dev_in_full, axis=(1, 2), keepdims=True)
    test_in = (test_in_full - np.mean(test_in_full, axis=(1,2), keepdims=True)) / np.std(test_in_full, axis=(1, 2), keepdims=True)

    # Add noise if asked
    train_in = train_in + np.random.randn(train_in.shape[0], train_in.shape[1], train_in.shape[2], train_in.shape[3]) * p['input_noise_std']
    dev_in = dev_in + np.random.randn(dev_in.shape[0], dev_in.shape[1], dev_in.shape[2], train_in.shape[3]) * p['input_noise_std']
    test_in = test_in + np.random.randn(test_in.shape[0], test_in.shape[1], test_in.shape[2], train_in.shape[3]) * p['input_noise_std']

    # format output for categorical
    if p['predict_direction']:
        train_out = (train_out > 0).astype(np.float32)
        dev_out = (dev_out > 0).astype(np.float32)
        test_out = (test_out > 0).astype(np.float32)

    # set up the model and fit it
    num_batches_per_epoch = train_in.shape[0] / p['batch_size']
    lr_decay = p['learning_rate_decay']/(num_batches_per_epoch*p['epochs'])
    adamOpt = optimizers.Adam(lr=p['learning_rate'], decay=lr_decay)

    if run_loc in (RunLocation.GClOUD, RunLocation.GCLOUD_DOCKER):
        log_dir = gcloud_run_group_storage + '/tensorboard_logs_' + worker_id
    else:
        log_dir = "logs/profile/log1"
    tensorboard_callback = TensorBoard(log_dir=log_dir, profile_batch=2)

    method_to_call = getattr(md, p['model_function_name'])

    with strategy.scope():
        model, pad_x, pad_t = method_to_call(input_shape=(size_t, size_x, n_c),
                                            filter_shape=[filter_indicies_t, filter_indicies_x],
                                            num_filter=p['num_filt'],
                                            output_noise_std=p['output_noise_std'],
                                            predict_direction=p['predict_direction'])
    
        if p['predict_direction']:
            model.compile(optimizer=adamOpt, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=adamOpt, loss='mean_squared_error', metrics=[md.r2])
    
    # format y data to fit with output
    train_out = np.tile(train_out, (1, 1, size_x-pad_x, 1))
    dev_out = np.tile(dev_out, (1, 1, size_x-pad_x, 1))
    test_out = np.tile(test_out, (1, 1, size_x-pad_x, 1))

    train_out = train_out[:, pad_t:, :, :]
    dev_out = dev_out[:, pad_t:, :, :]
    test_out = test_out[:, pad_t:, :, :]


    hist = model.fit(train_in, train_out, verbose=2, epochs=p['epochs'], batch_size=p['batch_size'], validation_data=(dev_in, dev_out))

    param_dict[p_ind]['model_name'] = model.name

    # grab the loss and R2 over time
    param_dict[p_ind]['loss'] = hist.history['loss']
    param_dict[p_ind]['val_loss'] = hist.history['val_loss']

    if p['predict_direction']:
        param_dict[p_ind]['accuracy'] = hist.history['accuracy']
        param_dict[p_ind]['val_accuracy'] = hist.history['val_accuracy']
    else:
        param_dict[p_ind]['r2'] = hist.history['r2']
        param_dict[p_ind]['val_r2'] = hist.history['val_r2']

    # extract the model weights
    weights = []
    biases = []

    for l in model.layers:
        all_weights = l.get_weights()

        if len(all_weights) > 0:
            weights.append(all_weights[0])
            if len(all_weights) == 2:
                biases.append(all_weights[1])
            else:
                biases.append([])

    param_dict[p_ind]['weights'] = weights
    param_dict[p_ind]['biases'] = biases

    run_end = time.time() - run_begin
    print('run ' + str(p_ind+1) + '/' + str(total_runs) + ' took ' + str(run_end/60) + ' minutes to train')
    average_run_time = (time.time() - train_begin)/(p_ind+1)
    print('aprox ' + str((total_runs-(p_ind+1))*average_run_time/60) + ' minutes remaining')

# Output results
output_filename = local_output_path + '/' + 'output' + worker_id + '.mat'

sio.savemat(output_filename, {'param_dict': param_dict})

if run_loc in (RunLocation.GClOUD, RunLocation.GCLOUD_DOCKER):
    tf.io.gfile.copy(output_filename,gcloud_run_group_storage + '/' + output_filename)
