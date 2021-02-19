import numpy as np
import pickle
import copy
import os
import datetime
import re
import math

useTPUs = False
useDocker = False
useGCloud = False

# If you want to use GCloud to train the networks, you'll need a storage bucket with the training data in a /data_sets/ folder
# If you want to use a docker image instead of Google AI platform, then set your image URI here.
image_uri = "gcr.io/YOUR_DOCKER_IMAGE_URI"
bucket_name = "YOUR_BUCKET_NAME"

# parameters of filters
# The full run would take a very long time on a single machine. A fast run lets you get some results more quickly
fast_run = True
if fast_run:
    num_workers = 1
    num_runs = 10
    filter_time = 0.3  # s
    filter_space = 15  # degrees
    batch_size = np.power(2, 7)
    learning_rate = 0.03
    learning_rate_decay = 10 # final learning rate will be lr * 1/(1 + decay)
    num_epochs = 100
    input_noise_std_list = [1]
    num_filt_list = [4]
    output_noise_std_list = [1]
    predict_direction_list = [False]
    image_type_idxes = [0]
    models_list = ['ln']
else:
    num_workers = 10
    num_runs =  50
    filter_time = 0.3  # s
    filter_space = 15  # degrees
    batch_size = np.power(2, 7)
    learning_rate = 0.03
    learning_rate_decay = 10 # final learning rate will be lr * 1/(1 + decay)
    num_epochs = 1000
    input_noise_std_list = [0, 0.125, 0.25, 0.5, 1]
    num_filt_list = [4]
    predict_direction_list = [False]
    output_noise_std_list = [0, 0.125, 0.25, 0.5, 1]
    image_type_idxes = [0, 1]
    models_list = ['ln', 'lnln', 'conductance']

image_types = ['nat', 'sine']
model_function_name_list = [models_list[i] + '_model_flip' for i in range(len(models_list))]
# generate a list of parameters to perform runs on
param_dict = []


for run_number in range(num_runs):
    for num_filt in num_filt_list:
        for input_noise_std in input_noise_std_list:
            for model_function_name in model_function_name_list:
                for image_type_idx in image_type_idxes:
                    for output_noise_std in output_noise_std_list:
                        for predict_direction in predict_direction_list:
                            param_dict.append({
                                'num_filt': num_filt,
                                'filter_space': filter_space,
                                'filter_time': filter_time,
                                'learning_rate': learning_rate,
                                'learning_rate_decay': learning_rate_decay,
                                'epochs': num_epochs,
                                'batch_size': batch_size,
                                'input_noise_std': input_noise_std,
                                'model_function_name': model_function_name,
                                'output_noise_std': output_noise_std,
                                'predict_direction': predict_direction,
                                'image_type': image_types[image_type_idx], 
                                })

num_runs = len(param_dict)
param_dict_split = np.array_split(param_dict, num_workers)

for filename in os.listdir('staging'):
    if filename.endswith('.p'):
        os.unlink('staging/' + filename)

num_digits = math.ceil(math.log10(num_workers))
for i in range(num_workers):
    pickle.dump(param_dict_split[i], open( f"staging/param_dict_{i:0{num_digits}}.p", "wb" ),protocol=2)

date_str = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

if useGCloud:
    region = "us-east1" if not useTPUs else "us-central1"

    run_name = 'train_flyml_' + date_str

    job_dir = f"gs://{bucket_name}/training_run_dir"

    
    if useDocker:
        os.system(f"docker build -f Dockerfile -t {image_uri} ./")
        os.system(f"docker push {image_uri}")

    for i in range(num_workers):
        copy_str = f"gsutil cp staging/param_dict_{i:0{num_digits}}.p gs://{bucket_name}/run_outputs/{run_name}/param_dict_{i:0{num_digits}}.p"
        print(copy_str)
        os.system(copy_str)

        job_name = run_name + "_" + f"{i:0{num_digits}}"
        stream_str = '--stream-logs' if fast_run else ''
        if useTPUs:
            machine_config_str = '--scale-tier BASIC_TPU'
        else:
            machine_config_str = '--scale-tier custom --master-machine-type n1-standard-4 --master-accelerator count=1,type=NVIDIA_TESLA_T4'

        if useDocker:
            run_image_str = f" --master-image-uri {image_uri}"
            run_loc_str = "GCOULD_DOCKER"
        else:
            run_image_str = f" --package-path trainer/ --module-name trainer.train_keras_net"\
                            f" --job-dir {job_dir} --runtime-version 1.15"
            run_loc_str = "GCLOUD"
        
        run_str = f"gcloud ai-platform jobs submit training {job_name}"\
                    f"{run_image_str}"\
                    f" --region {region} {machine_config_str} {stream_str} "\
                    f" -- --run_name {run_name} --worker_id {i:0{num_digits}} --run_location {run_loc_str}"
        
        print(run_str)
        os.system(run_str)
else: #local run
    for i in range(num_workers):
        os.system(f"python3 trainer/train_keras_net.py --worker_id {i:0{num_digits}} --run_location LOCAL")