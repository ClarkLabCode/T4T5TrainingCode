# T4T5TrainingCode
Code associated with training ANNs to predict natural scene velocities.

Use the following steps to run this code:

1) Download the natural scenes database from https://doi.org/10.4119/unibi/2689637 and extract into `generate_data_sets/pano_scenes`.

2) Run `NaturalScenesToContrastMeanImage.m` in `generate_data_sets/`

3) Run `GenerateNaturalScenesDataset.m` and `GenerateSinewaveDataset.m` in `generate_data_sets/` .  

4) Make sure you have `numpy`, `tensorflow`, `scipy`, and `h5py` installed

5) Run `train_multiple_models.py`

After training, you can run `analysis/ExampleAnalysis.m` to get a flavor of how to analyze the results.

For the paper, we trained thousands of models. This would take a very long time on a single machine. We used our university's research cluster, but we also experimented with using Google's AI Platform. We kept the code to make use of Google Cloud in `train_multiple_models.py`, but it is not well tested. In order to use Google Cloud, create a Google storage bucket and upload the `data_sets` folder after completing step 3. Enter the bucket name on line 15 in `train_multiple_models.py` and on line 43 in `train_keras_net.py`. You will also need to install `gsutil` on your machine. Then you can set `useGCloud` on line 11 of `train_multiple_models.py` to `True`.