## Setup

Download video data and real doppler data from Midas.
Install Midas from: https://github.com/dkkmy/Midas

We recommend setting up each step in a separate conda environment

## Human Region Indexing
Install dependencies

Run YoloToAdaBins.py


## Depth Prediction
Install AdaBins from: https://github.com/shariqfarooq123/AdaBins

## Human Mesh Fitting

Install OSX from: https://github.com/IDEA-Research/OSX

Move OSXBatchAll.py to OSX/main

Run OSXBatchAll.py with the following command:
```python OSXBatchFitting.py --gpu 0 --input_folder {folder containing output of yoloToAda} --output_folder {folder of choice}```

## Human Reflection Model
Run generateCourseDopplerDataBatch.py

## Fine Grain Translation

### UNet Auto-encoder
Set up the environment as follows:
```
conda create --name encoderdecoder
conda activate encoderdecoder
conda install python==3.8
conda install pip
which -a pip
PIPLOCATION install tensorflow==2.4.0
```

Run custom_train_encoder_decoder.py once for zzh and then again for dkk.

Run batch_run_encoder_decoder.py twice, using autoencoderDKK.hdf5 to generated data for zzh and autoencoderZZH.hdf5 for dkk.

You can visualise this data using the visualiseDopplerData.py script.

### Transformer Autoencoder

Run The TransformerAutoencoderTraining.py file to train the model. As with the Unet auto-encoder remember to change subject for leave one out. You will have to change file paths.

Run TransformerAutoencoderPredict.py to generate fine grain data, again in a leave one out fashion.

inspectModel.py is included to allow inspection of the architecture.

you can visualise this data using the visualiseDopplerDataDiffusion.py script.

### Diffusion model
Install Radar-Diffusion from: https://github.com/ZJU-FAST-Lab/Radar-Diffusion

Replace dist_util and train_util_cond inside the _Radar-Diffusion-custom/diffusion_consistency_radar/cm_ directory with the ones included in this project. They contain modifications that allow for training with the doppler data.

Move the files in launch into _Radar-Diffusion-custom/diffusion_consistency_radar/launch_. These are used to launch the scripts. You will have to modify the path at the start of each to match your system.

Move the files in scripts into _Radar-Diffusion-custom/diffusion_consistency_radar/scripts_


## Evaluation
Calculate MAE and SD using calculateMAEandSD.py
```
python calculateMAEandSD.py --input_folder /home/max/Results/generatedData_LOO --real_data_folder /home/max/mastersProject/Midas/doppler_data/radar_data
```
Install CMMD from: https://github.com/google-research/google-research/tree/master/cmmd

To calculate cmmd first convert the visualised doppler signals to jpg using convertToJPG.py

Then run with:
```
python -m cmmd.main /home/max/Results/dopplerVisualisationLOO/real_images_jpg /home/max/Results/dopplerVisualisationLOO/reconstructed_images_jpg
```
To calculate classification accuracy move train_vgg16_classifier_custom.py and train_vgg16_classifier_diffusion_custom.py into the folder you installed Midas then edit the data_path and real_data_path. 
Remember to change the subject for leave one out validation by changing the value on line 54/55.

## License
This work is published under the MIT license

