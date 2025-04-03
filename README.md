## Setup

Download video data and real doppler data from Midas.

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
Set up the environment as follows:
```
conda create --name encoderdecoder
conda activate encoderdecoder
conda install python==3.8
conda install pip
which -a pip
PIPLOCATION install tensorflow==2.4.0
```

Run custom_train_encoder_decoder.py once for zzh and then again for dkk
Run batch_run_encoder_decoder.py twice, using autoencoderDKK.hdf5 to generated data for zzh and autoencoderZZH.hdf5 for dkk
You can visualise this data using the visualiseDopplerData.py script

## Diffusion model


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



