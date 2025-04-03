import numpy as np
from tensorflow.keras.models import load_model
from helper import get_spectograms, root_mean_squared_error
from tensorflow.keras.optimizers import Adam
import os
import sys
import numpy
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid memory issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Use only the first GPU (optional)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU is being used:", gpus[0])
    except RuntimeError as e:
        print(e)

numpy.set_printoptions(threshold=sys.maxsize)

model_path = "models/"
autoencoder = load_model(model_path+"autoencoder_weights.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

ada = Adam(lr=0.001)
autoencoder.compile(
    optimizer = ada,
    loss = root_mean_squared_error,
    metrics=['mae', 'acc'],
)

TIME_CHUNK = 3
fps = 24

# Modify these to fit
synth_dir = "/home/max/Results/intensitymapping"
gt_dir = "/home/max/mastersProject/Midas/doppler_data/radar_data"

person = ['zzh']
angle = ['0', '45', '90']

gt_list = []
synth_list = []

for p in person:
    for a in angle:
        activity = sorted(os.listdir(os.path.join(synth_dir, p, a)))
        for active in activity:
            real_data = np.load(os.path.join(gt_dir, p, a, active, "doppler_gt.npy"))
            synth_data = np.load(os.path.join(synth_dir, p, a, active, "synthDopplerData.npy"))
            # Add an extra column of zeros
            synth_data = np.concatenate([synth_data, np.zeros((32, 1))], axis=1)

        
            print(os.path.join(synth_dir, p, a, active, "synthDopplerData.npy"))
            print(synth_data.T.shape)
            dopler1 = get_spectograms(real_data.T, TIME_CHUNK, fps)
            dopler1 = np.expand_dims(dopler1, axis=-1)
            dopler1 = (dopler1 - np.min(dopler1)) / (np.max(dopler1) - np.min(dopler1))
            dopler1 = dopler1.astype("float32")
            dopler2 = get_spectograms(synth_data.T, TIME_CHUNK, fps)
            print("inital dopler2 shape: ", dopler2.shape)
            dopler2 = np.expand_dims(dopler2, axis=-1)
            dopler2 = (dopler2 - np.min(dopler2)) / (np.max(dopler2) - np.min(dopler2))
            dopler2 = dopler2.astype("float32")
            print("final dopler2 shape: ", dopler2.shape)
            gt_list.append(dopler1)
            synth_list.append(dopler2)

print(len(gt_list))
print(len(synth_list))
gt_list = np.array(gt_list).reshape((-1,32,72,1))
synth_list = np.array(synth_list).reshape((-1,32,72,1))
print(gt_list.shape)
print(len(gt_list))
print(len(synth_list))
autoencoder.fit(synth_list, gt_list, batch_size=128, epochs=1000, shuffle=True)
autoencoder.save("save_weights/autoencoderZZH.hdf5")
