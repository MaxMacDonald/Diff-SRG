import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
import sys
import argparse
from helper import get_spectograms, root_mean_squared_error
from alive_progress import alive_bar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--input_folder', type=str, default='video_data/dkk')
    parser.add_argument('--model_path', type=str, default='save_weights/autoencoder.hdf5')
    return parser.parse_args()


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred*255) - (y_true*255))))


# Function to generate spectrograms (assuming it's the same as used in training)
def preprocess_data(data):
    data = np.concatenate([data, np.zeros((32, 1))], axis=1)
    doppler = get_spectograms(data.T, TIME_CHUNK, fps)
    doppler = np.expand_dims(doppler, axis=-1)  # Add channel dimension
    doppler = (doppler - np.min(doppler)) / (np.max(doppler) - np.min(doppler))  # Normalize
    return doppler.astype("float32")

args = parse_args()
model_path = args.model_path
autoencoder = load_model(model_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
decoder = autoencoder.get_layer("decoder")
modelinput = autoencoder.get_layer("input_1")
ada = Adam(lr=0.001)
autoencoder.compile(
    optimizer = ada,
    loss = root_mean_squared_error,
    metrics=['mae', 'acc'],
)
TIME_CHUNK = 3
fps = 24
input_folder = args.input_folder

with alive_bar(33, title="Processing videos") as bar:
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().startswith('synth'):
                    coarseDataPath = os.path.join(root, file)
                    directory_path = os.path.dirname(coarseDataPath)
                    coarseData = np.load(coarseDataPath)
                    # Preprocess test data
                    coarseData = preprocess_data(coarseData)

                    # Reshape to match the model input shape (batch_size, 32, 72, 1)
                    coarseData = coarseData.reshape((-1, 32, 72, 1))
                    predicted_output = autoencoder.predict(coarseData)
                    # If needed, rescale back to original values
                    predicted_output = predicted_output * (np.max(coarseData) - np.min(coarseData)) + np.min(coarseData)
                    predicted_output = np.squeeze(predicted_output)  # Shape (N, 32, 72)


                    # Initialize an empty array for reconstruction
                    reconstructed = np.zeros((32, 250))  # (250, 32)
                    frame_chunk = 72
                    frame_overlap = 1

                    # Reverse iteration: place back each extracted segment
                    for idx, spec in enumerate(predicted_output):
                        start = idx * frame_overlap
                        end = start + frame_chunk
                        reconstructed[:, start:end] = spec  # Directly placing it back
                    outputPath = os.path.join(directory_path, "reconstructed.npy")
                    np.save(outputPath,reconstructed)
                    bar()

