import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
from helper import get_spectograms, root_mean_squared_error
#import matplotlib.pyplot as plt


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square((y_pred*255) - (y_true*255))))



#model_path = '/home/max/mastersProject/Midas/models/autoencoder_weights.hdf5'
model_path = '/home/max/mastersProject/Midas/save_weights/autoencoder.hdf5'
#doppler_test = np.load('/home/max/mastersProject/Midas/doppler_data/radar_data/dkk/0/01/doppler_gt.npy')
doppler_test = np.load('/home/max/mastersProject/MastersProject/synthDopplerData.npy')
print(doppler_test)
autoencoder = load_model(model_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
print("-------------------------------------------------------")
print(autoencoder.summary())
print("-------------------------------------------------------")
decoder = autoencoder.get_layer("decoder")
modelinput = autoencoder.get_layer("input_1")
#print(modelinput.summary())
print(decoder.summary())
ada = Adam(lr=0.001)
autoencoder.compile(
    optimizer = ada,
    loss = root_mean_squared_error,
    metrics=['mae', 'acc'],
)
TIME_CHUNK = 3
fps = 24

# Function to generate spectrograms (assuming it's the same as used in training)
def preprocess_data(data):
    data = np.concatenate([data, np.zeros((32, 1))], axis=1)
    doppler = get_spectograms(data.T, TIME_CHUNK, fps)
    doppler = np.expand_dims(doppler, axis=-1)  # Add channel dimension
    doppler = (doppler - np.min(doppler)) / (np.max(doppler) - np.min(doppler))  # Normalize
    return doppler.astype("float32")

# Preprocess test data
test_input = preprocess_data(doppler_test)

# Reshape to match the model input shape (batch_size, 32, 72, 1)
test_input = test_input.reshape((-1, 32, 72, 1))
print(test_input.shape)
predicted_output = autoencoder.predict(test_input)

# If needed, rescale back to original values
predicted_output = predicted_output * (np.max(doppler_test) - np.min(doppler_test)) + np.min(doppler_test)
print(predicted_output.shape)


# Remove the last dimension (channel)

predicted_output = np.squeeze(predicted_output)  # Shape (N, 32, 72)
print(predicted_output.shape)
np.save("rawPredictedOutput.npy",predicted_output)
# Reshape back to the original time format (approximate reconstruction)
reconstructed_data = predicted_output.reshape(32, -1)  # Flatten across time windows

# If needed, trim or pad to match the exact original shape
target_shape = (32, 250)
if reconstructed_data.shape[1] > target_shape[1]:
    reconstructed_data = reconstructed_data[:, :target_shape[1]]  # Trim extra columns
elif reconstructed_data.shape[1] < target_shape[1]:
    padding = target_shape[1] - reconstructed_data.shape[1]
    reconstructed_data = np.pad(reconstructed_data, ((0, 0), (0, padding)), mode='constant')  # Zero-pad

print("Reconstructed shape:", reconstructed_data.shape)  # Should be (32, 250)



def display(values):
    print(f"Shape of input values is {np.shape(values)}")
    # Step 2: Reshape to (32, 250)
    doppler_data = np.array(values).T

    # Create an intensity map for the Doppler data
    plt.figure(figsize=(10, 6))
    plt.imshow(doppler_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Doppler Intensity Map')
    plt.xlabel('Time/Frequency Bins')
    plt.ylabel('Radial Velocity (m/s)')

    num_channels = doppler_data.shape[0]  # Number of sensor channels
    y_ticks = np.linspace(0, num_channels - 1, 5)  # Select 5 tick positions
    y_labels = np.linspace(-2, 2, 5)  # Corresponding velocity values (-2 to +2)
    plt.yticks(y_ticks, [f"{label:.1f} m/s" for label in y_labels])  # Set labels

    plt.tight_layout()
    plt.show()
    plt.savefig("reconstructedPlot.png")
    np.save("reconstructedDopplerData.npy",doppler_data)

np.save("reconstructedDopplerData.npy",reconstructed_data)
