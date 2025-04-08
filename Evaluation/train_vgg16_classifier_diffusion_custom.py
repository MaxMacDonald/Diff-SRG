import os
import numpy as np
from helper import get_spectograms
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as np_utils
from sklearn.metrics import confusion_matrix

def main():
    data_path = "/home/max/Results/generatedData_Diffusion"
    real_data_path = "/home/max/Results/realData"
    person = ['dkk', 'zzh']
    angle = ['0', '45', '90']

    model = load_model("models/classifier_weights.hdf5")
    TIME_CHUNK = 24
    fps = 3

    ada = Adam(lr=0.001)
    model.compile(
        optimizer = ada,
        loss = 'categorical_crossentropy',
        metrics=['accuracy'],
    )

    train_x, train_y, val_x, val_y = [], [], [], []

    for p in person:
        for a in angle:
            path = os.path.join(data_path, p, a)
            activities = sorted(os.listdir(path))
            for active in activities:
                    class_name = int(active[0:2])-1
                    dop_file = os.path.join(path, active, "fine_reconstructed.npy")
                    dopler = np.load(dop_file).T
                    dopler = get_spectograms(dopler, TIME_CHUNK, fps)
                    dopler = np.expand_dims(dopler[0:178], axis=-1)
                    class_arr = np.array([class_name] * dopler.shape[0])
                    dopler = (dopler - np.min(dopler))/(np.max(dopler) - np.min(dopler))
                    train_x.append(dopler)
                    train_y.append(class_arr)
                    
    for p in person:
        for a in angle:
            path = os.path.join(real_data_path, p, a)
            activities = sorted(os.listdir(path))
            for active in activities:
                    class_name = int(active[0:2])-1
                    dop_file = os.path.join(path, active, "doppler_gt.npy")
                    dopler = np.load(dop_file).T
                    dopler = get_spectograms(dopler, TIME_CHUNK, fps)
                    dopler = np.expand_dims(dopler[0:178], axis=-1)
                    class_arr = np.array([class_name] * dopler.shape[0])
                    dopler = (dopler - np.min(dopler))/(np.max(dopler) - np.min(dopler))
                    if p == 'zzh':
                        val_x.append(dopler)
                        val_y.append(class_arr)
                    else:
                        train_x.append(dopler)
                        train_y.append(class_arr)

    train_X = train_x[0]
    train_Y = train_y[0]
    val_X = val_x[0]
    val_Y = val_y[0]
    for i in range(1, len(train_x)):
        train_X = np.concatenate((train_X,train_x[i]), axis=0)
        train_Y = np.concatenate((train_Y,train_y[i]), axis=0)
    for i in range(1, len(val_x)):
        val_X = np.concatenate((val_X,val_x[i]), axis=0)
        val_Y = np.concatenate((val_Y,val_y[i]), axis=0)

    train_X = train_X.reshape((-1, 32, 72, 1))
    train_Y = np_utils.to_categorical(train_Y, num_classes=12)
    train_Y = train_Y.reshape((-1, 12))


    val_X = val_X.reshape((-1, 32, 72, 1))
    val_Y = np_utils.to_categorical(val_Y, num_classes=12)
    val_Y = val_Y.reshape((-1, 12))


    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)

    print(f"train_X:{len(train_X)}  train_Y:{len(train_Y)}  val_X:{len(val_X)}  val_Y:{len(val_Y)}")

    model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=64, epochs=100)
    model.save("save_weights/vgg16_classifier.hdf5")

    proba = model.predict(val_X)
    Y_pred = np.argmax(proba, axis=-1)
    Y_gt = np.argmax(val_Y, axis=-1)

    cm = confusion_matrix(np.array(Y_gt), np.array(Y_pred))
    cm = cm / np.expand_dims(np.sum(cm, axis=1), axis = 0) * 100
    print(np.around(cm, 2))
    print(np.mean(np.diagonal(cm)))

    import pandas as pd
    df = pd.DataFrame(np.around(cm/1.0, 2))
    df.to_csv('data.csv')

if __name__ == '__main__':
        main()
