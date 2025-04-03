import os
import numpy as np
from helper import get_spectograms
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import pickle
import argparse

def main(args):

    data_path = args.data_path
    model_path = args.model_path

    classifier = load_model(model_path+"classifier_weights.hdf5")
    lb = pickle.loads(open(model_path+"classifier_classes.lbl", "rb").read())
    print("#########################################")
    print(lb.classes_)
    print("#########################################")
    scale_vals = np.load(model_path+"scale_vals.npy")
    classes = ['01','02','03','04','05','06','07','08','09','10','11']
    class_mapping = {
    '01': 'Clapping',
    '02': 'Cleaning',
    '03': 'JumpRope',
    '04': 'Jumping',
    '05': 'JumpingJack',
    '06': 'Lunge',
    '07': 'Running',
    '08': 'Squat',
    '09': 'Walking',
    '10': 'WalkingUpSteps',
    '11': 'Waving'
    }
    participants = ['dkk','zzh']
    angles = ['0','45','90']
    fps = 24
    TIME_CHUNK = 3
    X_test, Y_test = [], []
    max_dopVal = scale_vals[0]
    max_synth_dopVal =  scale_vals[1]
    min_dopVal = scale_vals[2]
    min_synth_dopVal = scale_vals[3]
    for p in participants:
        for a in angles:
            activity = sorted(os.listdir(os.path.join(data_path, p, a)))
            for active in activity:
                activityPath = os.path.join(data_path, p, a, active, "doppler_gt.npy")
                real_data = np.load(activityPath)
                splitPath = activityPath.split('/')[-4:-1]
                #print(splitPath[-1].split('_')[0])
                class_name = class_mapping[splitPath[-1].split("_")[0]]
                #print(class_name)
                dopler1 = get_spectograms(real_data.T, TIME_CHUNK, fps)
                class_arr = np.array([class_name] * dopler1.shape[0])
                dopler1 = (dopler1 - np.min(dopler1)) / (np.max(dopler1) - np.min(dopler1))
                dopler1 = dopler1.astype("float32")

                X_test.append(dopler1)
                Y_test.append(class_arr)


    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    Y_pred, Y_gt = [], []

    for i in range(X_test.shape[0]):
        X_in = np.expand_dims(X_test[i], axis=-1)
        class_lbl = Y_test[i][0]
        proba = np.mean(classifier.predict(X_in),axis=0)
        idx = np.argmax(proba)
        label = lb.classes_[idx]
        Y_pred.append(label)
        Y_gt.append(class_lbl)
    for pred, gt in zip(Y_pred, Y_gt):
        print(f"Predicted: {pred}, Ground Truth: {gt}")

    acc = np.round(100*accuracy_score(Y_gt, Y_pred),2)
    print("Train on Synthetic Only & Test on Real World Doppler - Accuracy:",acc,"%")

if __name__ == '__main__':

        parser = argparse.ArgumentParser()

        parser.add_argument('--data_path', type=str, help='Path to data')

        parser.add_argument('--model_path', type=str, help='Path to DL models')

        args = parser.parse_args()

        main(args)

