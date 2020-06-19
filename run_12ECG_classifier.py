#!/usr/bin/env python

import numpy as np
import joblib
import pickle
from get_12ECG_features import get_12ECG_features, get_fft_peaks

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use the classifier here to obtain a label and score for each class. 
    features_t=np.asarray(get_12ECG_features(data,header_data)).astype(int)
    features_f=np.asarray(get_fft_peaks(data, header_data)).astype(int)
    #print(type(features_t),features_t)
    #print(type(features_f),features_f)

    features = np.concatenate([features_t,features_f])
    feats_reshape = features.reshape(1,-1)
    label = model.predict(feats_reshape)
    
    score = model.predict_proba(feats_reshape)

    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
    filename='classifier.model'
    loaded_model = pickle.load(open(filename,"rb"))
    
    return loaded_model
