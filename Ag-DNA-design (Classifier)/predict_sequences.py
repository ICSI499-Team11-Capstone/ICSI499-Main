import csv
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from dna_featuregenerator import create_feature_vectors
from balanced_class import balanced_subsample
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from itertools import product

test_sequences = ["file_name"]
sequences = ['dark_green.txt',
             'dark_red.txt',
             'dark_vred.txt',
             'dark_nir.txt',
             'green_red.txt',
             'green_vred.txt',
             'green_nir.txt',
             'red_vred.txt',
             'red_nir.txt',
             'vred_nir.txt']

classifiers=[]
df = pd.DataFrame()
for i in sequences:
    features=create_feature_vectors(i)
    for j in range(10):
        x1 = []
        y1 = []
        X = features.drop('color', axis=1)
        y = features['color']
        xs,ys=balanced_subsample(X,y)
        svm = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,random_state=0, tol=1e-5,max_iter=1000000, C=.1)
        svm.fit(xs,ys)
        clf = CalibratedClassifierCV(svm)
        clf.fit(xs,ys)
        classifiers.append(clf)
print("done")

k=0
output_file = "predictions.csv"

for i in sequences:
    seq = create_feature_vectors(test_sequences)
    seq = seq.drop('color', axis=1)
    
    # Initialize x2 and y2 with first prediction
    b = classifiers[k]
    a = b.predict_proba(seq)
    k += 1
    x2 = np.array([l for l, m in a])
    y2 = np.array([m for l, m in a])
    
    # Add remaining predictions
    for j in range(1, 10):
        b = classifiers[k]
        a = b.predict_proba(seq)
        k += 1
        x1 = np.array([l for l, m in a])
        y1 = np.array([m for l, m in a])
        x2 = np.add(x1, x2)
        y2 = np.add(y1, y2)
    
    # Average the predictions
    x2 = x2 / 10
    y2 = y2 / 10
    
    # Store in dataframe
    df[i] = x2.tolist()
    df[i+"2"] = y2.tolist()

df.to_csv(output_file, mode='a', header=False)
