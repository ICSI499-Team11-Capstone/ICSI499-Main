import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from dna_featuregenerator import create_feature_vectors
from balanced_class import balanced_subsample

out = "predictions.csv"
sequences = ['training data/dark_green.txt',
             'training data/dark_red.txt',
             'training data/dark_vred.txt',
             'training data/dark_nir.txt',
             'training data/green_red.txt',
             'training data/green_vred.txt',
             'training data/green_nir.txt',
             'training data/red_vred.txt',
             'training data/red_nir.txt',
             'training data/vred_nir.txt']

# Train classifiers on each sequence file
classifiers = []
for sequence_file in sequences:
    features = create_feature_vectors(sequence_file)
    X = features.drop('color', axis=1)
    y = features['color']
    
    # Train 10 classifiers per sequence file
    for _ in range(10):
        xs, ys = balanced_subsample(X, y)
        svm = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, 
                       random_state=0, tol=1e-5, max_iter=1000000, C=0.1)
        svm.fit(xs, ys)
        clf = CalibratedClassifierCV(svm)
        clf.fit(xs, ys)
        classifiers.append(clf)
print("Training complete")

# Make predictions for each test file
classifier_idx = 0
for test_file in sequences:
    df = pd.DataFrame()
    
    # Load test sequences
    test_features = create_feature_vectors(test_file)
    test_X = test_features.drop('color', axis=1)
    
    # Get predictions from each classifier set
    for sequence_file in sequences:
        prob_class0 = []
        prob_class1 = []
        
        # Average predictions across 10 classifiers
        for _ in range(10):
            clf = classifiers[classifier_idx]
            predictions = clf.predict_proba(test_X)
            classifier_idx += 1
            
            if len(prob_class0) == 0:
                for class0_prob, class1_prob in predictions:
                    prob_class0.append(class0_prob)
                    prob_class1.append(class1_prob)
            else:
                temp_class0 = []
                temp_class1 = []
                for class0_prob, class1_prob in predictions:
                    temp_class0.append(class0_prob)
                    temp_class1.append(class1_prob)
                prob_class0 = np.add(temp_class0, prob_class0)
                prob_class1 = np.add(temp_class1, prob_class1)
        
        # Average the probabilities
        prob_class0 = prob_class0 / 10
        prob_class1 = prob_class1 / 10
        
        df[sequence_file] = prob_class0.tolist()
        df[sequence_file + "_class1"] = prob_class1.tolist()
    
    # Save predictions for this test file
    print(f"Saving predictions for {test_file}")
    df.to_csv(out, mode='a', header=False)
    
    # Reset classifier index for next test file
    classifier_idx = 0
