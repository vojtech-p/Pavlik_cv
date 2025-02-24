
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import cv2 as cv
from DataPreprocessing import DataPreprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def feature_extract(image, num_blocks=12, num_blocks_vert=6):
    image_canny = cv.Canny(image, 50, 200, None, 3)
    features = []
    image_h = image.shape[0]
    image_w = image.shape[1]
    num_blocks_hor = num_blocks//num_blocks_vert
    block_h = image_h//num_blocks_vert
    block_w = image_w//num_blocks_hor

    for i in range(num_blocks_vert+1):
        for j in range(num_blocks_hor):
            block = image_canny[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.any():
                features.append(np.uint32(np.mean(block)))
            else:
                features.append(np.uint32(0))
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    features.append(len(keypoints))
    features.append(image_w)
    features.append(np.mean(image_canny))

    # Compute Hu Moments
    moments = cv.moments(image_canny)
    hu_moments = cv.HuMoments(moments).flatten()
    # Log scale transform to bring the values closer to each other
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)
    for i in hu_moments:
        features.append(i)
    """ 
    contour = image.squeeze()
    contour_complex = np.empty(contour.shape[0], dtype=complex)
    contour_complex.real = contour[:, 0]
    contour_complex.imag = contour[:, 1]

    # Compute Fourier Descriptors
    fourier_result = np.fft.fft(contour_complex)
    if fourier_result[1]:
        fourier_result = fourier_result / np.abs(fourier_result[1])

    # Take a fixed number of coefficients (truncated Fourier Descriptors)
    for i in np.abs(fourier_result[:10]):
        features.append(i)
    for n in range(47-len(features)):
        features.append(0)
    """
    return features


def train_test_model(X,y,n_estimator=50, n_features=24):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    rf = RandomForestClassifier(n_estimators=n_estimator, random_state=42)
    rf.fit(X_train, y_train)

    print(len(X_train[0]))
    rfe = RFE(estimator=rf, n_features_to_select=n_features)  # Choose how many features to keep
    rfe.fit(X_train, y_train)

    selected_features = np.where(rfe.support_ == True)[0]
    print("Selected Feature Indices: ", selected_features)

    # Transform your data to select the top features
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    #print(len(X_train_rfe[0]))

    param_grid = {
        'n_neighbors': [3,5,7,9,11,13,15,17],
        'weights': ['uniform', 'distance'],
       'metric': ['euclidean', 'manhattan']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train_rfe, y_train)

    #knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
    knn = grid.best_estimator_
    print(knn)

    knn.fit(X_train_rfe, y_train)
    y_pred = knn.predict(X_test_rfe)

    # Evaluate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Print classification report for detailed metrics
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_normalized_percentages = (cm / np.sum(cm, axis=0, keepdims=True)).ravel()
    group_normalized_percentages = ["{0:.2%}".format(value) for value in group_normalized_percentages]

    cell_labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_normalized_percentages)]
    cell_labels = np.asarray(cell_labels).reshape(10,10)
#    sns.heatmap(cm, annot=cell_labels, cmap="Blues", fmt="", ax=ax)
    # Step 5: Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.7)
    sns.heatmap(cm, annot=cell_labels, fmt='', cmap='Blues')
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion Matrix')
    plt.show()
    joblib.dump(knn, 'knn_model.pkl')



filePath = "trainData/"
refFile = pd.read_csv(f"{filePath}\\references.csv")
numRecords = refFile.shape[0]
X = []
y = []
for idx in range(numRecords):
    fileName = refFile.name[idx]
    inputData = cv.imread(f'{filePath}/{fileName}')

    targetClass = refFile.target[idx]
    if targetClass == 10:
        targetClass = 0
    y.append(targetClass)

    preprocessedData = DataPreprocessing(inputData)
        
    X.append(feature_extract(preprocessedData,18,6))


train_test_model(X,y,n_estimator=50,n_features=29)
