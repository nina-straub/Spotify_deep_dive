# Building a neural net for a multiple outcome variable using keras
# Step 1: Import packages and download/create csv file with data
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense
data = pd.read_csv('../Spotify_deep_dive/trainingsdata.csv', delimiter=',')

# transform into binary labels
data['mood'] = data['mood'].map({'happy': 1, 'sad': 0, 'energetic': 2, 'calm': 3})

# Step 2: Split data into training and test set
import numpy as np
labels = data['mood']
features = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']]
from sklearn.model_selection import train_test_split
X = features
y = np.ravel(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 3: z-standardize values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: built, compile and train model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

def base_model():
    model = Sequential()
    model.add(Dense(8,input_dim=11,activation='relu'))
    model.add(Dense(4,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=base_model,epochs=8, batch_size=10, verbose=1)

kfold = KFold(n_splits=10,shuffle=True)
results = cross_val_score(estimator,X,y,cv=kfold, error_score='raise')

estimator.fit(X_train,y_train)

# Step 5: Run predictions on test data:
y_preds = estimator.predict_proba(X_test)
discretePredictions = np.apply_along_axis(lambda arr : np.argmax(arr), 1, y_preds)

print(accuracy_score(y_test,discretePredictions))
cm = confusion_matrix(y_test,discretePredictions)


### Making predictions ###

# Step 1: Load df on which predictions should be made + cut it into right shape
new_data = pd.read_csv('../WiSe21-Project-Spotify_Deep_Dive/trainingsdata.csv', delimiter=',')
# Or instead of trainings data: '../WiSe21-Project-Spotify_Deep_Dive/03-Output/XX_merged.csv'
prediction_data = new_data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']]
print(prediction_data)

# Step 2: Standardize data:
scaler = StandardScaler().fit(prediction_data)
prediction_data = scaler.transform(prediction_data)

# Step 3: Use model
predicted_labels = (estimator.predict(prediction_data) > 0.5).astype("int32")
print(predicted_labels.tolist())

# Step 4: Transform predictions into list and add to original df
data['mood'] = predicted_labels.tolist()
print(data)
