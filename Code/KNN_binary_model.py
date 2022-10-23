# Building a neural net for a binary outcome variable using keras
# Step 1: Import packages and download/create csv file with data
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense
binary_data = pd.read_csv('../Spotify_deep_dive/trainingsdata.csv', delimiter=',')

# transform into binary labels
binary_data['mood'] = binary_data['mood'].map({'happy': 1, 'sad': 0, 'energetic': 0, 'calm': 0})

# Step 2: Split data into training and test set
import numpy as np
labels = binary_data['outcome']
features = binary_data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
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

# Step 4: Pick activation function (ReLU, sigmoid, etc.) for each layer
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(11, activation='relu', input_shape=(11,)))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 5: Train model
model.compile(loss='binary_crossentropy',
optimizer='sgd',
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4, batch_size=1, verbose=1)

# Step 6: Run predictions on test data
y_pred = model.predict_classes(X_test)

# Step 7: Print score/accuracy:
score = model.evaluate(X_test, y_test,verbose=1)
print(score)

# Step 8: Saving the model
model.save('path/to/location')
# and loading it back
model = keras.models.load_model('path/to/location')

### Making predictions ###

# Step 1: Load df on which predictions should be made + cut it into right shape
data = pd.read_csv('../WiSe21-Project-Spotify_Deep_Dive/trainingsdata.csv', delimiter=',')
# Or instead of trainings data: '../WiSe21-Project-Spotify_Deep_Dive/03-Output/XX_merged.csv'
prediction_data = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']]
print(prediction_data)

# Step 2: Standardize data:
scaler = StandardScaler().fit(prediction_data)
prediction_data = scaler.transform(prediction_data)

# Step 3: Use model
predicted_labels = (model.predict(prediction_data) > 0.5).astype("int32")
print(predicted_labels)

# Step 4: Transform predictions into list and add to original df
data['outcome'] = predicted_labels.tolist()
print(data)
