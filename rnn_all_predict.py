# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

### Part 1 - Setting up

# setting timestep for LSTM
timesteps = 20

# Importing the training set
dataset_train = pd.read_csv('train.csv')
training_set = dataset_train.iloc[:,5:6].values

len_training_set = len(training_set) #length of training set

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with n timesteps and t+1 output
X_train = []
y_train = []
for i in range(timesteps, len_training_set):
	X_train.append(training_set_scaled[i-timesteps:i, 0])
	y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

### Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# Initialising the RNN
regressor = Sequential()

regressor.add(LSTM(units = 8, input_shape = (X_train.shape[1], 1)))

# Adding the output layer
regressor.add(Dense(units = 1, activation='sigmoid'))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 250, batch_size = 32)

### Part 3 - Making the predictions and visualising the results

# Getting the real stock price
dataset_test = pd.read_csv('test.csv')
test_set = dataset_test.iloc[:,5:6].values

#Prediction right now until 30 days later
init = [training_set_scaled[len_training_set - timesteps:len_training_set,0]]
predicted_stock_price = []

for i in range(0,10):
	inputs_np = np.array(init)
	inputs_np = np.reshape(inputs_np, (inputs_np.shape[0], inputs_np.shape[1], 1))
	predicted_stock_price_scaled = regressor.predict(inputs_np)

	temp_array = training_set_scaled[len_training_set - timesteps:len_training_set,0]
	temp_array = np.delete(temp_array,[0])
	temp_array = np.append(temp_array,predicted_stock_price_scaled[0])

	init = [temp_array]
	predicted_stock_price_scaled = sc.inverse_transform(predicted_stock_price_scaled)[0][0]
	pprint(predicted_stock_price_scaled)
	predicted_stock_price.append(predicted_stock_price_scaled)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()