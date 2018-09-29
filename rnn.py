# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Part 1 - Setting up

# setting timestep for LSTM
timesteps = 60

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
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

### Part 3 - Making the predictions and visualising the results

# Getting the real stock price
dataset_test = pd.read_csv('test.csv')
test_set = dataset_test.iloc[:,5:6].values
real_stock_price = np.concatenate((training_set[0:len_training_set], test_set), axis = 0)

# Getting the predicted stock price of
scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(len_training_set, len_training_set + len(test_set)):
    inputs.append(scaled_real_stock_price[i-timesteps:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[len_training_set:], color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()