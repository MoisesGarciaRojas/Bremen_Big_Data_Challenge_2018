########################################################################################################################
# File_name:            DNN.py                                                                                         #
# Creator:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Thursday - March 1st, 2018                                                                     #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Tuesday - August 21st, 2018                                                                    #
# Description:          Deep Neural Network with 10 hidden layers and two input and output layers                      #
#                       The initial definition of the class formats the data, gets dummy variables and scales from 0   #
#                       to 1. It also creates a sequential neural network model in keras                               #
#                       The train() method, fits the model against a disjoint training and test sets derived from the  #
#                       train.csv file loaded in challenge.py. This is because the challenge set does not contain      #
#                       Target values in order to assess accuracy.                                                     #
#                       The plot_history() method, draws a graph of the number of epochs of the fitted model against   #
#                       the training and tests losses.                                                                 #
#                       The predict() method, assess model accuracy. It finds RMSE, R^2 and CAPE (defined in the       #
#                       challenge description) https://bbdc.csl.uni-bremen.de                                          #
#                       When used the predict_challenge() method, it makes predictions over the challenge set and      #
#                       saves the results in a .csv file                                                               #
########################################################################################################################
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from DNN_config.config import TARGET_COL, DUMMIE_VARS,\
    SCAL_LOW_LIM, SCAL_UPPER_LIM,TEST_SIZE, LEARNING_RATE,\
    EPOCHS, BATCH_SIZE


class EnergyPredictor:
    """
        Trains a deep neural network of eight layers
    """
    def __init__(self, train_url, challenge_url):

        # Read training and test files files
        trainSet = pd.read_csv(train_url)
        challengeSet = pd.read_csv(challenge_url)

        # Drop variables that don't add accuracy to the model
        drop_cols = [trainSet.columns.get_loc("Datum"), trainSet.columns.get_loc("Interpoliert"),
                     trainSet.columns.get_loc("Kapazitaat"), trainSet.columns.get_loc("Zeit"),
                     trainSet.columns.get_loc("Week_Day"), trainSet.columns.get_loc("Month_Day"),
                     trainSet.columns.get_loc("Year_Week"), trainSet.columns.get_loc("Year_Day"),
                     trainSet.columns.get_loc("Cloud"), trainSet.columns.get_loc("Precipitation"),
                     trainSet.columns.get_loc("Solar"), trainSet.columns.get_loc("Wind")]

        # Drop less correlated variables
        trainSet = trainSet.drop(trainSet.columns[drop_cols], axis=1)
        challengeSet = challengeSet.drop(challengeSet.columns[drop_cols], axis=1)

        # Separate target column in both, train and challenge sets
        trainTarget = trainSet[[TARGET_COL]]
        challengeTarget = challengeSet[[TARGET_COL]]

        # Separate train and test predictors
        trainPredictors = trainSet.drop(trainSet.columns[[trainSet.columns.get_loc(TARGET_COL)]], axis=1)
        challengePredictors = challengeSet.drop(challengeSet.columns[[challengeSet.columns.get_loc(TARGET_COL)]], axis=1)

        # Get train number of rows(will be used to separate train and test set matrix)
        train_rows = trainPredictors.__len__()

        # Insert zeros at testTarget to use MinMaxScaler in complete set, because testTarget.Output contains 'X'
        challengeTarget.Output = 0

        """
        Bind frames by column and row to perform scaling over the complete observations, train and challenge sets
        "cbind" targets at the end of each (train and test) frame
        """
        trainSet = pd.concat([trainTarget.reset_index(drop=True), trainPredictors], axis=1)
        challengeSet = pd.concat([challengeTarget.reset_index(drop=True), challengePredictors], axis=1)

        # row-bind data frames of predictors
        complete_set = trainSet.append(challengeSet)

        # Create "One Hot Encoding" categorical variables
        complete_set = pd.get_dummies(complete_set, columns= DUMMIE_VARS)
        data_set = complete_set.as_matrix()

        # Scale target and predictors (Normalize from 0 to 1)
        scaler = MinMaxScaler(feature_range=(SCAL_LOW_LIM, SCAL_UPPER_LIM))
        scaled_set = scaler.fit_transform(data_set)

        # Get number of column predictors
        n_cols = scaled_set.shape[1]-1 # -1 because of the Target column

        # Separate train and test predictors
        scaled_train_set = scaled_set[0:train_rows, :]
        scaled_challenge_set = scaled_set[train_rows:, :]

        # Divide training set into another training and test sets to validate the model and assess accuracy
        train_scaled_train_set, test_scaled_train_set = train_test_split(scaled_train_set,
                                                                         test_size= TEST_SIZE,
                                                                         random_state=777)

        """
        Specification of Neural Network parameters
        """
        input_shape = (n_cols,) #Number of inputs

        # Create a sequential neural network model in keras
        model = Sequential()
        model.add(Dense(n_cols, activation="relu", input_shape=input_shape)) # n_cols inputs with n_cols + 1 neurons in the input layer, one extra to add a bias term
        model.add(Dense(n_cols, activation="relu"))
        model.add(Dense(75, activation="relu"))
        model.add(Dense(50, activation="relu"))
        model.add(Dense(33, activation="relu"))
        model.add(Dense(22, activation="relu"))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(7, activation="relu"))
        model.add(Dense(5, activation="relu"))
        model.add(Dense(3, activation="relu"))
        model.add(Dense(1)) # 1 output neuron

        # Fit Neural Network to
        my_optimizer = optimizers.Adam(lr= LEARNING_RATE)   # Use Adam optimizer
        model.compile(optimizer=my_optimizer,
                      loss="mean_squared_error",
                      metrics=["accuracy"])

        self.model = model
        self.train_scaled_train_set = train_scaled_train_set
        self.test_scaled_train_set = test_scaled_train_set
        self.scaler = scaler
        self.data_set = data_set
        self.trainTarget = trainTarget
        self.scaled_train_set = scaled_train_set
        self.scaled_challenge_set = scaled_challenge_set

    def train(self):
        """
        Fit Network on train and test data derived from the original
        training data and get history of predictions vs real values
        :return:
        """
        history = self.model.fit(self.train_scaled_train_set[:, 1:],
                                 self.train_scaled_train_set[:,0],
                                 epochs=EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 validation_data=(self.test_scaled_train_set[:,1:],
                                                  self.test_scaled_train_set[:,0]),
                                 verbose=2,
                                 shuffle=False)
        return history

    def plot_history(self, history):
        """
        plot history of fitted model. Epochs vs loss
        :return:
        """
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        return True

    def predict(self):
        """
        Predict training data class from the features
        :return:
        """
        #Make predictions on training set
        scaled_train_prediction = self.model.predict(self.scaled_train_set[:, 1:])   # Calculate predictions

        # Create copy of the scaled train set to compare results and replace Output column with the Output predictions to invert values
        prediction_set = self.scaled_train_set
        prediction_set[:, 0] = scaled_train_prediction[:, 0]

        # Invert scaling for predictions on training
        inv_prediction = self.scaler.inverse_transform(prediction_set)

        plt.plot(self.data_set[:300, 0], label="Real")         # Plot two days of real data
        plt.plot(inv_prediction[:300, 0], label="Prediction")  # Plot the interval of two days for predicted values
        plt.legend()
        plt.show()

        #Assess R-squared, RMSE and CAPE
        r_sqrt = np.sum(pow(inv_prediction[:,0]-np.mean(self.data_set[:52508, 0]), 2))/np.sum(pow(self.data_set[:52508, 0]-np.mean(self.data_set[:52508, 0]), 2)) # R-squared= %98.79
        print('R-squared: %.4f' % r_sqrt)
        rmse = sqrt(mean_squared_error(self.data_set[:52508, 0], inv_prediction[:,0]))   # 1762.769
        print('RMSE: %.4f' % rmse)
        cape = np.sum(abs(inv_prediction[:,0]-self.data_set[:52508, 0]))/self.trainTarget.sum()  # Cumulative Absolute Percentage Error= %4.76
        print('CAPE: %.4f' % cape)
        return True

    def predict_challenge(self):
        """
        Predict challenge data class from the features
        :return:
        """
        scaled_test_prediction = self.model.predict(self.scaled_challenge_set[:,1:])

        # Create copy of the scaled challenge set to correctly invert values
        test_prediction_set = self.scaled_challenge_set
        test_prediction_set[:, 0] = scaled_test_prediction[:, 0]

        # Invert scaling for predictions on test
        test_inv_prediction = self.scaler.inverse_transform(test_prediction_set)

        #convert your array into a dataframe
        df = pd.DataFrame(test_inv_prediction)
        ## save to xlsx file
        filepath = 'Prediction_NN.xlsx'
        df.to_excel(filepath, index=False)

        return True
