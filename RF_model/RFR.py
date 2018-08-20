########################################################################################################################
# File_name:            RFR.py                                                                                         #
# Creator:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Thursday - March 1st, 2018                                                                     #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Tuesday - August 21st, 2018                                                                    #
# Description:          Random Forest model. As in the Neural Network, the initial definition formats the data in      #
#                       order to make use of sklearn's RandomForestRegressor                                           #
#                       The train() method, fits the model against a disjoint training and test sets derived from the  #
#                       train.csv file loaded in challenge.py. This is because the challenge set does not contain      #
#                       Target values in order to assess accuracy.                                                     #
#                       The predict() method, assess model accuracy. It finds RMSE, R^2 and CAPE (defined in the       #
#                       challenge description) https://bbdc.csl.uni-bremen.de                                          #
#                       When used the predict_challenge() method, it makes predictions over the challenge set and      #
#                       saves the results in a .csv file                                                               #
########################################################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error
from RF_config.config import TARGET_COL, TEST_SIZE, DROP_VAR, RNDM_STATE, TREES, CORES, OOB, LEAF_SAMPLES, IMPURITY_DECREASE

class EnergyPredictor:
    """
        Trains a Random Forest of n TREES
    """
    def __init__(self, train_url, challenge_url):
        # Read training and test files files
        trainSet = pd.read_csv(train_url)
        challengeSet = pd.read_csv(challenge_url)

        # Drop Year variable
        trainSet = trainSet.drop(DROP_VAR, axis=1)
        challengeSet = challengeSet.drop(DROP_VAR, axis=1)

        # Print shape of training and test set
        #print("Shape of Train Set:", trainSet.shape)
        #print("Shape of  Challenge Set:", challengeSet.shape)

        # Summary statistics for training and test sets
        #trainSet.describe()
        #challengeSet.describe()

        # Separate test and challenge Outputs
        train_Target = trainSet[TARGET_COL]
        challenge_Target = challengeSet[TARGET_COL]

        # Drop Output from train and challenge predictors
        train_Predictors = trainSet.drop(TARGET_COL, axis=1)
        challenge_Predictors = challengeSet.drop(TARGET_COL, axis=1)

        # Get train and challenge number of rows(will be used to separate train and test set matrix)
        train_rows = train_Predictors.__len__()
        challenge_rows = challenge_Predictors.__len__()

        # All frames as matrix
        trainT = train_Target.values
        trainP = train_Predictors.values

        # Split the training data into other training and testing sets
        train_trainPredictors, test_trainPredictors, train_trainTarget, test_trainTarget = train_test_split(trainP,
                                                                                                            trainT,
                                                                                                            test_size=TEST_SIZE,
                                                                                                            random_state=RNDM_STATE)

        train_trainTarget = train_trainTarget.reshape((len(train_trainTarget),))
        test_trainTarget = test_trainTarget.reshape((len(test_trainTarget),))

        # Make sure it is splitted correctly
        #print('Training Predictors Shape:', train_trainPredictors.shape)
        #print('Training Target Shape:', train_trainTarget.shape)
        #print('Testing Predictors Shape:', test_trainPredictors.shape)
        #print('Testing Target Shape:', test_trainTarget.shape)

        # Instantiate model with n decision TREES
        rf = RandomForestRegressor(n_estimators= TREES,
                                   n_jobs= CORES,
                                   random_state=RNDM_STATE,
                                   oob_score = OOB,
                                   min_samples_leaf = LEAF_SAMPLES,
                                   min_impurity_decrease= IMPURITY_DECREASE)

        self.rf = rf
        self.train_trainPredictors = train_trainPredictors
        self.train_trainTarget = train_trainTarget
        self.test_trainPredictors = test_trainPredictors
        self.test_trainTarget = test_trainTarget
        self.challenge_Predictors = challenge_Predictors
        self.train_Predictors = train_Predictors

    def train(self):
        """
        Fit Random Forest on train
        :return:
        """
        self.rf.fit(self.train_trainPredictors, self.train_trainTarget)
        return True

    def predict(self):
        """
        Predict training data class from the features
        :return:
        """
        # Use the forest's predict method on the test data
        predictions = self.rf.predict(self.test_trainPredictors)

        # Calculate the absolute errors
        errors = abs(predictions - self.test_trainTarget)

        # Print out the mean absolute error (mae), RMSE, R^2 and CAPE
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        r_sqrt = np.sum(pow(predictions - np.mean(self.test_trainTarget), 2))/np.sum(pow(self.test_trainTarget - np.mean(self.test_trainTarget), 2)) # R-squared= %92.5
        print('R-squared: %.4f' % r_sqrt)
        rmse = sqrt(mean_squared_error(self.test_trainTarget, predictions))
        print('RMSE: %.4f' % rmse)
        cape = np.sum(abs(predictions - self.test_trainTarget))/self.test_trainPredictors[:, self.train_Predictors.columns.get_loc("Kapazitaat")].sum()  # Cumulative Absolute Percentage Error= %6.7
        print('CAPE: %.4f' % cape) # 0.1146

        return True

    def predict_challenge(self):
        """
        Predict challenge data class from the features
        :return:
        """
        #Make predictions on the challenge set
        challengePredictions = self.rf.predict(self.challenge_Predictors)

        #convert your array into a dataframe
        df = pd.DataFrame(challengePredictions)
        ## save to xlsx file
        filepath = 'Prediction_RF.xlsx'
        df.to_excel(filepath, index=False)

        return True