########################################################################################################################
# File_name:            challenge.py                                                                                   #
# Creator:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Thursday - March 1st, 2018                                                                     #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Tuesday - August 21st, 2018                                                                    #
# Description:          It imports the EnergyPredictor classes from the DNN_model and RF_model modules.                #
#                       It instantiates both the neural network and the random forest, the .csv files are passed as    #
#                       arguments.                                                                                     #
#                       In the case of the Neural Network approach, it trains and gets the history of the trained      #
#                       models. Then, it plots the number of epochs against the losses of both the training and test   #
#                       sets. Finally, it predicts and assesses the model accuracy. The commented line, predicts the   #
#                       challenge set and saves a .csv file with the results.                                          #
#                       The random forest approach only trains the model and predicts over training and test sets.     #
#                       It also assesses the accuracy of the model, and the commented line also predicts over the      #
#                       challenge set and saves a .csv file with the results.                                          #
########################################################################################################################
from DNN_model.DNN import EnergyPredictor as EP_NN
from RF_model.RFR import EnergyPredictor as EP_RF

"""
    Neural Network
    Instantiate the EnergyPredictor class
    get the loss history of the training method
    plot the history of the trained model
    Get prediction accuracy over the training set     
"""
prd_NN = EP_NN('train.csv',
               'challenge.csv')
history = prd_NN.train()
prd_NN.plot_history(history)
prd_NN.predict()
# prd_NN.predict_challenge()

"""
    Random Forest
    Instantiate the EnergyPredictor class
    Train the tree
    Get prediction accuracy over the training set     
"""
prd_RF = EP_RF('train.csv',
               'challenge.csv')
prd_RF.train()
prd_RF.predict()
#prd_RF.predict_challenge()