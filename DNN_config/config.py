########################################################################################################################
# File_name:            DNN_config\config.py                                                                           #
# Creator:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Thursday - March 1st, 2018                                                                     #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Tuesday - August 21st, 2018                                                                    #
# Description:          Variables to format the data and the neural network                                            #
########################################################################################################################
# Variables used to format the data-sets
TARGET_COL = 'Output'
DUMMIE_VARS = ["Year_Month", "Time_Slot", "Hour"]
SCAL_LOW_LIM = 0
SCAL_UPPER_LIM = 1
TEST_SIZE = 0.25

# Neural Network configuration
LEARNING_RATE = 0.00001
EPOCHS = 50
BATCH_SIZE = 100