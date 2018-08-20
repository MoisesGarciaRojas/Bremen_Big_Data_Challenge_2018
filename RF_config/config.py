########################################################################################################################
# File_name:            RF_config\config.py                                                                            #
# Creator:              Moises Daniel Garcia Rojas                                                                     #
# Created:              Thursday - March 1st, 2018                                                                     #
# Last editor:          Moises Daniel Garcia Rojas                                                                     #
# Last modification:    Tuesday - August 21st, 2018                                                                    #
# Description:          Variables to format the data and the Random Forest Tree                                        #
########################################################################################################################
# To format data-sets
TARGET_COL = ['Output']
DROP_VAR = ['Datum','Zeit']
TEST_SIZE = 0.20
RNDM_STATE = 777

# Random Forest configuration
TREES = 200
CORES = -1
OOB = True
LEAF_SAMPLES = 25
IMPURITY_DECREASE = 0.01