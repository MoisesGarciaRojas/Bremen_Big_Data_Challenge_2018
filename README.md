# Bremen_Big_Data_Challenge_2018
Random Forest and Deep Neural Network models which predict the energy generation in a farm of wind generators in Netherlands.
https://bbdc.csl.uni-bremen.de/

# Description
As part of the Data Mining subject at Jacobs University Bremen I teamed up with a classmate, and we botht participate in the Bremen Big Data Challenge 2018 edition. The challenge was to predict the produced wind energy of an onshore wind farm in the Netherlands. In order to train the forecast models, the organizers of the event posted the wind measurements in 15-minute intervals from January 2016 to June 2017 and the wind energy generated by the wind farm in this period of time ("training data"). For the second half of 2017 ("test data") received the wind measurements and had to predict the wind energy.

The participants were evaluated by means of an error score to determine their ranking. The lower the error score, the better the submission level. The error score was calculated as Cumulative Absolute Percentage Error (CAPE) between the predicted wind energy and the actual wind energy produced.

In the present repository a Deep Neural Network and a Random Forest are presented. The models were trained after data analysis and visualization were performed, these two tasks were job of my teammate.

The best results were achieved with the Random Forest model obtaining:

Measure   | Value
  ---     |   ---
CAPE      | 0.074
R-squared | 0.8786
RMSE      | 8871.5433

## Python Setup
1.	The “Project Interpreter” used is a 64 bits version of Python 3.6.
2.	The code was written on a 64 bits version of PyCharm Community Edition 2016.

Package   | Version
  ---     |   ---
pandas    | 0.23.4
numpy     | 1.15.0
sklearn   | 0.19.2
keras     | 2.2.2
matplotlib| 2.1.2
math      | built-in
