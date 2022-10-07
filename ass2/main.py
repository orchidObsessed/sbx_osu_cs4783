# ===== < IMPORTS & CONSTANTS > =====
from helpers import algebra as alg # Math
from helpers import sapilog as sl
from neural import * # ML stuff
import numpy as np

sl.vtalk = 4

globalpath = "C:\\Users\\wwaddell\\Desktop\\Quick Projects\\sbx-osu-cs4783\\ass2\\"
x_train, y_train = list(np.loadtxt(globalpath + "X_train.csv")), list(np.loadtxt(globalpath + "Y_train.csv"))
x_test, y_test = list(np.loadtxt(globalpath + "X_test.csv")), list(np.loadtxt(globalpath + "Y_test.csv"))

# ===== < MAIN > =====
model = network.NNetwork()
model += layer.Flatten(2, (2, 1)) # Input layer
model += layer.Layer(2, alg.sigmoid, alg.q_sigmoid) # Hidden layer
model += layer.Layer(1, alg.identity, alg.q_identity) # Output layer

model.train(x_train, y_train, alg.mse, alg.q_mse, 5, 50000, learning_rate=.01, report_freq=5000)
model.evaluate(x_test, y_test, alg.mse)
model.tell_params()
