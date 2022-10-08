# ===== < IMPORTS & CONSTANTS > =====
! git clone https://github.com/orchidObsessed/sbx_osu_cs4783.git
from sbx_osu_cs4783.ass2.helpers import algebra as alg # Math
from sbx_osu_cs4783.ass2.helpers import sapilog as sl
from sbx_osu_cs4783.ass2.neural import * # ML stuff
import numpy as np
import matplotlib.pyplot as plt
sl.vtalk = 2 # Log print level (don't set this to 3 or higher, it will be spammy; check logfile if curious)
sl.vwrite = 4 # Log outfile level
x_train, y_train = list(np.loadtxt("X_train.csv")), list(np.loadtxt("Y_train.csv"))
x_test, y_test = list(np.loadtxt("X_test.csv")), list(np.loadtxt("Y_test.csv"))
network.VALIDATA = x_test
network.VALILABEL = y_test
n_epochs = 100

# ===== < MAIN > =====
model = network.NNetwork()
model += layer.Flatten(2, (2, 1)) # Input layer
# model += layer.Layer(2, alg.sigmoid, alg.q_sigmoid) # Hidden layer
model += layer.Layer(1, alg.identity, alg.q_identity) # Output layer

cost_per_epoch, acc_per_epoch = model.train(x_train, y_train, alg.mse, alg.q_mse, 1, n_epochs, learning_rate=.01, report_freq=100)
model.evaluate(x_test, y_test, alg.mse)
model.tell_params()

# ===== < DISCUSSION > =====
# 1. I chose the identity activation function for the output layer. I chose this activation because it varies on (-inf, inf), and the output also seems to vary on a large (*potentially* infinite) range. If I chose an activation function like ReLU, Sigmoid, etc., it would be stuck at a minimum of 0, which would not be able to represent all the data.

# 2. There should only be 1 neuron. Since there is only one parameter as output (single scalar).

# 3. Average loss and MSE is reported after training, in the neural.network.NNetwork.evaluate() function. For most of my cases when using sigmoid as the activation function for the hidden layer, the average MSE loss was ~3186, and the accuracy was 0%.

# 4.
fig, (top, bot) = plt.subplots(2)
x = list(range(0, n_epochs))
fig.suptitle('Question #4 - Accuracy (top) and Cost (bot) per epoch')
top.plot(x, acc_per_epoch)
bot.plot(x, cost_per_epoch)
plt.show()


# 5. As the learning rate increased, the number of epochs needed to converge decreased. However, the likelihood that it would not converge increased, and the accuracy / average cost seemed more sporadic. When using a lower learning rate, the number of epochs needed to converge increased, but it would converge well every time, with a very high accuracy and low average cost.

# 6a. In my case, the update rule does not need to be derived again, as it is done automatically. Even still, when using matrix notation, it does not need to be derived again; the size and shape of the matrix is irrelevant, and can remain symbolic. If I was calculating the backpop / update rules for each neuron by hand, then yes, it would (but that would be a bad idea).

# 6b.

# 7. When I used sigmoid activation, my nework would always converge in a really weird way that I can't really explain: it would converge to the mean values of the train data (approx. +- 63), and would be either positive or negative depending on the sign of the label (eg. if yhat = -0.1, a=-63, and if yhat = 100, a=63). I suspect that this is because using sigmoid in the hidden layer makes it difficult to represent the numerical values of the input. When I used an activation function like the identity (linear) function, it would work with 100% accuracy every time. When I used functions that could accomodate for negative values (tanh, leaky relu, etc), it would also do better.

# 7a. Again, I do not, because it is done automatically. Even if it were not done automatically, it would still not need to be changed, as 99% of things can be left in a symbolic form. All we need is the local gradient (which depends on the derivative of the activation function, which can be left in symbolic form).

# 7b. Since I have a file with a few activation functions and their derivatives, the changes I need to make are minimal; I just specify the activation function and it's derivative (not very dynamic, I know), and the backprop algorithm will take care of the rest.

# 7c.
