import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x) /np.sum(np.exp(x))

x = np.linspace(-10,10,100)
print(x)

# create plots for each activation function
fig,axs = plt.subplots(2,2,figsize = (8,8))

axs[0, 0].plot(x, sigmoid(x))
axs[0, 0].set_title('Sigmoid')

axs[0,1].plot(x, tanh(x))
axs[0,1].set_title('tanh')

axs[1,0].plot(x,relu(x))
axs[1,0].set_title('relu')

axs[1,1].plot(x,softmax(x))
axs[1,1].set_title('softmax')

fig.suptitle("common activation function")

plt.show()