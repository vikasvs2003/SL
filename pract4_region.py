import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
#extract sepal length and  petal length fetures

X = iris.data[:, [0,2]]
y = iris.target
w = np.zeros(2)
b = 0
lr = 0.1
epochs = 50

y = np.where(y == 0,0,1)

# define the perceptom
def perceptron(x,w,b) :
    
    z = np.dot(x,w ) + b
        # apply step function 
    return np.where(z >= 0 , 1, 0)

# train the perceptron model

for epochs in range(epochs):
    for i in range(len(X)):
        x = X[i]
        target = y[i]
        output = perceptron(x,w, b)
        error = target - output
        w += lr*error*x
        b += lr*error

x_min ,x_max = X[:,0].min() - 0.5,  X[:,0].max() + 0.5

y_min ,y_max = X[:,1].min() - 0.5,  X[:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min,x_max , 0.02), np.arange(y_min,y_max ,0.02))

Z = perceptron(np.c_[xx.ravel(), yy.ravel()], w, b)
Z = Z.reshape(xx.shape)


plt.contourf(xx,yy,Z,cmap=plt.cm.Paired)
plt.xlabel("sepal lengh")
plt.ylabel("petal length")
plt.title("percerptron decision result")
plt.show()

    
