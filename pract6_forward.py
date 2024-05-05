
#Implement Artificial Neural Network training process in Python by using forward propagation,
#backpropagation.

# error = target - output
# delta = error * sig_derivative (output of that layer)
# w = w + transpose_input @ that_layer_delta
# b = b + sum of columns of delta of that layer

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.randn(1, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.random.randn(1, self.output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # Forward propagation
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return self.output
    
    def backward(self, x, y, output, learning_rate):
        # Backpropagation
        self.output_error = y - output
        self.output_delta = self.output_error
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta) * learning_rate
        self.bias_hidden_output += np.sum(self.output_delta) * learning_rate
        self.weights_input_hidden += x.T.dot(self.hidden_delta) * learning_rate
        self.bias_input_hidden += np.sum(self.hidden_delta) * learning_rate
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')

# Example usage:
# Define dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize neural network
nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Train neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained network
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print("Predictions after training:")
print(nn.forward(test_input))
