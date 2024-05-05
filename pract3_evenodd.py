import numpy as np

# Define the activation function (step function)
def step_function(x):
    return 1 if x >= 0 else 0

# Define the perceptron function
def perceptron(weights, inputs):
    # Add bias term (assuming it's the first element of weights)
    weighted_sum = np.dot(weights[1:], inputs) + weights[0]
    return step_function(weighted_sum)

# Define the training data
training_data = [
    # ASCII representation of digits 0 to 9
    (1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 0
    (0, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # 1
    (1, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),  # 2
    (0, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # 3
    (1, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # 4
    (0, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # 5
    (1, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # 6
    (0, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # 7
    (1, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # 8
    (0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])   # 9
]

# Initialize weights randomly
weights = np.random.rand(11)

# Set learning rate
learning_rate = 0.1

# Train the perceptron
for _ in range(1000):
    for target, input_data in training_data:
        prediction = perceptron(weights, input_data)
        error = target - prediction
        weights[1:] += learning_rate * error * np.array(input_data)
        weights[0] += learning_rate * error

# Test the perceptron
test_data = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 0
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 1
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 2
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 3
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 6
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 9
]

print("Testing the perceptron:")
for input_data in test_data:
    prediction = perceptron(weights, input_data)
    print("Input:", input_data, "Prediction:", "Even" if prediction == 1 else "Odd")
