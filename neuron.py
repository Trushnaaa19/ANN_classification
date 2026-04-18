# neuron.py

import torch
import torch.nn as nn

# Step 1: Check environment
print("Python path:", __import__("sys").executable)

# Step 2: Create input tensor
inputs = torch.tensor([1.0, 2.0, 3.0])

# Step 3: Create a single neuron (Linear layer)
neuron = nn.Linear(in_features=3, out_features=1)

# Step 4: Forward pass
output = neuron(inputs)

# Step 5: Print results
print("Input:", inputs)
print("Weights:", neuron.weight)
print("Bias:", neuron.bias)
print("Output:", output)

# Step 6: Manual calculation check
manual_output = neuron.weight @ inputs + neuron.bias
print("Manual Output:", manual_output)