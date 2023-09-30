import tensorflow as tf
import numpy as np

# Create a tf.Variable
original_variable = tf.Variable(np.array([[1], [2], [3], [4], [5]]), dtype=tf.int32)

# Perform a slice operation on the variable
sliced_variable = original_variable[1:4,:]  # Slicing from index 1 to 3

# Modify the sliced variable
sliced_variable.assign(np.array([[10], [20], [30]]))
print(type(sliced_variable))

# Print the original variable and the modified sliced variable
print("Original Variable:", original_variable.numpy())
print("Modified Sliced Variable:", sliced_variable.numpy())