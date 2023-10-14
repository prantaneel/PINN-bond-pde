import numpy as np

# Create a NumPy array with some data
data = np.array([1, 2, 2, 3, 4, 4, 5, 5, 6])

# Use the unique function to find unique elements
unique_elements = np.unique(data).flatten()[:, None]

# Print the unique elements
print(unique_elements)
