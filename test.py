import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

# Sample heatmap data (1D array of scalar values)
data = np.linspace(0, 1, 10)  # values from 0 to 1

# Define a custom colormap from blue to white
cmap = LinearSegmentedColormap.from_list("blue_white", ["blue", "white"])

# Normalize the data range to [0, 1]
norm = Normalize(vmin=np.min(data), vmax=np.max(data))

# Convert to RGB using colormap
rgb_array = cmap(norm(data))[:, :3]  # exclude alpha channel

print("RGB array (n x 3):")
print(rgb_array)
