import numpy as np
import matplotlib.pyplot as plt

# 1. Reuse the previous code for the ellipse
center_x = 0
center_y = 0
semi_axis_a = 3
semi_axis_b = 2

# Generate data points for the ellipse path
t = np.linspace(0, 2 * np.pi, 200)
x = center_x + semi_axis_a * np.cos(t)
y = center_y + semi_axis_b * np.sin(t)

# 2. Define and plot the particle's trajectory
# The trajectory is the ellipse itself. We'll show the particle's position at a few time points.
particle_indices = [25, 75, 125, 175] # Points at t=pi/4, 3pi/4, 5pi/4, 7pi/4
particle_x = x[particle_indices]
particle_y = y[particle_indices]

# Create the plot
# Plot the full trajectory path
plt.plot(x, y, linestyle='--', color='gray', label='Trajectory Path')

# Plot the particle 'p' at different points in time
plt.scatter(particle_x, particle_y, color='red', s=100, zorder=5, label='Particle p')

# Add labels for clarity
for i, index in enumerate(particle_indices):
    plt.text(x[index] + 0.1, y[index] + 0.1, f'p(t{i+1})')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Trajectory of Particle 'p' on an Ellipse")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()