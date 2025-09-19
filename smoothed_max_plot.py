import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 14,
    'font.family': 'Times New Roman',
    "text.latex.preamble": r"\usepackage{amsmath}",
})

# Define the functions based on the provided formulas
def f_prime(t, a):
    """
    Defines the first smoothed max function F'(t).
    F'(t) = -a + sqrt(max(t, 0)^2 + a^2)
    """
    return -a + np.sqrt(np.maximum(t, 0)**2 + a**2)

def f_double_prime(t, a):
    """
    Defines the second smoothed max function F''(t).
    F''(t) is a piecewise function.
    F''(t) = 0 for t <= 0
    F''(t) = t / sqrt(t^2 + a^2) for t > 0
    """
    result = np.zeros_like(t)
    positive_t_indices = t > 0
    result[positive_t_indices] = t[positive_t_indices] / np.sqrt(t[positive_t_indices]**2 + a**2)
    return result

# --- Plotting Setup ---

# Set a value for 'a'. This parameter controls the 'smoothness'.
# A smaller 'a' value makes the functions more closely approximate the non-smoothed versions.
a_value = 0.1

# Create a range of t values for the plot.
t_values = np.linspace(0, 3, 1000)

# Calculate the function values
y_prime = f_prime(t_values, a_value)
y_double_prime = f_double_prime(t_values, a_value)

# --- Create the plot ---
plt.figure(figsize=(6, 6))

# Plot F'(t)
plt.plot(
    t_values, y_prime,
    label=r"$F'(t) = -a + \sqrt{\max(t, 0)^2 + a^2}$",
    color='blue'
)

# Plot F''(t)
plt.plot(
    t_values, y_double_prime,
    label=r"$F''(t) = \begin{cases} 0 & t \leq 0 \\ \frac{t}{\sqrt{t^2 + a^2}} & t > 0 \end{cases}$",
    color='red'
)

# Add a dashed line for y = 0 for better visualization
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Add a dashed line for the non-smoothed max and its derivative for comparison
# F(t) = max(t, 0) is a ReLU-like function
# F'(t) = 0 for t <= 0, 1 for t > 0 (Heaviside step function)
plt.plot(
    t_values, np.maximum(t_values, 0),
    'g--',
    label=r"$\max(t, 0)$" + " (Non-smoothed max)"
)
plt.axvline(0, color='k', linestyle=':', linewidth=0.8)

# Add labels, title, and legend
plt.title(f'Smoothed Max Functions for a = {a_value}', fontsize=16)
plt.xlabel('t', fontsize=12)
plt.ylabel('Function Value', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()