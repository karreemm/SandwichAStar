# Import numpy  and matplotlib libraries
import matplotlib.pyplot as plt
import numpy as np

# Define the motion space parameters
x_min = 0  # minimum x coordinate
x_max = 100  # maximum x coordinate
y_min = 0  # minimum y coordinate
y_max = 100  # maximum y coordinate
n_obstacles = 5  # number of obstacles
obstacle_size = 10  # obstacle size
obstacle_shape = "square"  # obstacle shape
psi_min = 0  # minimum stream value
psi_max = 1  # maximum stream value
n_channels = n_obstacles + 1  # number of navigable channels
delta_psi = (psi_max - psi_min) / n_channels  # stream value increment

# Define the planning space parameters
phi_min = 0  # minimum potential value
phi_max = 1  # maximum potential value
n_x = 101  # number of grid points along x_axis
n_y = 101  # number of grid points along y_axis
delta_phi = (phi_max - phi_min) / (n_x - 1)  # potential value increment
delta_psi_2 = (psi_max - psi_min) / (
    n_y - 1
)  # stream value increment for the planning space grid

# Generate the motion space grid
x = np.linspace(x_min, x_max, n_x)  # x coordinates
y = np.linspace(y_min, y_max, n_y)  # y coordinates
X, Y = np.meshgrid(x, y)  # 2D grid

# Generate the planning space grid
phi = np.linspace(phi_min, phi_max, n_x)  # phi coordinates
psi = np.linspace(psi_min, psi_max, n_y)  # psi coordinates
PHI, PSI = np.meshgrid(phi, psi)  # 2D grid

# Initialize the obstacle and navigable masks
obstacle_mask = np.zeros((n_y, n_x), dtype=bool)  # True for obstacle points
navigable_mask = np.ones((n_y, n_x), dtype=bool)  # True for navigable points

# Generate the obstacles randomly
np.random.seed(0)  # set the random seed for reproducibility
obstacle_centers = np.random.uniform(
    low=obstacle_size, high=100 - obstacle_size, size=(n_obstacles, 2)
)  # obstacle centers
obstacle_psi = np.linspace(
    psi_min + delta_psi, psi_max - delta_psi, n_obstacles
)  # obstacle stream values
for i in range(n_obstacles):
    # Compute the distance from the obstacle center
    distance = np.sqrt(
        (X - obstacle_centers[i, 0]) ** 2 + (Y - obstacle_centers[i, 1]) ** 2
    )
    # Update the obstacle mask based on the obstacle shape
    if obstacle_shape == "circle":
        obstacle_mask = np.logical_or(obstacle_mask, distance <= obstacle_size / 2)
    elif obstacle_shape == "square":
        obstacle_mask = np.logical_or(
            obstacle_mask, np.abs(X - obstacle_centers[i, 0]) <= obstacle_size / 2
        )
        obstacle_mask = np.logical_or(
            obstacle_mask, np.abs(Y - obstacle_centers[i, 1]) <= obstacle_size / 2
        )
    # Update the navigable mask based on the obstacle stream values
    navigable_mask = np.logical_and(
        navigable_mask, PSI > obstacle_psi[i] + delta_psi_2 / 2
    )
    navigable_mask = np.logical_and(
        navigable_mask, PSI < obstacle_psi[i] - delta_psi_2 / 2
    )

# Update the navigable mask to exclude the boundary points
navigable_mask[:, 0] = False
navigable_mask[:, -1] = False
navigable_mask[0, :] = False
navigable_mask[-1, :] = False

# Solve the PDEs for x and y using the finite difference method
x = np.zeros((n_y, n_x))  # x values in the motion space
y = np.zeros((n_y, n_x))  # y values in the motion space
x[:, 0] = x_min  # left boundary condition
x[:, -1] = x_max  # right boundary condition
y[0, :] = y_min  # bottom boundary condition
y[-1, :] = y_max  # top boundary condition
# Loop over the interior points
for j in range(1, n_y - 1):
    for i in range(1, n_x - 1):
        # Check if the point is navigable
        if navigable_mask[j, i]:
            # Compute the coefficients a, b, c
            a = (x[j + 1, i] - x[j - 1, i]) ** 2 + (y[j + 1, i] - y[j - 1, i]) ** 2
            b = (x[j + 1, i] - x[j - 1, i]) * (x[j, i + 1] - x[j, i - 1]) + (
                y[j + 1, i] - y[j - 1, i]
            ) * (y[j, i + 1] - y[j, i - 1])
            c = (x[j, i + 1] - x[j, i - 1]) ** 2 + (y[j, i + 1] - y[j, i - 1]) ** 2
            # Solve the linear system for x and y
            x[j, i] = (
                a * (x[j, i + 1] + x[j, i - 1])
                - 2 * b * (x[j + 1, i] + x[j - 1, i])
                + c * (x[j + 1, i] + x[j - 1, i])
            ) / (2 * (a + c))
            y[j, i] = (
                a * (y[j, i + 1] + y[j, i - 1])
                - 2 * b * (y[j + 1, i] + y[j - 1, i])
                + c * (y[j + 1, i] + y[j - 1, i])
            ) / (2 * (a + c))


