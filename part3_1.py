import torch
import matplotlib.pyplot as plt

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_POINTS = 100000


def generate_barnsley_fern(n_points=N_POINTS):
    # Define transformation matrices and bias for the fern
    # Format: [A, B, C, D, E, F, probability]
    transforms = torch.tensor([
        [0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01],
        [0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85],
        [0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07],
        [-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07]
    ]).to(device)

    point = torch.tensor([0.0, 0.0]).to(device)
    points = [point]

    for _ in range(n_points):
        rand = torch.rand(1).to(device)
        if rand < 0.01:
            transform = transforms[0]
        elif rand < 0.86:
            transform = transforms[1]
        elif rand < 0.93:
            transform = transforms[2]
        else:
            transform = transforms[3]

        new_point = torch.tensor([
            transform[0] * point[0] + transform[1] * point[1] + transform[4],
            transform[2] * point[0] + transform[3] * point[1] + transform[5]
        ]).to(device)

        points.append(new_point)
        point = new_point

    return torch.stack(points)


points = generate_barnsley_fern()

# Visualization
plt.figure(figsize=(6, 10))
plt.scatter(points[:, 0].cpu(), points[:, 1].cpu(), s=0.1, color='green')
plt.title("Barnsley Fern")
plt.axis('off')
plt.tight_layout()
plt.show()
