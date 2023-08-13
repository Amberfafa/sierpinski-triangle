import torch
import matplotlib.pyplot as plt

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_POINTS = 100000

def generate_sierpinski(n_points=N_POINTS):
    # Initial vertices of the triangle
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]]).to(device)

    # Starting point (can be random or a fixed point)
    point = torch.tensor([0.1, 0.5]).to(device)

    points = [point]
    for _ in range(n_points):
        # Randomly select a vertex
        random_vertex = vertices[torch.randint(0, 3, (1,)).item()]
        # Move halfway from current point to chosen vertex
        point = 0.5 * (point + random_vertex)
        points.append(point)

    return torch.stack(points)

points = generate_sierpinski()

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(points[:, 0].cpu(), points[:, 1].cpu(), s=0.1, color='blue')
plt.title("Sierpinski Triangle")
plt.axis('off')
plt.tight_layout()
plt.show()


