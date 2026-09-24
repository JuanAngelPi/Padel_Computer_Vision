import pandas as pd
import numpy as np
import cv2
from pandas.core import groupby
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


df = pd.read_csv(r"C:\Padel_Project\results\players_keypoints.csv")

ankle_point = df[(df["keypoint_id"] == 15) | (df["keypoint_id"] == 16)]
combination = ankle_point.groupby(["player_id", "frame"])

X_feet = combination.apply(lambda group: np.average(group["x"], weights=group["conf"]))
Y_feet = combination.apply(lambda group: np.average(group["y"], weights=group["conf"]))

feet_positions = pd.DataFrame({
    "x": X_feet,
    "y": Y_feet
}).reset_index()

# Homography 

image_points = np.float32([
    [582, 325],    # Top-left
    [1390, 325],   # Top-right
    [214, 958],    # Bottom-left
    [1715, 958],    # Bottom-right

    # Net line (meets side walls)
    [450, 450],      # Net - left post
    [1483, 450],     # Net - right post

    # Far service line (near the back/far baseline, y=0)
    [548, 382],    # Far service line - left sideline intersection
    [1387, 382],   # Far service line - right sideline intersection
    [965, 382],    # Far service line - center T
    
    # Near service line (near the front/camera baseline, y=20)
    [310, 796],   # Near service line - left sideline intersection
    [1622, 796],  # Near service line - right sideline intersection
    [964, 796],   # Near service line - center T


])


court_points = np.float32([
    # Original 4 corners
    [0, 0],       # Back-left corner
    [10, 0],      # Back-right corner
    [0, 20],      # Front-left corner
    [10, 20],     # Front-right corner

    # Net line (meets side walls)
    [0, 10],      # Net - left post
    [10, 10],     # Net - right post

    # Far service line (near the back/far baseline, y=0)
    [0, 3.05],    # Far service line - left sideline intersection
    [10, 3.05],   # Far service line - right sideline intersection
    [5, 3.05],    # Far service line - center T

    # Near service line (near the front/camera baseline, y=20)
    [0, 16.95],   # Near service line - left sideline intersection
    [10, 16.95],  # Near service line - right sideline intersection
    [5, 16.95],   # Near service line - center T
])


H, status = cv2.findHomography(
    image_points,
    court_points
)

# Extract image coordinates
points = feet_positions[["x", "y"]].values.astype(np.float32)

# Transform into court coordinates
court_positions = cv2.perspectiveTransform(
    points.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

# Add transformed coordinates
feet_positions["court_x"] = court_positions[:, 0]
feet_positions["court_y"] = court_positions[:, 1]

print(feet_positions.head())

# ============================================================
# 1. COURT DIMENSIONS
# ============================================================

court_width = 10
court_length = 20

# ============================================================
# 2. GRID FOR HEATMAP
# ============================================================

nx = 200
ny = 400

x_edges = np.linspace(0, court_width, nx + 1)
y_edges = np.linspace(0, court_length, ny + 1)

x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

X, Y = np.meshgrid(x_centers, y_centers)

# ============================================================
# 3. PLAYER COLORS
# ============================================================

player_colors = {
    1: "Reds",
    2: "Blues",
    3: "Greens",
    4: "Purples"
}

# ============================================================
# 4. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 16))

# ============================================================
# 5. HEATMAP FOR EACH PLAYER
# ============================================================

for player_id in sorted(feet_positions["player_id"].unique()):

    player_data = feet_positions[
        feet_positions["player_id"] == player_id
    ]

    x = player_data["court_x"].to_numpy()
    y = player_data["court_y"].to_numpy()

    # Remove invalid points
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    # Keep only points inside court
    valid = (
        (x >= 0) & (x <= court_width) &
        (y >= 0) & (y <= court_length)
    )

    x = x[valid]
    y = y[valid]

    # 2D histogram
    heatmap, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges]
    )

    # Transpose because histogram2d returns [x,y]
    heatmap = heatmap.T

    # Smooth the heatmap
    heatmap = gaussian_filter(
        heatmap,
        sigma=8
    )

    # Normalize this player's heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Plot
    ax.contourf(
        X,
        Y,
        heatmap,
        levels=np.linspace(0.05, 1, 15),
        cmap=player_colors[player_id],
        alpha=0.45
    )

# ============================================================
# 6. DRAW PADEL COURT
# ============================================================
# Court dimensions
court_width = 10
court_length = 20

# -------------------------
# Outer court
# -------------------------

ax.plot(
    [0, 10, 10, 0, 0],
    [0, 0, 20, 20, 0],
    color="black",
    linewidth=2
)

# -------------------------
# Net
# -------------------------

ax.plot(
    [0, 10],
    [10, 10],
    color="black",
    linewidth=2
)

# -------------------------
# Service lines
# -------------------------

# Near side service line
ax.plot(
    [0, 10],
    [3.05, 3.05],
    color="black",
    linewidth=1.5
)

# Far side service line
ax.plot(
    [0, 10],
    [16.95, 16.95],
    color="black",
    linewidth=1.5
)

# -------------------------
# Center service lines
# -------------------------

# Near side T
ax.plot(
    [5, 5],
    [3.05, 10],
    color="black",
    linewidth=1.5
)

# Far side T
ax.plot(
    [5, 5],
    [10, 16.95],
    color="black",
    linewidth=1.5
)
# ============================================================
# 7. LABELS
# ============================================================

ax.set_xlim(0, 10)
ax.set_ylim(0, 20)

ax.set_aspect("equal")

ax.set_xlabel("Court width (m)")
ax.set_ylabel("Court length (m)")

ax.set_title("Padel Player Heatmap")
ax.invert_yaxis()


# ============================================================
# 8. LEGEND
# ============================================================

from matplotlib.patches import Patch

legend = [
    Patch(
        facecolor=plt.get_cmap(player_colors[player_id])(0.7),
        label=f"Player {player_id}"
    )
    for player_id in sorted(feet_positions["player_id"].unique())
]

ax.legend(handles=legend)

plt.show()
