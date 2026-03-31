import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(rgb)
ax.plot(int(x * W), int(y * H), "r+", markersize=18, markeredgewidth=3,
        label="predicted")
ax.set_title(f"Logo centroid: ({int(x*W)}, {int(y*H)})")
ax.legend()
plt.tight_layout()
plt.show()