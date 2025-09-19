import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the generated figure
try:
    img = Image.open('fig_per_dataset_accuracy.png')
    print(f"fig_per_dataset_accuracy.png size: {img.size}")

    # Convert to numpy array
    img_array = np.array(img)

    # Check if image has content (not all white/empty)
    if img_array.mean() < 250:  # Not mostly white
        print("Image has content (not blank)")
    else:
        print("WARNING: Image might be blank or mostly white")

    # Check specific regions
    print(f"Image shape: {img_array.shape}")
    print(f"Mean pixel value: {img_array.mean():.2f}")
    print(f"Min pixel value: {img_array.min()}")
    print(f"Max pixel value: {img_array.max()}")

except Exception as e:
    print(f"Error loading image: {e}")

# Let's also check what matplotlib is actually drawing
import matplotlib.pyplot as plt
fig = plt.gcf()
axes = plt.gca()

# Get current figure info
print(f"\nCurrent matplotlib figure:")
print(f"  Number of axes: {len(fig.axes)}")
if len(fig.axes) > 0:
    ax = fig.axes[0]
    print(f"  Legend entries: {ax.get_legend()}")
    if ax.get_legend():
        print(f"  Legend labels: {[t.get_text() for t in ax.get_legend().get_texts()]}")