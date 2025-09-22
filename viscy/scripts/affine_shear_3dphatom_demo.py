# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from viscy.transforms._transforms import BatchedRandAffined


def create_3d_phantom(shape=(64, 128, 128)):
    """
    Create a 3D phantom with geometric shapes in ZYX coordinates.

    Parameters
    ----------
    shape : tuple
        Shape of the phantom in (Z, Y, X) order.

    Returns
    -------
    np.ndarray
        3D phantom array.
    """
    z_size, y_size, x_size = shape
    phantom = np.zeros(shape)

    # Create coordinate grids
    z, y, x = np.mgrid[0:z_size, 0:y_size, 0:x_size]

    # Center coordinates
    z_center, y_center, x_center = z_size // 2, y_size // 2, x_size // 2

    # Sphere in the center
    sphere_radius = min(shape) // 6
    sphere_mask = (
        (z - z_center) ** 2 + (y - y_center) ** 2 + (x - x_center) ** 2
    ) <= sphere_radius**2
    phantom[sphere_mask] = 1.0

    # Cylinder along Z-axis
    cyl_radius = sphere_radius // 2
    cyl_x_offset = x_center + sphere_radius * 2
    cyl_y_offset = y_center
    cylinder_mask = ((y - cyl_y_offset) ** 2 + (x - cyl_x_offset) ** 2) <= cyl_radius**2
    phantom[cylinder_mask] = 0.7

    # Box
    box_size = sphere_radius
    box_z_start = z_center - sphere_radius * 2
    box_z_end = box_z_start + box_size
    box_y_start = y_center - box_size // 2
    box_y_end = box_y_start + box_size
    box_x_start = x_center - box_size // 2
    box_x_end = box_x_start + box_size

    if box_z_start >= 0 and box_z_end < z_size:
        phantom[box_z_start:box_z_end, box_y_start:box_y_end, box_x_start:box_x_end] = (
            0.5
        )

    return phantom


def plot_3d_projections(phantom, title="3D Phantom Projections"):
    """
    Plot 3D phantom with maximum intensity projections along each axis.

    Parameters
    ----------
    phantom : np.ndarray
        3D phantom array in ZYX order.
    title : str
        Title for the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    mip_xy = np.max(phantom, axis=0)
    mip_xz = np.max(phantom, axis=1)
    mip_yz = np.max(phantom, axis=2)

    z_center = phantom.shape[0] // 2

    im1 = axes[0, 0].imshow(mip_xy, cmap="viridis", origin="lower")
    axes[0, 0].set_title("MIP XY (along Z)")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(mip_xz, cmap="viridis", origin="lower")
    axes[0, 1].set_title("MIP XZ (along Y)")
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Z")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(mip_yz, cmap="viridis", origin="lower")
    axes[1, 0].set_title("MIP YZ (along X)")
    axes[1, 0].set_xlabel("Y")
    axes[1, 0].set_ylabel("Z")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(phantom[z_center, :, :], cmap="viridis", origin="lower")
    axes[1, 1].set_title(f"Central XY slice (Z={z_center})")
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    return fig


def apply_shear_transform(phantom, shear_values, prob=1.0):
    """
    Apply shear transformation using BatchedRandAffined.

    Parameters
    ----------
    phantom : np.ndarray
        3D phantom array in ZYX order.
    shear_values : list or tuple
        Shear values for each facet [sxy, sxz, syx, syz, szx, szy] in radians.
    prob : float
        Probability of applying transform.

    Returns
    -------
    np.ndarray
        Transformed phantom.
    """
    # Convert to torch tensor with batch and channel dimensions
    phantom_tensor = torch.from_numpy(phantom).float()
    phantom_tensor = phantom_tensor.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and channel dims
    transform = BatchedRandAffined(
        keys=["image"], prob=prob, shear_range=shear_values, mode="bilinear"
    )
    sample = {"image": phantom_tensor}
    transformed_sample = transform(sample)
    transformed_phantom = transformed_sample["image"].squeeze().numpy()

    return transformed_phantom


if __name__ == "__main__":
    phantom = create_3d_phantom((64, 128, 128))

    fig1 = plot_3d_projections(phantom, "Original Phantom")
    shear_names = ["s01", "s02", "s10", "s12", "s20", "s21"]

    """
    s{ij}:

        [
            [1.0, params[0], params[1], 0.0],
            [params[2], 1.0, params[3], 0.0],
            [params[4], params[5], 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    """

    for axis in range(6):
        shear = [0.0] * 6
        shear[axis] = 0.5
        phantom_sheared = apply_shear_transform(phantom, shear)
        fig = plot_3d_projections(
            phantom_sheared, f"Shear applied: {shear_names[axis]}=0.5"
        )
        plt.show()

    shear_combined = [0.2, 0.2, 0.0, 0.2, 0.0, 0.2]
    phantom_combined = apply_shear_transform(phantom, shear_combined)
    fig6 = plot_3d_projections(phantom_combined, "Combined Shears")

    plt.show()

# %%
