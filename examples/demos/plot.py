import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity


def plot_vs_n_fluor(vs_nucleus, vs_membrane, fluor_nucleus, fluor_membrane):
    colormap_1 = [0.1254902, 0.6784314, 0.972549]  # bop blue
    colormap_2 = [0.972549, 0.6784314, 0.1254902]  # bop orange
    colormap_3 = [0, 1, 0]  # green
    colormap_4 = [1, 0, 1]  # magenta

    # Rescale the intensity
    vs_nucleus = rescale_intensity(vs_nucleus, out_range=(0, 1))
    vs_membrane = rescale_intensity(vs_membrane, out_range=(0, 1))
    # VS Nucleus RGB
    vs_nucleus_rgb = np.zeros((*vs_nucleus.shape[-2:], 3))
    vs_nucleus_rgb[:, :, 0] = vs_nucleus * colormap_1[0]
    vs_nucleus_rgb[:, :, 1] = vs_nucleus * colormap_1[1]
    vs_nucleus_rgb[:, :, 2] = vs_nucleus * colormap_1[2]
    # VS Membrane RGB
    vs_membrane_rgb = np.zeros((*vs_membrane.data.shape[-2:], 3))
    vs_membrane_rgb[:, :, 0] = vs_membrane * colormap_2[0]
    vs_membrane_rgb[:, :, 1] = vs_membrane * colormap_2[1]
    vs_membrane_rgb[:, :, 2] = vs_membrane * colormap_2[2]
    # Merge the two channels
    merged_vs = np.zeros((*vs_nucleus.shape[-2:], 3))
    merged_vs[:, :, 0] = vs_nucleus * colormap_1[0] + vs_membrane * colormap_2[0]
    merged_vs[:, :, 1] = vs_nucleus * colormap_1[1] + vs_membrane * colormap_2[1]
    merged_vs[:, :, 2] = vs_nucleus * colormap_1[2] + vs_membrane * colormap_2[2]

    # Rescale the intensity
    fluor_nucleus = rescale_intensity(fluor_nucleus, out_range=(0, 1))
    fluor_membrane = rescale_intensity(fluor_membrane, out_range=(0, 1))
    # fluor Nucleus RGB
    fluor_nucleus_rgb = np.zeros((*fluor_nucleus.shape[-2:], 3))
    fluor_nucleus_rgb[:, :, 0] = fluor_nucleus * colormap_3[0]
    fluor_nucleus_rgb[:, :, 1] = fluor_nucleus * colormap_3[1]
    fluor_nucleus_rgb[:, :, 2] = fluor_nucleus * colormap_3[2]
    # fluor Membrane RGB
    fluor_membrane_rgb = np.zeros((*fluor_membrane.shape[-2:], 3))
    fluor_membrane_rgb[:, :, 0] = fluor_membrane * colormap_4[0]
    fluor_membrane_rgb[:, :, 1] = fluor_membrane * colormap_4[1]
    fluor_membrane_rgb[:, :, 2] = fluor_membrane * colormap_4[2]
    # Merge the two channels
    merged_fluor = np.zeros((*fluor_nucleus.shape[-2:], 3))
    merged_fluor[:, :, 0] = (
        fluor_nucleus * colormap_3[0] + fluor_membrane * colormap_4[0]
    )
    merged_fluor[:, :, 1] = (
        fluor_nucleus * colormap_3[1] + fluor_membrane * colormap_4[1]
    )
    merged_fluor[:, :, 2] = (
        fluor_nucleus * colormap_3[2] + fluor_membrane * colormap_4[2]
    )

    # %%
    # Plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))

    # Virtual staining plots
    ax[0, 0].imshow(vs_nucleus_rgb)
    ax[0, 0].set_title("VS Nuclei")
    ax[0, 1].imshow(vs_membrane_rgb)
    ax[0, 1].set_title("VS Membrane")
    ax[0, 2].imshow(merged_vs)
    ax[0, 2].set_title("VS Nuclei+Membrane")

    # Experimental fluorescence plots
    ax[1, 0].imshow(fluor_nucleus_rgb)
    ax[1, 0].set_title("Experimental Fluorescence Nuclei")
    ax[1, 1].imshow(fluor_membrane_rgb)
    ax[1, 1].set_title("Experimental Fluorescence Membrane")
    ax[1, 2].imshow(merged_fluor)
    ax[1, 2].set_title("Experimental Fluorescence Nuclei+Membrane")

    # turnoff axis
    for a in ax.flatten():
        a.axis("off")
    plt.margins(0, 0)
    plt.show()
