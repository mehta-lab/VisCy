
''' compute the pixel ratio of background (0), uninfected (1) and infected (2) pixels in the zarr dataset'''

from iohub.ngff import open_ome_zarr

def calculate_pixel_ratio(dataset_path: str, target_channel: str):
    """
      find ratio of background, uninfected and infected pixels
        in the input dataset
    Args:
        dataset_path (str): Path to the dataset
    Returns:
        pixel_ratio (list): List of ratios of background, uninfected and infected pixels
    """
    zarr_input = open_ome_zarr(
        dataset_path,
        layout="hcs",
        mode="r+",
    )
    in_chan_names = zarr_input.channel_names

    num_pixels_bkg = 0
    num_pixels_uninf = 0
    num_pixels_inf = 0
    num_pixels = 0
    for well_id, well_data in zarr_input.wells():
        well_name, well_no = well_id.split("/")

        for pos_name, pos_data in well_data.positions():
            data = pos_data.data
            T, C, Z, Y, X = data.shape
            out_data = data.numpy()
            for time in range(T):
                Inf_mask = out_data[time, in_chan_names.index(target_channel), ...]
                # Calculate the number of pixels valued 0, 1, and 2 in 'Inf_mask'
                num_pixels_bkg = num_pixels_bkg + (Inf_mask == 0).sum()
                num_pixels_uninf = num_pixels_uninf + (Inf_mask == 1).sum()
                num_pixels_inf = num_pixels_inf + (Inf_mask == 2).sum()
                num_pixels = num_pixels + Z * X * Y

    pixel_ratio_1 = [
        num_pixels / num_pixels_bkg,
        num_pixels / num_pixels_uninf,
        num_pixels / num_pixels_inf,
    ]
    pixel_ratio_sum = sum(pixel_ratio_1)
    pixel_ratio = [ratio / pixel_ratio_sum for ratio in pixel_ratio_1]

    return pixel_ratio