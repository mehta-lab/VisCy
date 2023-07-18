## Preprocessing

The main command for preprocessing is:

```buildoutcfg
python viscy/cli/preprocess_script.py --config <config path (.yml)>
```

The following settings can be adjusted in preprocessing using a config file (see example in preprocess_config.yml):

* input_dir: (str) Directory where data to be preprocessed is located
* output_dir: (str) folder name where all processed data will be written
* channel_ids: (list of ints) specify channel numbers (default is -1 for all indices)
* num_workers: (int) Number of workers for multiprocessing
* slice_ids: (int/list) Value(s) of z-indices to be processed (default is -1 for all indices)
* time_ids: (int/list) Value(s) of timepoints to be processed (default is -1 for all indices)
* pos_ids: (int/list) Value(s) of FOVs/positions to be processed (default is -1 for all indices)
* verbose: (int) Logging verbosity levels: NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50
* resize:
  * scale_factor(float/list): Scale factor for resizing 2D frames, e.g. to match resolution in z or resizing volumes
  * num_slices_subvolume (int): number of slices to be included in each subvolume, default=-1, includes all slices in           slice_ids

* masks:
  * channels: (list of ints) which channels should be used to generate masks from
  * str_elem_radius: (int) structuring element radius for morphological operations on masks
  * normalize_im (bool): Whether to normalize image before generating masks
  * mask_dir (str): As an alternative to channels/str_element_radius, you can specify a directory
    containing already generated masks (e.g. manual annotations). Masks must match input images in
    terms of shape and indices.
  * csv_name (str): If specifying mask_dir, the directory must contain a csv file matching mask names
    with image names. If left out, the script will look for first a frames_meta.csv,
    second one csv file containing mask names in one column and matched image names in a
    second column.
* do_tiling: (bool) do tiling (recommended)
* tile:
  * tile_size: (list of ints) tile size in pixels for each dimension
  * step_size: (list of ints) step size in pixels for each dimension
  * depths: (list of ints) tile z depth for all the channels specified
  * mask_depth: (int) z depth of mask
  * image_format (str): 'zyx' (default) or 'xyz'. Order of tile dimensions
  * train_fraction (float): If specified in range (0, 1), will randomly select that fraction
    of training data in each epoch. It will update steps_per_epoch in fit_generator accordingly.
  * min_fraction: (float) minimum fraction of image occupied by foreground in masks
  * hist_clip_limits: (list of ints) lower and upper intensity percentiles for histogram clipping

The tiling class will take the 2D image files, assemble them to stacks in case 3D tiles are required,
and store them as tiles based on input tile size, step size, and depth.

All data will be stored in the specified output dir, where a 'preprocessing_info.json' file

During preprocessing, a csv file named frames_csv.csv will be generated, which
will be used for further processing. The csv contains the following fields for each image tile:

* 'time_idx': the timepoint it came from
* 'channel_idx': its channel
* 'slice_idx': the z index in case of 3D data
* 'pos_idx': the field of view index
* 'file_name': file name
* 'row_start': starting row for tile (add tile_size for endpoint)
* 'col_start': start column (add tile_size for endpoint)
