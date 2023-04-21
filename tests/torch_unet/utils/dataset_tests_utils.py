import os
import zarr


def build_zarr_store(self, temp, arr_spatial, num_stores=1):
    # TODO rewrite using io_utils.
    """
    Builds a test zarr store conforming to OME-NGFF Zarr format with 5d arrays
    in the directory 'temp'

    :param str temp: dir path to build zarr store in
    :param str zarr_name: name of zarr store inside temp dir (discluding extension)
    :param int num_stores: of zarr_stores to build
    :param tuple arr_spatial: spatial dimensions of data
    :raises FileExistsError: cannot overwrite a currently written directory, so
                            temp must be a new directory
    """
    try:
        os.makedirs(temp, exist_ok=True)
    except Exception as e:
        raise FileExistsError(f"parent directory cannot already exist {e.args}")

    def recurse_helper(group, names, subgroups, depth):
        """
        Recursively makes heirarchies of 'num_subgroups' subgroups until 'depth' is reached
        as children of 'group', filling the bottom most groups with arrays of value
        arr_value, and incrementing arr_value.

        :param zarr.heirarchy.Group group: Parent group ('.zarr' store)
        :param list names: names subgroups at each depth level + [name of arr]
        :param int subgroups: number of subgroups at each level (width)
        :param int depth: height of subgroup tree (height)
        :param int ar_channels: number of channels of data array
        :param int depth: window size of data array
        """
        if depth == 0:
            for j in range(subgroups):
                z1 = zarr.open(
                    os.path.join(
                        group.store.dir_path(),
                        group.path,
                        names[-depth - 1] + f"_{j}",
                    ),
                    mode="w",
                    shape=([1, arr_channels] + [dim * 2 for dim in arr_spatial]),
                    chunks=([1, 1] + list(arr_spatial)),
                    dtype="float32",
                )
                val = arr_value.pop(0)
                z1[:] = val
                arr_value.append(val + 1)
        else:
            for j in range(subgroups):
                subgroup = group.create_group(names[-depth - 1] + f"_{j}")
                recurse_helper(subgroup, names, subgroups, depth - 1)

    # set parameters for store creation
    max_input_channels = 0
    max_target_channels = 0
    for config in self.all_dataset_configs:
        max_input_channels = max(len(config["input_channels"]), max_input_channels)
        max_target_channels = max(len(config["target_channels"]), max_target_channels)
    self.num_channels = max_input_channels + max_target_channels

    # build stores
    self.groups = []
    for i in range(num_stores):
        store_path = os.path.join(temp, f"example_{i}.zarr")
        store = zarr.DirectoryStore(store_path)
        g1 = zarr.group(store=store, overwrite=True)
        self.groups.append(g1)

        arr_value = [0]
        arr_channels = self.num_channels

        recurse_helper(g1, ["Row", "Col", "Pos", "arr"], 3, 3)
    self.zarr_dir = store_path
