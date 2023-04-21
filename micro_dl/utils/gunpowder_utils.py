import glob
import gunpowder as gp
import os
import pathlib
import random
import re
import zarr

import iohub.ngff as ngff
import micro_dl.input.gunpowder_nodes as nodes
import micro_dl.utils.augmentation_utils as aug_utils
import micro_dl.utils.io_utils as io_utils



def gpsum(nodelist, verbose=True):
    """
    Interleaves printing nodes in between nodes listed in nodelist.
    Returns pipeline of nodelist. If verbose set to true pipeline will print
    call sequence information upon each batch request.

    :param list nodelist: list of nodes to construct pipeline from
    :param bool verbose: whether to include gpprint notes, defaults to True
    :return gp.Node pipeline: gunpowder data pipeline
    """
    pipeline = nodelist.pop(0)
    prefix = 0
    while len(nodelist) > 0:
        pipeline += nodelist.pop(0)
        if verbose:
            pipeline += nodes.LogNode(str(prefix), time_nodes=verbose)
            prefix += 1
    return pipeline


def multi_zarr_source(
    zarr_dir,
    array_spec,
    data_split={},
    use_recorded_split=False,
):
    """
    Generates a tuple of source nodes for for each dataset type (train, test, val),
    containing one source node for every well in the zarr_dir specified.

    Applies same specification to all source datasets. Note that all source datasets of the
    same name exhibit **key sharing**. That is, the source key will be the same for all datasets
    of name 'arr_0' (for example) and a different source key will be shared amongst
    'arr_0_preprocessed' sources. This feature is only relevant if array_name matches
    multiple arrays in the specified zarr stores.

    Note: The zarr store in 'zarr_dir' must have the _same number of array types_. This is to
    enable key sharing, which is necessary for the RandomProvider node to be able to utilize all
    positions.

    :param str zarr_dir: path to HCS-compatible zarr store.
    :param gp.ArraySpec array_spec: specification for zarr datasets, defaults to None
    :param dict data_split: dict containing fractions  to split data for train, test, validation.
                            keys must be 'train', 'test', and 'val'. By default does not split
                            and returns one source tuple. Is overridden by "use_recorded_split".
    :param bool use_recorded_split: if true, will use recorded data split given in 'data_split'.

    :return tuple all_sources: (if no data_split) multi-source node from zarr_dir stores (equivalent to
                            s_1 + ... + s_n)
    :return tuple train_sources: (if data_split) random subset of sources for training
    :return tuple test_sources: (if data_split) random subset of sources for testing
    :return tuple val_sources: (if data_split) random subset of sources for validation
    :return list all_keys: list of shared keys for each dataset type across all source subsets.
    """
    plate = ngff.open_ome_zarr(zarr_dir)
    position_paths, positions = list(zip(*list(plate.positions())))

    # all positions must contain the same array names
    ar_names_at_position = None
    for i,position_path in enumerate(position_paths):
        ar_names = list(zarr.open(os.path.join(zarr_dir, position_path)).array_keys())
        if ar_names_at_position is not None:
            assert ar_names_at_position == ar_names, (
                f"Found different set of array names at positions {position_paths[i]}"
                f" and {position_paths[i-1]}"
            )
    
    # Construct one ZarrSource for each position. Sources share the same set of keys
    sources, keys = build_sources(
        zarr_dir = zarr_dir,
        position_paths = position_paths,
        arr_spec = array_spec,
    )

    # partition the sources and return
    if len(data_split) > 0 and use_recorded_split == False:
        assert "train" in data_split and "test" in data_split and "val" in data_split, (
            f"Incorrect format for data_split: {data_split}."
            " \n Must contain 'train', 'test', and 'val' "
        )

        # randomly generate split
        random.shuffle(sources)
        train_idx = int(len(sources) * data_split["train"])
        test_idx = int(len(sources) * (data_split["train"] + data_split["test"]))
        val_idx = len(sources)

        train_sources = tuple(sources[0:train_idx])
        test_sources = tuple(sources[train_idx:test_idx])
        val_sources= tuple(sources[test_idx:val_idx])

        # return positions of each source with their data split
        data_split = {}
        split = ["train", "test", "val"]
        for i, source_list in enumerate([train_sources, test_sources, val_sources]):
            positions = list(map(validate_source_positions, source_list))
            data_split[split[i]] = positions

        return train_sources, test_sources, val_sources, keys, data_split

    elif use_recorded_split:
        # use split provided in data_split
        assert "train" in data_split and "test" in data_split and "val" in data_split, (
            f"Incorrect format for data_split: {data_split}."
            " \n Must contain 'train', 'test', and 'val' "
        )

        # map each source to its position
        train_sources, test_sources, val_sources = [], [], []
        for i, source in enumerate(sources):
            source_path = validate_source_positions(source)
            if source_path in data_split["train"]:
                train_sources.append(source)
            elif source_path in data_split["test"]:
                test_sources.append(source)
            elif source_path in data_split["val"]:
                val_sources.append(source)

        train_sources = tuple(train_sources)
        test_sources = tuple(test_sources)
        val_sources = tuple(val_sources)

        return train_sources, test_sources, val_sources, keys, data_split
    else:
        source = tuple(sources)
        return source, keys


def build_sources(zarr_dir, position_paths, arr_spec):
    """
    Builds a source for every well_path (position) in a zarr store, specified by
    position_paths and zarr_dir, and returns each source.

    The sources will have a different key for each array type at each well.
    For example, if your wells contain:
        |- arr_0
        |- arr_1
        .
        .
        |- arr_n

    This method will build a source for each well, each of which can be accessed by a
    any of a list corresponding of gp.ArrayKey keys that are returned in order:

        [keys_0, keys_1, ... , key_n]

    The keys used to access each store map to the corresponding array type.

    Note: the implication with this implementation is that all wells contain the same
    array types (and number of array types). This enforces no use of non-uniform
    numbers of array types across a single store.

    :param str zarr_dir: path to zarr directory to build sources for
    :param list position_paths: list of paths inside zarr store to positions
    :param gp.ArraySpec arr_spec: ArraySpec pertaining to datasets (supports one global spec)

    :return list sources: dictionary of datasets locations and corresponding arraykeys
    :return list keys: list of ArrayKeys for each array type, shared across sources
    """
    sources = []
    for position_path in position_paths:
        
        # Note that as we generate every 'name', the names are the relative paths
        ar_names = list(zarr.open(os.path.join(zarr_dir, position_path)).array_keys())
        keys = [gp.ArrayKey(name) for name in ar_names]
        
        dataset_dict = {}
        spec_dict = {}
        for i, key in enumerate(keys):
            dataset_dict[key] = os.path.join(position_path, ar_names[i])
            spec_dict[key] = arr_spec
        
        # Source is like a dictionary: each ArrayKey -> {spec (metadata), dataset (array)}
        source = gp.ZarrSource(
            filename=zarr_dir,
            datasets=dataset_dict,
            array_specs=spec_dict,
        )
        sources.append(source)
    return sources, keys


def validate_source_positions(zarr_source):
    """
    Requires that all datasets inside this zarr_source refer to the same
    position

    This method retrieves the position path by scraping the path. The total path.

    :param gp.ZarrSource zarr_source: source that refers to a single position in an
                                HCS compatible zarr store
    """
    position_path = None
    for key in zarr_source.datasets:
        #remove the array name
        path = "/".split(zarr_source.datasets[key])[:-1]
        if position_path:
            assert path == position_path, (
                "Found two datasets with different positions",
                f"in the same source: {path} and {position_path}",
            )
        position_path = path

    return position_path

def generate_array_spec(network_config):
    """
    Generates an array_spec for the zarr source data based upon the model used in
    training (as specified in network config)

    :param network_config: config dictionary containint
    :returns gp.ArraySpec array_spec: ArraySpec metadata compatible with given config
    """
    assert (
        "architecture" in network_config
    ), f"no 'architecture' specified in network config"

    arch = network_config["architecture"]
    if arch not in {"2D", "2.5D"}:
        raise AttributeError(
            f"Architecture {network_config['architecture']} not supported"
        )
    voxel_size = (1, 1, 1)
    array_spec = gp.ArraySpec(
        interpolatable=True,
        voxel_size=voxel_size,
    )

    return array_spec


def generate_augmentation_nodes(aug_config, augmentation_keys):
    """
    Returns a list of augmentation nodes as specified in 'aug_config'.
    Return order is insensitive to the order of creation in augmentation_config...
    Augmentations can be given in any order, they will always be returned in a
    compatible sequence.

    :param augmentation_config: dict of augmentation type -> hyperparameters,
                                see torch_config readme.md for more details
    :return list aug_nodes: list of augmentation nodes in order
    """
    augmentation_builder = aug_utils.AugmentationNodeBuilder(
        aug_config,
        noise_key=augmentation_keys,
        blur_key=augmentation_keys,
        intensities_key=augmentation_keys,
        defect_key=augmentation_keys,
        shear_key=augmentation_keys,
    )
    augmentation_builder.build_nodes()
    aug_nodes = augmentation_builder.get_nodes()

    return aug_nodes
