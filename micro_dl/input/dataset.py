
import gunpowder as gp
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

import micro_dl.input.gunpowder_nodes as custom_nodes
import micro_dl.utils.gunpowder_utils as gp_utils
from micro_dl.utils.aux_utils import ToTensor

class TorchDatasetContainer(object):
    """
    Dataset container object which initalizes multiple TorchDatasets depending upon the parameters
    given in the training and network config files during initialization.

    By default, randomly selects samples to divy up sources between train, test, and validation datasets.
    """

    def __init__(
        self,
        zarr_dir,
        train_config,
        network_config,
        dataset_config,
        device=None,
        workers=0,
        use_recorded_split=False,
        data_split={},
        allow_augmentations=True,
    ):
        """
        Inits an object which builds a testing, training, and validation dataset
        from .zarr data files using gunpowder pipelines

        If acting as a container object:
        :param dict train_config: dict object of train_config
        :param dict network_config: dict object of network_config
        :param dict dataset_config: dict object of dataset_config
        :param str device: device on which to place tensors in child datasets,
                            by default, places on 'cuda'
        :param int workers: number of cpu workers for simultaneous data fetching
        :param bool use_recorded_split: whether to use random or previously recorded
                            train/test/val data split
        :param dict data_split: dictionary of data split containing integer list of positions
                            OR decimal fractions indicating split under {'train', 'test', 'val'}
                            keys
        :param bool allow_augmentations: If False disallows augmentations in dataloading
                            pipelines, by default True
        """
        self.zarr_dir = zarr_dir
        self.train_config = train_config
        self.network_config = network_config
        self.dataset_config = dataset_config
        self.device = device
        self.workers = workers
        self.data_split = data_split

        # acquire sources from the zarr directory
        array_spec = gp_utils.generate_array_spec(network_config)
        (
            self.train_source,
            self.test_source,
            self.val_source,
            self.dataset_keys,
            self.data_split,
        ) = gp_utils.multi_zarr_source(
            zarr_dir=self.zarr_dir,
            array_spec=array_spec,
            data_split=self.data_split,
            use_recorded_split=use_recorded_split,
        )
        try:
            assert len(self.test_source) > 0
            assert len(self.train_source) > 0
            assert len(self.val_source) > 0
        except Exception as e:
            raise AssertionError(
                f"All datasets must have at least one source node / zarr store,"
                f" not enough source arrays found.\n {e.args}"
            )
        self.mask_key = None
        mask_key_identifier = ""
        if "mask_type" in dataset_config:
            mask_key_identifier = "_".join(["mask", dataset_config["mask_type"]])

        for key in self.dataset_keys.copy():
            if key.identifier == mask_key_identifier:
                self.mask_key = key
                self.dataset_keys.remove(self.mask_key)
            elif "mask" in key.identifier:
                self.dataset_keys.remove(key)
        if "mask_type" in dataset_config and self.mask_key == None:
            raise ValueError(f"Specified mask type's corresponding array not found")

        # build augmentation nodes if allowed
        train_aug_nodes = []
        test_aug_nodes = []
        val_aug_nodes = []
        if "augmentations" in train_config and allow_augmentations:
            assert isinstance(train_config["augmentations"], dict), "Augmentations"
            " section of config must be a dictionary of parameters"
            train_aug_nodes = gp_utils.generate_augmentation_nodes(
                train_config["augmentations"], self.dataset_keys
            )
            test_aug_nodes = gp_utils.generate_augmentation_nodes(
                train_config["augmentations"], self.dataset_keys
            )
            val_aug_nodes = gp_utils.generate_augmentation_nodes(
                train_config["augmentations"], self.dataset_keys
            )

        # set up epoch length for each dataset type or allow for later calculation
        if "samples_per_epoch" in self.train_config:
            train_epoch_length = self.train_config["samples_per_epoch"]
        else:
            train_epoch_length = 0
        train_fraction = self.dataset_config["split_ratio"]["train"]
        total = train_epoch_length // train_fraction

        test_epoch_length = int(self.dataset_config["split_ratio"]["test"] * total)
        val_epoch_length = int(self.dataset_config["split_ratio"]["val"] * total)

        # assign each source subset to a child dataset object
        self.train_dataset = self.init_torch_dataset(
            self.train_source, train_aug_nodes, train_epoch_length
        )
        self.test_dataset = self.init_torch_dataset(
            self.test_source, test_aug_nodes, test_epoch_length
        )
        self.val_dataset = self.init_torch_dataset(
            self.val_source, val_aug_nodes, val_epoch_length
        )

    def init_torch_dataset(self, source, augmentation_nodes, dataset_epoch_length):
        """
        Initializes a torch dataset to sample 'source' tuple through the given
        augmentations 'augmentations'.

        :param tuple(gp.ZarrSource) source: tuple of source nodes representing the
                                            dataset sample space
        :param list augmentation_nodes: list of augmentation nodes in order
        :param int dataset_epoch_length: length of epoch for this dataset.
        """
        # NOTE: not passing the whole dataset config is a design choice here. The
        # elements of the config are a black box until theyre indexed. I do this
        # to make them more readable. This can change with PyDantic later
        dataset = TorchDataset(
            # gunpowder params
            data_source=source,
            augmentation_nodes=augmentation_nodes,
            data_keys=self.dataset_keys,
            # preprocessing node params
            mask_key=self.mask_key,
            normalization_scheme=self.dataset_config["normalization"]["scheme"],
            normalization_type=self.dataset_config["normalization"]["type"],
            min_foreground_fraction=self.dataset_config["min_foreground_fraction"],
            # dataloading params
            data_dimensionality=self.network_config["architecture"],
            batch_size=self.dataset_config["batch_size"],
            epoch_length=dataset_epoch_length,
            target_channel_idx=tuple(self.dataset_config["target_channels"]),
            input_channel_idx=tuple(self.dataset_config["input_channels"]),
            spatial_window_size=tuple(self.dataset_config["window_size"]),
            spatial_window_offset=(0,) * len(tuple(self.dataset_config["window_size"])),
            device=self.device,
            workers=self.workers,
            random_sampling=True,
        )
        return dataset

    def __getitem__(self, idx):
        """
        Provides indexing capabilities to reference train, test, and val datasets

        :param int or str idx: index/key of dataset to retrieve:
                                train -> 0 or 'train'
                                test -> 1 or 'test'
                                val -> 2 or 'val'
        """
        if isinstance(idx, str):
            return {
                "train": self.train_dataset,
                "test": self.test_dataset,
                "val": self.val_dataset,
            }[idx]
        else:
            return [self.train_dataset, self.test_dataset, self.val_dataset][idx]


class TorchDataset(Dataset):
    """
    Based off of torch.utils.data.Dataset:
        - https://pytorch.org/docs/stable/data.html

    Custom dataset class that builds gunpowder pipelines composed of multi-zarr source nodes
    and a series of augmentation nodes. This object will call from the gunpowder pipeline directly,
    and transform resulting data into tensors to be placed onto the gpu for processing.

    Multiprocessing is supported with num_workers > 0. However, there are non-fatal warnings about
    "...processes being terminated before shared CUDA tensors are released..." with torch 1.10.

    These warnings are discussed on the following post, and I believe have been since fixed:
        - https://github.com/pytorch/pytorch/issues/71187
    """

    def __init__(
        self,
        data_source,
        augmentation_nodes,
        data_keys,
        mask_key,
        normalization_scheme,
        normalization_type,
        min_foreground_fraction,
        data_dimensionality,
        batch_size,
        epoch_length,
        target_channel_idx,
        input_channel_idx,
        spatial_window_size,
        spatial_window_offset,
        device,
        workers,
        random_sampling,
    ):
        """
        Creates a dataset object which draws samples directly from a gunpowder pipeline.

        :param tuple(gp.ZarrSource) source: tuple of source nodes representing the
                                            dataset sample space
        :param list augmentation_nodes: list of augmentation nodes in order
        :param tuple data_source: tuple of gp.Source nodes which the given pipeline draws samples
        :param list data_keys: list of gp.ArrayKey objects which access the given source
        :param gp.ArrayKey mask_key: key to untracked mask array in source node, if applicable
        :param str normalization_scheme: see name, one of {"dataset", "FOV", "tile"}
        :param str normalization_type: see name, one of {"median_and_iqr", "mean_and_std"}
        :param float min_foreground_fraction: minimum foreground required to be present in sample
                                    for region to be selected, must be within [0, 1]
        :param str data_dimensionality: whether to collapse the first channel of 3d data,
                                    one of {2D, 2.5D}
        :param int batch_size: number of samples per batch
        :param tuple(int) target_channel_idx: indices of target channel(s) within zarr store
        :param tuple(int) input_channel_idx: indices of input channel(s) within zarr store
        :param tuple spatial_window_size: tuple of sample dimensions, specifies batch request ROI
                                    expects 3D tuple; dimensions zyx, where z = 1 for 2d request
        :param tuple spatial_window_offset: tuple of offset dimensions, specifies batch request ROI
                                    expects 3D tuple; dimensions zyx, where z = 1 for 2d request
        :param str device: device on which to place tensors in child datasets,
                                    by default, places on 'cuda'
        :param int workers: number of simultaneous threads reading data into batch requests
        :param bool random_sampling: whether to sample source data randomly or not.
        """
        self.data_source = data_source
        self.augmentation_nodes = augmentation_nodes
        self.data_keys = data_keys
        self.mask_key = mask_key
        self.normalization_scheme = normalization_scheme
        self.normalization_type = normalization_type
        self.min_foreground_fraction = min_foreground_fraction
        self.data_dimensionality = data_dimensionality
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.target_idx = target_channel_idx
        self.input_idx = input_channel_idx
        self.window_size = spatial_window_size
        self.window_offset = spatial_window_offset
        self.device = device
        self.active_key = 0
        self.workers = max(1, workers - 1)
        self.random_sampling = random_sampling

        # safety checks: iterate through keys and data sources to ensure that they match
        voxel_size = None
        for key in self.data_keys:
            # check that all data voxel sizes are the same, exclude masks
            if not "mask" in key.identifier:
                for i, source in enumerate(self.data_source):
                    try:
                        array_spec = source.array_specs[key]
                        if len(array_spec.voxel_size) == 2: # for 2D models
                            array_spec.voxel_size = (1,)+array_spec.voxel_size
                        if not voxel_size:
                            voxel_size = array_spec.voxel_size
                        else:
                            assert (
                                voxel_size == array_spec.voxel_size
                            ), f"Voxel size of array {array_spec.voxel_size} does not match"
                            f" voxel size of previous array {voxel_size}."
                    except Exception as e:
                        raise AssertionError(
                            f"Error matching keys to source in dataset: {e.args}"
                        )

        # calculate epoch length: if no epoch length specified, set according to data size
        self._calculate_epoch_length()

        # construct pipeline: generate batch request, construct nodes, make iterable
        assert len(self.window_size) == 3, (
            f"Window size {self.window_size} must be 3-dimensional. If 2D data, "
            "desired, make the first dimension of window size 1; (1, X, Y)."
        )
        assert len(self.window_size) == len(voxel_size), (
            f"Incompatible voxel size {voxel_size}. "
            f"Must be same length as spatial_window_size {self.window_size}."
        )
        assert len(self.window_size) == len(self.window_offset), (
            f"Incompatible window offset {self.window_offset}. "
            f"Must be same length as spatial_window_size {self.window_size}."
        )
        self.batch_request = self._generate_batch_request(
            window_offset=self.window_offset,
            window_size=self.window_size,
        )
        self.preprocessing_nodes = self._generate_preprocessing_nodes()
        self.pipeline = self._build_pipeline()

        self.batch_generator = self._generate_batches()

    def _calculate_epoch_length(self, oversample_factor=2):
        """
        Calculates the epoch length that should be used for this dataset based upon the
        tile window size and the size of the data. Assumes that sampling will be uniformly
        random over the spatial data, and that we will oversample our data to the specified
        factor (default of 2).

        :param int oversample_factor: factor with which we will oversample our data to ensure
                            that data is given a fair chance at being at the "center" of a tile
        """
        if self.epoch_length == 0 or self.epoch_length == None:
            source = self.data_source[0]
            for key in source.datasets:
                z1 = zarr.open(source.filename)
                fov_shape = z1[source.datasets[key]][:].shape
                break
            data_spatial_dims = fov_shape[-len(self.window_size) :]

            # assuming stride length providing oversampling, calculate # theoretical tiles/fov
            samples_per_fov = 1
            for dim_index in range(len(self.window_size)):
                data_length = data_spatial_dims[dim_index]
                if len(self.window_size) == 3 and dim_index == 0:  # if z
                    samples_per_fov *= data_length - (self.window_size[dim_index] - 1)
                else:  # if x or y
                    samples_per_fov *= (
                        (data_length // self.window_size[dim_index]) * oversample_factor
                    ) - 1

            self.epoch_length = int(samples_per_fov * len(self.data_source))
            self.epoch_length = self.epoch_length // self.batch_size

    def _generate_batch_request(self, window_offset, window_size):
        """
        Generates a series of batch requests according to a window offset and window size

        :param tuple(int, int, int) window_offset: offset at start of window
        :param tuple(int, int, int) window_size: size of dimensions of window

        :return gp.BatchRequest() batch_request: batch request corresponding to given window
        """
        batch_request = gp.BatchRequest()

        for key in self.data_keys:
            batch_request[key] = gp.Roi(window_offset, window_size)
            # NOTE: the keymapping we are performing here makes it so that if
            # we DO end up generating multiple arrays at the position/well level,
            # we can access all of them by requesting that key. The index we request
            # in __getitem__ ends up being the index of our key.
        if self.mask_key:
            batch_request[self.mask_key] = gp.Roi(window_offset, window_size)

        return batch_request

    def __len__(self):
        """
        Returns number of source data arrays in this dataset.
        """
        return self.epoch_length

    def __getitem__(self, idx=0):
        """
        Requests a batch from the data pipeline using the key selected by self.use_key,
        and applying that key to call a batch from its corresponding source using
        self.batch_request.
        """
        assert self.active_key != None, "No active key. Try '.use_key()'"

        # TODO: this implementation pulls 5 x 256 x 256 of all channels. We may not want
        # all of those slices in all channels if theyre note being used. Fix this inefficiency

        sample = next(self.batch_generator)
        sample_data = sample[self.active_key].data

        # NOTE We assume the .zarr ALWAYS has an extra batch channel.
        # SO, 3d -> 5d data, 2d -> 4d data

        # remove extra dimension from stack node, if 2d remove z dimension
        sample_data = sample_data[:, 0, ...]
        if self.window_size[0] == 1 and self.data_dimensionality == "2D":
            sample_data = sample_data[..., 0, :, :]

        # stack multiple channels
        full_input = []
        for idx in self.input_idx:  # assumes bczyx or bcyx
            channel_input = sample_data[:, idx, ...]
            full_input.append(channel_input)
        full_input = np.stack(full_input, 1)

        full_target = []
        for idx in self.target_idx:  # assumes bczyx or bcyx
            channel_target = sample_data[:, idx, ...]
            full_target.append(channel_target)
        full_target = np.stack(full_target, 1)

        if len(full_target.shape) == 5:
            # target is always 2 dimensional, we select middle z dim
            middle_z_idx = full_target.shape[-3] // 2
            full_target = np.expand_dims(full_target[..., middle_z_idx, :, :], -3)

        # convert to tensor and place onto gpu
        convert = ToTensor(self.device)
        input_, target_ = convert(full_input), convert(full_target)

        return (input_, target_)

    def use_key(self, selection):
        """
        Sets self.active_key to selection if selection is an ArrayKey. If selection is an int,
        sets self.active_key to the index in self.data_keys given by selection.

        :param int or gp.ArrayKey selection: key index of key in self.data_keys to activate
        """
        if isinstance(selection, int):
            try:
                self.active_key = self.data_keys[selection]
            except IndexError as e:
                raise IndexError(
                    f"Handling exception {e.args}: index of selection"
                    " must be within length of data_keys"
                )
        elif isinstance(selection, gp.ArrayKey):
            # TODO change this assertion to reference the source-> arrayspec-> keys
            assert (
                selection in self.data_keys
            ), "Given key not associated with dataset sources"
            self.active_key = selection
        else:
            raise AttributeError("Selection must be int or gp.ArrayKey")

    def _build_pipeline(self):
        """
        Builds a gunpowder data pipeline given a source node and a list of augmentation
        nodes.

        :param gp.ZarrSource or tuple source: source node/tuple of source nodes from which to draw
        :param list nodes: list of augmentation nodes, by default empty list

        :return gp.Pipeline pipeline: see name
        """
        # ---- Sourcing Nodes ----#
        # if source is multi_zarr_source, attach a RandomProvider
        if self.random_sampling:
            source = self.data_source
            if isinstance(source, tuple):
                source = [source, gp.RandomProvider()]
            source = source + [gp.RandomLocation()]
        else:
            source = [self.data_source[0]]
        # NOTE: not random sampling with multiple providers only calls from the first

        if self.min_foreground_fraction and self.mask_key:
            source = source + [
                gp.Reject(
                    mask=self.mask_key,
                    min_masked=self.min_foreground_fraction,
                ),
                custom_nodes.PrepMaskRoi(self.data_keys, self.mask_key),
            ]

        # ---- Preprocessing Nodes ----#
        source = source + self.preprocessing_nodes

        # ---- Batch Creation Nodes ----#
        batch_creation = []
        batch_creation.append(
            gp.PreCache(
                cache_size=self.epoch_length // 10,
                num_workers=max(1, self.workers),
            )
        )
        batch_creation.append(gp.Stack(self.batch_size))

        # attach additional nodes, if any, and sum
        pipeline = source + self.augmentation_nodes + batch_creation
        pipeline = gp_utils.gpsum(pipeline, verbose=False)
        return pipeline

    def _generate_batches(self):
        """
        Returns pipeline as a generator. This is done in a separate method from __getitem__()
        to preserve compatibility with the torch dataloader's item calling signature
        while also performing appropriate context management for the pipeline via generation.
        See:
            https://github.com/funkey/gunpowder/issues/181

        :param gp.Pipeline pipeline: pipeline to generate batches from
        :param gp.BatchRequest request: batch request for pipeline

        :yield gp.Batch batch: single batch yielded from pipeline at each generation
        :rtype: Iterator[gp.Batch]
        """
        with gp.build(self.pipeline):
            while True:
                yield self.pipeline.request_batch(self.batch_request)
                # batch = self.pipeline.request_batch(self.batch_request)
                # self.pipeline.internal_teardown()
                # yield batch

    def _generate_preprocessing_nodes(self):
        """
        Returns a list of preprocessing nodes configured according to the current state of
        this object.

        :return list preprocessing_nodes: see name
        """
        preprocessing_nodes = []

        if self.normalization_scheme and self.normalization_type:
            normalize = custom_nodes.Normalize(
                array=self.data_keys,
                scheme=self.normalization_scheme,
                type=self.normalization_type,
            )
            preprocessing_nodes.append(normalize)

        return preprocessing_nodes


class DatasetEnsemble(TorchDataset):
    def __init__(self, torch_datasets):
        """
        Groups the datasets in torch_datasets together, pulling from each of them
        individually.

        Used instead of ConcatDataset for context-management issues with gunpowder
        generator pipeline

        :param list torch_datasets: list of TorchDataset objects to sample from
        :return: object instance
        """
        self.datasets = torch_datasets
        self.last_pipeline = None

    def __getitem__(self, idx):
        """
        Returns one sample by calling the __getitem__ of the subsequent TorchDataset at
        index idx.

        :param int idx: index of dataset to call sample from in torch_datasets
        :return: item from __getitem__ of torch_datasets[idx]
        """
        dataset = self.datasets[idx]
        self._check_source_uniqueness(dataset)

        return dataset.__getitem__()

    def _check_source_uniqueness(self, dataset):
        dataset_pipeline = dataset.pipeline
        if not self.last_pipeline:
            self.last_pipeline = dataset_pipeline
        else:
            pipeline_1 = self.last_pipeline
            pipeline_2 = dataset_pipeline

            assert (
                pipeline_1.output is not pipeline_2.output
            ), f"Two nodes shared: {pipeline_1.output} is {pipeline_2.output}"

            while len(pipeline_1.children) > 0:
                pipeline_1 = pipeline_1.children[0]
                pipeline_2 = pipeline_2.children[0]

                assert (
                    pipeline_1.output is not pipeline_2.output
                ), f"Two nodes shared: {pipeline_1.output} is {pipeline_2.output}"

            self.last_pipeline = dataset_pipeline

    def __len__(self):
        return len(self.datasets)
