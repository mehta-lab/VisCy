import cv2
import collections
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

import micro_dl.utils.aux_utils as aux_utils
import micro_dl.utils.masks as mask_utils
import micro_dl.utils.normalize as norm_utils
import micro_dl.utils.train_utils as train_utils


class TorchDataset(Dataset):
    """
    Based off of torch.utils.data.Dataset:
        - https://pytorch.org/docs/stable/data.html

    Custom dataset class that draws samples from a  'micro_dl.input.dataset.BaseDataSet' object,
    and converts them to PyTorch inputs.

    Also takes lists of transformation classes. Transformations should be primarily designed for
    data refactoring and type conversion, since augmentations are already applied to tensorflow
    BaseDataSet object.

    The dataset object will cache samples so that the processing required to produce samples
    does not need to be repeated. This drastically speeds up training time.

    Multiprocessing is supported with num_workers > 0. However, there is a non-fatal warning about
    "...processes being terminated before shared CUDA tensors are released..." with torch 1.10.

    They are discussed on the following post, and I believe have been since fixed:
        - https://github.com/pytorch/pytorch/issues/71187
    """

    def __init__(
        self,
        train_config=None,
        tf_dataset=None,
        transforms=None,
        target_transforms=None,
        caching=False,
        device=torch.device("cpu"),
        meta_dir="",
    ):
        """
        Init object.
        Params:
        :param str train_config -> str: path to .yml config file from which to create BaseDataSet
                                        object if none given
        :param micro_dl.input.dataset.BaseDataSet tf_dataset: tensorflow-based dataset
                                                                containing samples to convert
        :param iterable(Transform object) transforms: transforms to be applied to every sample
                                                      *after tf_dataset transforms*
        :param iterable(Transform object) target_transforms: transforms to be applied to every
                                                            target *after tf_dataset transforms*
        :param str device: device name, example: 'cuda:0' for gpu 0, 'cpu' for cpu.
        :param str meta_dir: directory to save dataset selection metadata in
        """
        assert (
            train_config or tf_dataset
        ), "Must provide either train config file or tf dataset"

        self.tf_dataset = None
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.caching = caching
        self.device = device

        if tf_dataset != None:
            self.tf_dataset = tf_dataset
            self.sample_cache = collections.defaultdict()

            self.train_dataset = None
            self.test_dataset = None
            self.val_dataset = None
            self.mask = True
            self.last_idx = -1
        else:
            config = aux_utils.read_config(train_config)

            dataset_config, trainer_config = config["dataset"], config["trainer"]
            tile_dir, image_format = train_utils.get_image_dir_format(dataset_config)

            tiles_meta = pd.read_csv(os.path.join(tile_dir, "frames_meta.csv"))
            tiles_meta = aux_utils.sort_meta_by_channel(tiles_meta)

            masked_loss = False

            all_datasets, split_samples = train_utils.create_train_datasets(
                tiles_meta,
                tile_dir,
                dataset_config,
                trainer_config,
                image_format,
                masked_loss,
                meta_dir,
            )

            self.train_dataset = TorchDataset(
                None,
                tf_dataset=all_datasets["df_train"],
                transforms=self.transforms,
                target_transforms=self.target_transforms,
                device=self.device,
            )
            self.test_dataset = TorchDataset(
                None,
                tf_dataset=all_datasets["df_test"],
                transforms=self.transforms,
                target_transforms=self.target_transforms,
                device=self.device,
            )
            self.val_dataset = TorchDataset(
                None,
                tf_dataset=all_datasets["df_val"],
                transforms=self.transforms,
                target_transforms=self.target_transforms,
                device=self.device,
            )

            self.split_samples_metadata = split_samples

    def __len__(self):
        """
        Returns number of sample (or sample stack)/target pairs in dataset
        """
        if self.tf_dataset:
            return len(self.tf_dataset)
        else:
            return sum(
                [
                    1 if self.train_dataset else 0,
                    1 if self.test_dataset else 0,
                    1 if self.val_dataset else 0,
                ]
            )

    def __getitem__(self, idx):
        """
        If acting as a dataset object, returns the sample target pair at 'idx'
        in dataset, after applying augment/transformations to sample/target pairs.

        If acting as a dataset container object, returns subsidary dataset
        objects.

        :param int idx: index of dataset item to transform and return
        """
        # remove the tensor accessed last time from cuda memory
        if self.caching and self.last_idx != -1:
            del self.sample_cache[self.last_idx]
            self.last_idx = idx

        # if acting as dataset object
        if self.tf_dataset:
            if self.sample_cache.get(idx, None):
                assert len(self.sample_cache[idx]) > 0, "Sample caching error"
            else:
                sample = self.tf_dataset[idx]
                sample_input = sample[0]
                sample_target = sample[1]

                # match num dims as safety check
                samp_dims, targ_dims = len(sample_input.shape), len(sample_target.shape)
                for i in range(max(0, samp_dims - targ_dims)):
                    sample_target = np.expand_dims(sample_target, 1)
                for i in range(max(0, targ_dims - samp_dims)):
                    sample_input = np.expand_dims(sample_input, 1)

                if self.transforms:
                    for transform in self.transforms:
                        sample_input = transform(sample_input)

                if self.target_transforms:
                    for transform in self.target_transforms:
                        sample_target = transform(sample_target)

                # depending on the transformation we might return lists or tuples, which we must unpack
                if self.caching:
                    self.sample_cache[idx] = tuple(
                        map(self.to_cpu, self.unpack(sample_input, sample_target))
                    )
                else:
                    return self.unpack(sample_input, sample_target)

            return tuple(map(self.to_gpu, self.sample_cache[idx]))

        # if acting as container object of dataset objects
        else:
            keys = {}
            if self.val_dataset:
                keys["val"] = self.val_dataset
            if self.train_dataset:
                keys["train"] = self.train_dataset
            if self.test_dataset:
                keys["test"] = self.test_dataset

            if idx in keys:
                return keys[idx]
            else:
                raise KeyError(
                    f"This object is a container. Acceptable keys:{[k for k in keys]}"
                )

    def to_cpu(tensor):
        """
        Sends tensor to cpu. Used because lambda's cant be pickled.

        :param torch.tensor tensor: tensor to sent to cpu
        """
        return tensor.cpu()

    def to_gpu(self, tensor):
        """
        Sends tensor to current gpu device. Used because lambda's cant be pickled.

        :param torch.tensor tensor: tensor to sent to gpu
        """
        return tensor.to(self.device)

    def unpack(self, sample_input, sample_target):
        """
        Helper function for unpacking tuples returned by some transformation objects
        (e.g. GenerateMasks) into outputs.

        Unpacking before returning allows transformation functions which produce variable amounts of
        additional tensor information to pack that information in tuples with the sample and target
        tensors.

        :param torch.tensor/tuple(torch.tensor) sample_input: input sample to unpack
        :param torch.tensor/tuple(torch.tensor) sample_target: target sample to unpack
        """
        inp, targ = type(sample_input), type(sample_target)

        if inp == list or inp == tuple:
            if targ == list or targ == tuple:
                return (*sample_input, *sample_target)
            else:
                return (*sample_input, sample_target)
        else:
            if targ == list or targ == tuple:
                return (sample_input, *sample_target)
            else:
                return (sample_input, sample_target)


class ToTensor(object):
    """
    Transformation. Converts input to torch.Tensor and returns. By default also places tensor
    on cpu.

    :param torch.device device: device transport tensor to
    """

    def __init__(self, device=torch.device("cpu")):
        self.device = device

    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device)
        else:
            sample = torch.tensor(sample, dtype=torch.float32).to(self.device)
        return sample


class Resize(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.

    Transformation. Resises called sample to 'output_size'.
    """

    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, sample):
        sample = cv2.resize(sample, self.output_size)
        sample = cv2.resize(sample, self.output_size)
        return sample


class RandTile(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.

    Transformation. Selects and returns random tile size 'tile_size' from input.
    """

    def __init__(self, tile_size=(256, 256), input_format="zxy"):
        Warning("RandTile is unrecommended for preprocessed data")
        self.tile_size = tile_size
        self.input_format = input_format

    def __call__(self, sample):
        if self.input_format == "zxy":
            x_ind, y_ind = -2, -1
        elif self.input_format == "xyz":
            x_ind, y_ind = -3, -2

        x_shape, y_shape = sample.shape[x_ind], sample.shape[y_ind]
        assert (
            self.tile_size[0] < y_shape and self.tile_size[1] < x_shape
        ), f"Sample size {(x_shape, y_shape)} must be greater than tile size {self.tile_size}."

        randx = np.random.randint(0, x_shape - self.tile_size[1])
        randy = np.random.randint(0, y_shape - self.tile_size[0])

        sample = sample[
            randy : randy + self.tile_size[0], randx : randx + self.tile_size[1]
        ]
        return sample


class RandFlip(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.
    Transformation. Flips input in random direction and returns.
    """

    def __call__(self, sample):
        rand = np.random.randint(0, 2, 2)
        if rand[0] == 1:
            sample = np.flipud(sample)
        if rand[1] == 1:
            sample = np.fliplr(sample)
        return sample


class GenerateMasks(object):
    """
    NOTE: this function is currently unused (and already superceded). I wrote this to provide
    options for transforms performed after the dataloader. These should be removedas they will
    be superceded by gunpowder.

    Appends target channel thresholding based masks for each sample to the sample in a third
    channel, ordered respective to the order of each sample within its minibatch.

    Masks generated are torch tensors.

    :param str masking_type: type of thresholding to apply:
                                1.) Rosin/unimodal: https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
                                2.) Otsu: https://en.wikipedia.org/wiki/Otsu%27s_method
    :param bool clipping: whether or not to clip the extraneous values in the data before
                                    thresholding
    :param int/tiple clip_amount: amount to clip from both ends of brightness histogram
                                    as a percentage (%) if clipping==True but clip_amount == 0,
                                    clip for default amount (2%)
    """

    def __init__(self, masking_type="rosin", clipping=False, clip_amount=0):

        assert masking_type in {"rosin", "unimodal", "otsu"}, (
            f"Unaccepted masking" "type: {masking_type}"
        )
        self.masking_type = masking_type
        self.clipping = clipping
        self.clip_amount = clip_amount

    def __call__(self, sample):
        original_sample = sample

        # convert to numpy
        if type(sample) != type(np.asarray([1, 1])):
            sample = sample.detach().cpu().numpy()

        # clip top and bottom 2% of images for better thresholding
        if self.clipping:
            if type(self.clip_amount) == tuple:
                sample = norm_utils.hist_clipping(
                    sample, self.clip_amount[0], 100 - self.clip_amount[1]
                )
            else:
                if self.clip_amount != 0:
                    sample = norm_utils.hist_clipping(
                        sample, self.clip_amount, 100 - self.clip_amount
                    )
                else:
                    sample = norm_utils.hist_clipping(sample)

        # generate masks
        masks = []
        for i in range(sample.shape[0]):
            if self.masking_type == "otsu":
                masks.append(mask_utils.create_otsu_mask(sample[i, 0, 0]))
            elif self.masking_type == "rosin" or self.masking_type == "unimodal":
                masks.append(mask_utils.create_unimodal_mask(sample[i, 0, 0]))
            else:
                raise NotImplementedError(
                    f"Masking type {self.masking_type} not implemented."
                )
                break
        masks = ToTensor()(np.asarray(masks)).unsqueeze(1).unsqueeze(1)

        return [original_sample, masks]


class Normalize(object):
    """
    NOTE: this function is currently unused. I wrote this to provide options
    for transforms performed after the dataloader. These should be removed
    as they will be superceded by gunpowder.

    Normalizes the sample sample according to the mode in init.

    Params:
    :param str mode: type of normalization to apply
            - one: normalizes sample values proportionally between 0 and 1
            - zeromax: centers sample around zero according to half of its
                        normalized (between -1 and 1) maximum
            - median: centers samples around zero, according to their respective
                        median, then normalizes (between -1 and 1)
            - mean: centers samples around zero, according to their respective
                        means, then normalizes (between -1 and 1)
    """

    def __init__(self, mode="max"):
        self.mode = mode

    def __call__(self, sample, scaling=1):
        """
        Forward call of Normalize
        Params:
            - sample -> torch.Tensor or numpy.ndarray: sample to normalize
            - scaling -> float: value to scale output normalization by
        """
        # determine module
        if isinstance(sample, torch.Tensor):
            module = torch
        elif isinstance(sample, np.ndarray):
            module = np
        else:
            raise NotImplementedError(
                "Only numpy array and torch tensor inputs handled."
            )

        # apply normalization
        if self.mode == "one":
            sample = (sample - module.min(sample)) / (
                module.max(sample) - module.min(sample)
            )
        elif self.mode == "zeromax":
            sample = (sample - module.min(sample)) / (
                module.max(sample) - module.min(sample)
            )
            sample = sample - module.max(sample) / 2
        elif self.mode == "median":
            sample = sample - module.median(sample)
            sample = sample / module.max(module.abs(sample))
        elif self.mode == "mean":
            sample = sample - module.mean(sample)
            sample = sample / module.max(module.abs(sample))
        else:
            raise NotImplementedError(f"Unhandled mode type: '{self.mode}'.")

        return sample * scaling


class RandomNoise(object):
    """
    NOTE: this function is currently unused. I wrote this to provide options
    for transforms performed after the dataloader. These should be removed
    as they will be superceded by gunpowder.

    Augmentation for applying random noise. High variance.

    :param str noise_type: type of noise to apply: 'gauss', 's&p', 'poisson', 'speckle'
    :param numpy.ndarray/torch.tensor sample: input sample
    :return numpy.ndarray/torch.tensor: noisy sample of type input type
    """

    def __init__(self, noise_type):
        self.noise_type = noise_type

    def __call__(self, sample):
        pt = False
        if isinstance(sample, torch.Tensor):
            pt = True
            sample = sample.detach().cpu().numpy()

        if self.noise_type == "gauss":
            row, col, ch = sample.shape
            mean = 0
            var = 0.1
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = sample + gauss
            return noisy

        elif self.noise_type == "s&p":
            row, col, ch = sample.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(sample)

            # Salt mode
            num_salt = np.ceil(amount * sample.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in sample.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * sample.size * (1.0 - s_vs_p))
            coords = [
                np.random.randint(0, i - 1, int(num_pepper)) for i in sample.shape
            ]
            out[coords] = 0
            return out

        elif self.noise_typ == "poisson":
            vals = len(np.unique(sample))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(sample * vals) / float(vals)
            return noisy

        elif self.noise_typ == "speckle":
            row, col, ch = sample.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = sample + sample * gauss
            return noisy

        if pt:
            sample = ToTensor()(sample)
        return sample
