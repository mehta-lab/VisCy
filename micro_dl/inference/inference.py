import math
import os
import time

import iohub.ngff as ngff
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import micro_dl.inference.inference_dataset as inference_dataset
import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.utils.aux_utils as aux_utils


def _pad_input(x: Tensor, num_blocks: int):
    """
    Zero-pads row and col dimensions of inputs to a multiple of 2**num_blocks

    :param torch.tensor x: input tensor

    :return torch.tensor x_padded: zero-padded x
    :return tuple pad_shape: shape x was padded by (left, top, right, bottom)
    """
    down_factor = 2**num_blocks
    sizes = [down_factor * math.ceil(s / down_factor) - s for s in x.shape[-2:]]
    pads = [(p // 2, p - p // 2) for p in sizes]
    pads = (pads[1][0], pads[0][0], pads[1][1], pads[0][1])
    x = TF.pad(x, pads)
    return x, pads


class TorchPredictor:
    """
    TorchPredictor class
    TorchPredictor object handles all procedures involved with model inference.
    Utilizes an InferenceDataset object for reading data in from the given zarr store

    Params:
    :param dict config: inference config file
    :param torch.device device: device to perform prediction on
    :param bool single_prediction: whether to initialize this object for
                single_prediction only
    """

    def __init__(self, config, device, single_prediction=False) -> None:
        self.config = config
        self.device = device
        self.single_prediction = single_prediction

        # use params logged from training to set up model and dataset
        self.model = None
        self._read_model_params()

        self.network_config = self.model_meta["model"]["model_config"]
        self.network_z_depth = self.network_config["in_stack_depth"]

        # initialize dataset reader
        if not single_prediction:
            self.positions = self.config["positions"]
            self.input_channels = self.config["input_channels"]
            self.time_indices = self.config["time_indices"]

            self.dataset = inference_dataset.TorchInferenceDataset(
                zarr_dir=self.config["zarr_dir"],
                batch_pred_num=self.config["batch_size"],
                normalize_inputs=self.config["normalize_inputs"],
                norm_type=self.config["norm_type"],
                norm_scheme=self.config["norm_scheme"],
                sample_depth=self.network_z_depth,
                device=self.device,
            )

            # set output zarr store location
            self._get_save_location()

    def load_model(self) -> None:
        """
        Initializes a model according to the network configuration dictionary used
        to train it, and loads the parameters saved in model_dir into the model's state dict.

        :param str init_dir: directory containing model weights and biases (should be true)
        """
        model = model_utils.model_init(
            self.network_config,
            device=self.device,
            debug_mode=False,
        )

        model_dir = self.config["model_dir"]
        model_name = self.config["model_name"]
        light_state_dict = torch.load(
            os.path.join(model_dir, "checkpoints", model_name), map_location=self.device
        )["state_dict"]

        # clean lightning state dict
        clean_state_dict = {}
        for key, val in light_state_dict.items():
            if isinstance(key, str):
                if "model." in key:
                    newkey = key[6:]
                    clean_state_dict[newkey] = light_state_dict[key]
                else:
                    clean_state_dict[key] = light_state_dict[key]

        readout = model.load_state_dict(clean_state_dict)
        print(f"PyTorch model load status: {readout}")
        self.model = model

    def predict_image(
        self,
        input_image,
        model=None,
    ):
        """
        Runs prediction on entire image field of view.
        If the input XY size is not compatible with the model
        (a multiple of :math:`2^{blocks}`),
        it will be padded with zeros on all sides for inference
        and cropped to the original size before output.
        Input must be either 4 or 5 dimensions, and output is returned with the
        same dimensionality as given in input.

        Params:
        :param numpy.ndarray/torch.Tensor input_image: input image or image stack on which
                                                        to run prediction
        :param Torch.nn.Module model: trained model to use for prediction

        :return np.ndarray prediction: prediction
        """
        assert (
            self.model != None or model != None
        ), "must specify model in init or prediction call"
        assert 5 - len(input_image.shape) <= 1, (
            f"input image has {len(input_image.shape)} dimensions"
            ", either 4 (2D) or 5 (2.5D) required."
        )
        if model == None:
            model = self.model
        model.eval()

        if self.network_config["architecture"] == "2.5D":
            if len(input_image.shape) != 5:
                raise ValueError(
                    f"2.5D unet must take 5D input data. Received {len(input_image.shape)}."
                    " Check preprocessing config."
                )
            img_tensor = aux_utils.ToTensor(device=self.device)(input_image)

        elif self.network_config["architecture"] == "2D":
            # Torch Unet 2D takes 2 spatial dims, handle lingering 1 in z dim
            if len(input_image.shape) != 4:
                raise ValueError(
                    f"2D unet must take 4D input data. Received {len(input_image.shape)}."
                    " Check preprocessing config."
                )
            img_tensor = aux_utils.ToTensor(device=self.device)(input_image)

        img_tensor, pads = _pad_input(img_tensor, num_blocks=model.num_blocks)
        pred = model(img_tensor, validate_input=False)
        return TF.crop(
            pred.detach().cpu(), *(pads[1], pads[0]) + input_image.shape[-2:]
        ).numpy()

    def run_inference(self):
        """
        Performs inference on the entire validation dataset.

        Model inputs are normalized before predictions are generated. Predictions are saved in an
        HCS-compatible zarr store in the specified output location.
        """
        assert (
            self.single_prediction == False
        ), "Must be initialized for dataset prediction."

        # init io and saving
        start = time.time()
        self.log_writer = SummaryWriter(log_dir=self.save_folder)
        self.output_writer = ngff.open_ome_zarr(
            os.path.join(self.save_folder, "preds.zarr"),
            layout="hcs",
            mode="w-",
            channel_names=self.input_channels,
        )
        self.model.eval()

        # generate list of position tuples from dictionary for iteration
        positions_dict = self._get_positions()
        position_paths = []
        for row_k, row_v in positions_dict.items():
            for well_k, well_v in row_v.items():
                fov_path_tuples = [(row_k, well_k, pos_k) for pos_k in well_v]
                position_paths.extend(fov_path_tuples)

        # run inference on each position
        print("Running inference: \n")

        for row_name, col_name, fov_name in tqdm(
            position_paths, position=0, desc="positions "
        ):
            for time_idx in tqdm(
                self.config["time_indices"], desc="timepoints ", position=1, leave=False
            ):
                # split up prediction generation by time idx for very large arrays
                shape, dtype = self.dataset.set_source_array(
                    row_name, col_name, fov_name, time_idx, self.input_channels
                )
                output_array = self._new_empty_array(
                    row_name, col_name, fov_name, shape, dtype
                )
                dataloader = iter(DataLoader(self.dataset))

                for batch, z0, size, _ in dataloader:
                    batch_pred = self.predict_image(batch[0])  # redundant batch dim
                    batch_pred = np.squeeze(np.swapaxes(batch_pred, 0, 2), axis=0)
                    output_array[time_idx, :, z0 : z0 + size, ...] = batch_pred

        # write config to save dir
        aux_utils.write_yaml(self.config, os.path.join(self.save_folder, "config.yaml"))
        self.output_writer.close()
        print(f"Done! Time taken: {time.time() - start:.2f}s")
        print(f"Predictions saved to {self.save_folder}")

    def _get_positions(self):
        """
        Positions should be specified in config in format:
            positions:
                row #:
                    col #: [pos #, pos #, ...]
                        ...
                    ...
        where row # and col # together indicate the well on the plate, and pos # indicates
        the number of the position in the well.

        :return dict positions: Returns dictionary tree specifying all the positions
                            in the format written above
        """
        # Positions are specified
        if isinstance(self.positions, dict):
            print("Predicting on positions specified in inference config.")
            return self.positions
        else:
            raise AttributeError(
                "Invalid 'positions'. Expected dict of positions by rows and cols"
                f" but got {self.positions}"
            )

    def _new_empty_array(self, row_name, col_name, fov_name, shape, dtype):
        """
        Subroutine: builds an empty array for outputs in the specified position
        of the specified shape and datatype
        """
        # get time indices, channels, z-depth of output
        z_slices_output = shape[-3] - (self.network_z_depth - 1)
        output_tcz = (len(self.time_indices), len(self.input_channels), z_slices_output)

        output_position = self.output_writer.create_position(
            row_name, col_name, fov_name
        )
        output_array = output_position.create_zeros(
            name="0",
            shape=output_tcz + shape[-2:],
            dtype=dtype,
            chunks=(1,) * (len(shape) - 2) + shape[-2:],
        )
        return output_array

    def _read_model_params(self, model_dir=None):
        """
        Reads the model parameters from the given model dir and stores it as an attribute.
        Use here is to allow for inference to 'intelligently' infer it's configuration
        parameters from a model directory to reduce hassle on user.

        :param str model_dir: global path to model directory in which 'model_metadata.yml'
                        is stored. If not specified, infers from inference config.
        """
        if not model_dir:
            model_dir = self.config["model_dir"]

        model_meta_filename = os.path.join(model_dir, "config.yaml")
        self.model_meta = aux_utils.read_config(model_meta_filename)

    def _get_save_location(self):
        """
        Sets save location as specified in config files.

        Note: for save location to be different than training save location,
        not only does inference/save_preds_to_model_dir need to be False,
        but you must specify a new location in inference/custom_save_preds_dir

        This is to encourage saving model inference with training models.

        """
        if self.config["save_preds_to_model_dir"]:
            save_dir = self.config["model_dir"]
        elif "custom_save_preds_dir" in self.config:
            custom_save_dir = self.config["custom_save_preds_dir"]
            save_dir = custom_save_dir
        else:
            raise ValueError(
                "Must provide custom_save_preds_dir if save_preds_to"
                "_model_dir is False."
            )

        now = aux_utils.get_timestamp()
        self.save_folder = os.path.join(save_dir, "inference_predictions", f"{now}")
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
