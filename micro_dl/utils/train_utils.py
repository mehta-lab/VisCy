"""Utility functions used for training"""
import warnings
from keras import backend as K, losses as keras_losses, metrics as keras_metrics
import numpy as np
import os
import subprocess

from micro_dl.input.dataset import BaseDataSet
from micro_dl.input import DataSetWithMask
from micro_dl.input.training_table import BaseTrainingTable
from micro_dl.train import losses as custom_losses, metrics as custom_metrics
import micro_dl.utils.preprocess_utils as preprocess_utils


def get_image_dir_format(dataset_config):
    """Get dir with input images for generating full path from frames_meta

    If the tiled dir is passed as data dir there will be no
    preprocessing_info.json. If json present use it, else read images from the
    given dir.
    """

    # tile dir pass directly as data_dir
    tile_dir = dataset_config["data_dir"]
    image_format = "zyx"
    try:
        preprocess_config = preprocess_utils.get_preprocess_config(
            os.path.dirname(tile_dir)
        )

        # Get shape order from recent_json
        if "image_format" in preprocess_config["tile"]:
            image_format = preprocess_config["tile"]["image_format"]
    except Exception as e:
        print(
            "Error while reading preprocess config: {}. "
            'Use default image format "zyx"'.format(e)
        )

    return tile_dir, image_format


def create_train_datasets(
    df_meta,
    tile_dir,
    dataset_config,
    trainer_config,
    image_format,
    masked_loss,
    meta_dir,
):
    """Create train, val and test datasets

    Saves val_metadata.csv and test_metadata.csv for checking model performance

    :param pd.DataFrame df_meta: Dataframe containing info on split tiles
    :param str tile_dir: directory containing training image tiles
    :param dict dataset_config: dict with dataset related params
    :param dict trainer_config: dict with params related to training
    :param str image_format: Tile shape order: 'xyz' or 'zyx'
    :param bool masked_loss: Whether or not to use masks
    :param str meta_dir: actual directory of model to save split metadata to,
                            different from config model dir as directory is
                            generated with timestamp
    :return: Dict containing
     :BaseDataSet df_train: training dataset
     :BaseDataSet df_val: validation dataset
     :BaseDataSet df_test: test dataset
     :dict split_idx: dict with keys [train, val, test] and list of sample
      numbers as values
    """
    mask_channels = None
    if masked_loss:
        mask_channels = dataset_config["mask_channels"]

    random_seed = None
    if "random_seed" in dataset_config:
        random_seed = dataset_config["random_seed"]

    tt = BaseTrainingTable(
        df_metadata=df_meta,
        input_channels=dataset_config["input_channels"],
        target_channels=dataset_config["target_channels"],
        split_by_column=dataset_config["split_by_column"],
        split_ratio=dataset_config["split_ratio"],
        mask_channels=mask_channels,
        random_seed=random_seed,
    )
    all_metadata, split_samples = tt.train_test_split()
    csv_names = ["train_metadata.csv", "val_metadata.csv", "test_metadata.csv"]
    df_names = ["df_train", "df_val", "df_test"]
    all_datasets = {}
    for i in range(3):
        metadata = all_metadata[df_names[i]]
        if isinstance(metadata, type(None)):
            all_datasets[df_names[i]] = None
        else:
            if masked_loss:
                dataset = DataSetWithMask(
                    tile_dir=tile_dir,
                    input_fnames=metadata["fpaths_input"],
                    target_fnames=metadata["fpaths_target"],
                    mask_fnames=metadata["fpaths_mask"],
                    dataset_config=dataset_config,
                    batch_size=trainer_config["batch_size"],
                    image_format=image_format,
                )
            else:
                dataset = BaseDataSet(
                    tile_dir=tile_dir,
                    input_fnames=metadata["fpaths_input"],
                    target_fnames=metadata["fpaths_target"],
                    dataset_config=dataset_config,
                    batch_size=trainer_config["batch_size"],
                    image_format=image_format,
                )
            metadata.to_csv(os.path.join(meta_dir, csv_names[i]), sep=",")
            all_datasets[df_names[i]] = dataset

    return all_datasets, split_samples


def check_gpu_availability(gpu_id):
    """
    Check if mem_frac is available in given gpu_id

    :param int/list gpu_id: id of the gpu to be used. Int for single GPU
     training, list for distributed training
    :param list gpu_mem_frac: mem fraction for each GPU in gpu_id
    :return bool gpu_availability: True if all mem_fracs are greater than
        gpu_mem_frac
    :return list curr_mem_frac: list of current memory fractions available
        for gpus in to gpu_id
    """
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    cur_mem_frac = []
    for idx, gpu in enumerate(gpu_id):
        query = (
            "nvidia-smi --id={} --query-gpu=memory.free,memory.total " "--format=csv"
        ).format(gpu)
        sp = subprocess.Popen([query], stdout=subprocess.PIPE, shell=True)
        query_output = sp.communicate()
        query_output = query_output[0].decode("utf8")
        query_output = query_output.split("\n")
        mem = query_output[1].split(",")
        mem = [int(val.replace("MiB", "")) for val in mem]
        cur_mem_frac.append(mem[0] / mem[1])
    return cur_mem_frac


def select_gpu(gpu_ids=None, gpu_mem_frac=None):
    """
    Find the GPU ID with highest available memory fraction.
    If ID is given as input, set the gpu_mem_frac to maximum available,
    or if a memory fraction is given, make sure the given GPU has the desired
    memory fraction available.
    Currently only supports single GPU runs.

    :param int gpu_ids: Desired GPU ID. If None, find GPU with the most memory
        available.
    :param float gpu_mem_frac: Desired GPU memory fraction [0, 1]. If None,
        use maximum available amount of GPU.
    :return int gpu_ids: GPU ID to use.
    :return float cur_mem_frac: GPU memory fraction to use
    :raises NotImplementedError: If gpu_ids is not int
    :raises AssertionError: If requested memory fraction isn't available
    """
    # Check if user has specified a GPU ID to use
    if not isinstance(gpu_ids, type(None)):
        # Currently only supporting one GPU as input
        if not isinstance(gpu_ids, int):
            raise NotImplementedError("Currently gpu id specification must be int")
        if gpu_ids == -1:
            return -1, 0
        cur_mem_frac = check_gpu_availability(gpu_ids)
        cur_mem_frac = cur_mem_frac[0]
        if not isinstance(gpu_mem_frac, type(None)):
            assert (
                cur_mem_frac >= gpu_mem_frac
            ), "Not enough memory available. Requested/current fractions: {0:.4g}/{0:.4g}".format(
                gpu_mem_frac, cur_mem_frac
            )
        print("Using GPU {} with memory fraction {}.".format(gpu_ids, cur_mem_frac))
        return gpu_ids, cur_mem_frac

    # User has not specified GPU ID, find the GPU with most memory available
    sp = subprocess.Popen(
        ["nvidia-smi --query-gpu=index --format=csv"],
        stdout=subprocess.PIPE,
        shell=True,
    )
    gpu_ids = sp.communicate()
    gpu_ids = gpu_ids[0].decode("utf8")
    gpu_ids = gpu_ids.split("\n")
    # If no GPUs are found, run on CPU (debug mode)
    if len(gpu_ids) <= 2:
        print("No GPUs found, run will be slow. Query result: {}".format(gpu_ids))
        return -1, 0
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids[1:-1]]
    cur_mem_frac = check_gpu_availability(gpu_ids)
    # Get the GPU with maximum memory fraction
    max_mem = max(cur_mem_frac)
    idx = cur_mem_frac.index(max_mem)
    gpu_id = gpu_ids[idx]
    # Subtract a little margin to be safe
    max_mem = max_mem - np.finfo(np.float32).eps
    print("Using GPU {} with memory fraction {}.".format(gpu_id, max_mem))
    return gpu_id, max_mem


def set_keras_session(gpu_ids, gpu_mem_frac):
    raise DeprecationWarning("Keras sessions are no longer supported as of 2.0.0")
    """Set the Keras session"""

    assert K.backend() == "tensorflow"
    tf = K.tf
    # assumes only one process is run per GPU, if not get num_processes and
    # change accordingly
    gpu_options = tf.GPUOptions(
        visible_device_list=str(gpu_ids),
        allow_growth=True,
        per_process_gpu_memory_fraction=gpu_mem_frac,
    )
    config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False
    )
    # log_device_placement to find out which devices the operations and tensors
    # are assigned to
    sess = tf.Session(config=config)
    K.set_session(sess)
    return sess


def get_loss(loss_str):
    raise DeprecationWarning("Tensorflow losses are no longer supported as of 2.0.0")
    """Get loss type from config"""

    def _get_one_loss(cur_loss_str):
        if hasattr(keras_losses, cur_loss_str):
            loss_cls = getattr(keras_losses, cur_loss_str)
        elif hasattr(custom_losses, cur_loss_str):
            loss_cls = getattr(custom_losses, cur_loss_str)
        else:
            raise ValueError("%s is not a valid loss" % cur_loss_str)
        return loss_cls

    if not isinstance(loss_str, list):
        loss_cls = _get_one_loss(loss_str)
        return loss_cls
    else:
        loss_cls_list = []
        for cur_loss in loss_str:
            loss_cls = _get_one_loss(cur_loss)
            loss_cls_list.append(loss_cls)
        return loss_cls_list


def get_metrics(metrics_list):
    """Get the metrics from config"""

    metrics_cls = []
    if not isinstance(metrics_list, list):
        metrics_list = [metrics_list]

    for m in metrics_list:
        if hasattr(keras_metrics, m):
            warnings.warn(f"metric {m} dependent on keras, tensorflow 1.1")
            # TODO implement keras metrics in torch
            cur_metric_cls = getattr(keras_metrics, m)
        elif hasattr(custom_metrics, m):
            cur_metric_cls = getattr(custom_metrics, m)
        else:
            raise ValueError("%s is not a valid metric" % m)
        metrics_cls.append(cur_metric_cls)
    return metrics_cls
