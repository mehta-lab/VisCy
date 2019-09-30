"""Utility functions used for training"""
from keras import backend as K, losses as keras_losses, \
    metrics as keras_metrics
import numpy as np
import subprocess

from micro_dl.train import losses as custom_losses, metrics as custom_metrics


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
        query = ('nvidia-smi --id={} --query-gpu=memory.free,memory.total '
                 '--format=csv').format(gpu)
        sp = subprocess.Popen([query], stdout=subprocess.PIPE, shell=True)
        query_output = sp.communicate()
        query_output = query_output[0].decode('utf8')
        query_output = query_output.split('\n')
        mem = query_output[1].split(',')
        mem = [int(val.replace('MiB', '')) for val in mem]
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
            raise NotImplementedError
        if gpu_ids == -1:
            return -1, 0
        cur_mem_frac = check_gpu_availability(gpu_ids)
        if not isinstance(gpu_mem_frac, type(None)):
            if isinstance(gpu_mem_frac, float):
                gpu_mem_frac = [gpu_mem_frac]
            assert np.all(np.array(cur_mem_frac >= gpu_mem_frac)), \
                ("Not enough memory available. Requested/current fractions:",
                    "\n".join([str(c) + " / " + "{0:.4g}".format(m)
                              for c, m in zip(gpu_mem_frac, cur_mem_frac)]))
        return gpu_ids, cur_mem_frac[0]

    # User has not specified GPU ID, find the GPU with most memory available
    sp = subprocess.Popen(['nvidia-smi --query-gpu=index --format=csv'],
                          stdout=subprocess.PIPE,
                          shell=True)
    gpu_ids = sp.communicate()
    gpu_ids = gpu_ids[0].decode('utf8')
    gpu_ids = gpu_ids.split('\n')
    # If no GPUs are found, run on CPU (debug mode)
    if len(gpu_ids) <= 2:
        print('No GPUs found, run will be slow. Query result: {}'.format(gpu_ids))
        return -1, 0
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids[1:-1]]
    cur_mem_frac = check_gpu_availability(gpu_ids)
    # Get the GPU with maximum memory fraction
    max_mem = max(cur_mem_frac)
    idx = cur_mem_frac.index(max_mem)
    gpu_id = gpu_ids[idx]
    # Subtract a little margin to be safe
    max_mem = max_mem - np.finfo(np.float32).eps
    print('Using GPU {} with memory fraction {}.'.format(gpu_id, max_mem))
    return gpu_id, max_mem


def set_keras_session(gpu_ids, gpu_mem_frac):
    """Set the Keras session"""

    assert K.backend() == 'tensorflow'
    tf = K.tf
    # assumes only one process is run per GPU, if not get num_processes and
    # change accordingly
    gpu_options = tf.GPUOptions(visible_device_list=str(gpu_ids),
                                allow_growth=True,
                                per_process_gpu_memory_fraction=gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            allow_soft_placement=True,
                            log_device_placement=False)
    # log_device_placement to find out which devices the operations and tensors
    # are assigned to
    sess = tf.Session(config=config)
    K.set_session(sess)
    return sess


def get_loss(loss_str):
    """Get loss type from config"""

    def _get_one_loss(cur_loss_str):
        if hasattr(keras_losses, cur_loss_str):
            loss_cls = getattr(keras_losses, cur_loss_str)
        elif hasattr(custom_losses, cur_loss_str):
            loss_cls = getattr(custom_losses, cur_loss_str)
        else:
            raise ValueError('%s is not a valid loss' % cur_loss_str)
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
            cur_metric_cls = getattr(keras_metrics, m)
        elif hasattr(custom_metrics, m):
            cur_metric_cls = getattr(custom_metrics, m)
        else:
            raise ValueError('%s is not a valid metric' % m)
        metrics_cls.append(cur_metric_cls)
    return metrics_cls
