"""Utility functions used for training"""
from keras import backend as K, losses as keras_losses, \
    metrics as keras_metrics
import numpy as np
import subprocess

from micro_dl.train import losses as custom_losses, metrics as custom_metrics


def check_gpu_availability(gpu_id, gpu_mem_frac):
    """Check if mem_frac is available in given gpu_id

    :param int/list gpu_id: id of the gpu to be used. Int for single GPU
     training, list for distributed training
    :param float/list gpu_mem_frac: mem fraction for each GPU in gpu_id
    :return: boolean indicator for gpu_availability
    """

    gpu_availability = []
    if isinstance(gpu_id, int):
        gpu_id = [gpu_id]

    if isinstance(gpu_mem_frac, float):
        gpu_mem_frac = [gpu_mem_frac]

    msg = 'There is no matching memory fraction for all the given gpu_ids'
    assert len(gpu_id) == len(gpu_mem_frac), msg
    for idx, gpu in enumerate(gpu_id):
        query = ('nvidia-smi --id={} --query-gpu=memory.free,memory.total '
                 '--format=csv').format(gpu)
        sp = subprocess.Popen([query], stdout=subprocess.PIPE, shell=True)
        query_output = sp.communicate()
        query_output = query_output[0].decode('utf8')
        query_output = query_output.split('\n')
        mem = query_output[1].split(',')
        mem = [int(val.replace('MiB', '')) for val in mem]
        curr_mem_frac = mem[0] / mem[1]
        curr_availability = curr_mem_frac>=gpu_mem_frac[idx]
        gpu_availability.append(curr_availability)
    gpu_availability = np.array(gpu_availability)
    return np.all(gpu_availability)


def split_train_val_test(num_samples, train_ratio, test_ratio,
                          val_ratio=None):
    """Generate indices for train, validation and test split

    This can be achieved by using sklearn.model_selection.train_test_split
    twice... :-)

    :param int num_samples: total number of samples/datasets
    :param float train_ratio: between 0 and 1, percent of samples to be
     used for training
    :param float test_ratio: between 0 and 1, percent of samples to be
     used for test set
    :param float val_ratio: between 0 and 1, percent of samples to be
     used for the validation set
    :return: dict split_idx with keys [train, val, test] and values as lists
    """

    msg = 'train, val and test ratios do not add upto 1'
    assert train_ratio + val_ratio + test_ratio == 1, msg
    num_test = int(test_ratio * num_samples)
    num_test = max(num_test, 1)

    split_idx = {}
    test_idx = np.random.randint(0, num_samples, num_test)
    test_idx = list(test_idx)
    if num_test == 1:
        test_idx = [test_idx[0]]
    split_idx['test'] = test_idx
    rem_set = set(range(0, num_samples)) - set(test_idx)

    if val_ratio:
        num_val = int(val_ratio * num_samples)
        num_val = max(num_val, 1)
        idx = np.random.randint(0, len(rem_set), num_val).astype('int')
        rem_set_as_list = list(rem_set)
        val_idx = [rem_set_as_list[val] for val in idx] 
        if num_val == 1:
            rem_set.remove(val_idx[0])
        else:
            rem_set = rem_set - set(val_idx)
        split_idx['val'] = val_idx
    train_idx = list(rem_set)
    split_idx['train'] = train_idx

    return split_idx


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

    if hasattr(keras_losses, loss_str):
        loss_cls = getattr(keras_losses, loss_str)
    elif hasattr(custom_losses, loss_str):
        loss_cls = getattr(custom_losses, loss_str)
    else:
        raise ValueError('%s is not a valid loss' % loss_str)
    return loss_cls


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