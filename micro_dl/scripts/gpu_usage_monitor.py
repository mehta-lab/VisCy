import argparse
import multiprocessing
import matplotlib.pyplot as plt
import nvidia_smi
import time
import os


def parse_args():
    """
    Parse command line arguments
    In python namespaces are implemented as dictionaries

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        help=(
            "'listen' or 'train'. If training multithreaded, this script cannot"
            " run training, and can only be run simultaneously in listen mode"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to yaml configuration file. if --mode == 'train' must be provided",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="intended gpu device number for monitoring. by default all gpus will be summed",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory to save gpu usage plot",
    )
    parser.add_argument(
        "--run_time",
        type=int,
        help="length of time over which to record gpu usage",
    )
    args = parser.parse_args()
    return args


def record_gpu_usage(usage_list, max_list, gpu):
    """
    Records current gpu memory usage and limit and appends them to input
    lists in-place, respectively

    :param list total_usage: total gpu memory usage across all gpus in mb
    :param list possible: total gpu memory available
    :param int gpu: gpu inde to record from
    """
    total_usage = 0
    available = 0

    if gpu == None:
        device_count = nvidia_smi.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total_usage += info.used
            available += info.total
    else:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        total_usage += info.used
        available += info.total

    usage_list.append(total_usage)
    max_list.append(available)


def plot_usage(usage_over_time, total_over_time, time_recorded, save_dir):
    """
    Plots gpu usage over time and saves plot to save_folder.

    :param list usage_over_time: list of samples of gpu usage at time points
    :param list total_over_time: list of samples of total gpu memory
    :param list time_recorded: time of recording of each of above samples
    :param str save_dir: dir to save plot to
    """

    plt.figure(figsize=(14, 7))
    plt.plot(time_recorded, usage_over_time, label="memory usage over time")
    plt.plot(time_recorded, total_over_time, label="total gpu memory")
    plt.legend()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = os.path.join(save_dir, "gpu_usage")
    plt.savefig(save_name)
    plt.close()


def main():
    """
    Main function for script.
    Starts a training session and records gpu memory usage for the
    specified length (in seconds), then terminates and saves a plot
    of usage over recording time period.

    NOTE: multiprocessing does not work for training sessions with multiple
    gpu processes, as python has really limited multiprocessing offerings.
    If training with multiple processers, specify '--mode listen' in args

    """
    args = parse_args()
    usage_list, max_list, time_recorded = [], [], []
    nvidia_smi.nvmlInit()

    start = time.time()
    if args.mode != "listen":
        import micro_dl.cli.torch_train_script as training

        p = multiprocessing.Process(
            target=training.main,
            name="training",
            args=(args,),
        )
        p.start()
    print(f"--- Begin recording for {args.run_time} seconds ---")
    while time.time() - start < args.run_time:
        record_gpu_usage(usage_list, max_list, args.gpu)
        print(
            f"\tUsage: {usage_list[-1]/1000000:.2f}, Available:{max_list[-1]/1000000:.2f}",
            end="\r",
        )

        time_recorded.append(time.time() - start)
        time.sleep(0.1)  # record no faster than 10/second

    if args.mode != "listen":
        p.terminate()
        print("--- Terminate Training ---")
        p.join()
    print(
        f"\tRecorded {len(usage_list)} values over {time.time() - start:.2f} seconds."
    )
    print(f"\tPeak availability: {max(max_list)/1000000:.2f} mb")
    print(f"\tPeak usage: {max(usage_list)/1000000:.2f} mb")
    print("--- Saving recording plots ---")
    plot_usage(usage_list, max_list, time_recorded, args.save_dir)


if __name__ == "__main__":
    main()
