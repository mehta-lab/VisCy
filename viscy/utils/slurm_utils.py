import psutil
import torch


def calculate_dataloader_settings(
    batch_size: int,
    sample_memory_mb: float,
    available_ram_gb: float | None = None,
    available_cpu_cores: int | None = None,
    target_ram_usage: float = 0.25,
    target_vram_usage: float = 0.1,
    available_vram_gb: float | None = None,
    use_gpu: bool = True,
) -> dict:
    """
    Calculate optimal DataLoader settings based on system resources including GPU VRAM.

    Parameters
    ----------
    batch_size: int
        Size of each batch
    sample_memory_mb: float
        Approximate memory per sample in MB
    available_ram_gb: float, optional
        Available RAM in GB. If None, will use system RAM
    available_cpu_cores: int, optional
        Number of CPU cores. If None, will use system cores
    target_ram_usage: float, optional
        Target fraction of RAM to use for prefetching. If None, will use 0.25 of RAM.
    target_vram_usage: float, optional
        Target fraction of VRAM to use for prefetching. If None, will use 0.1 of VRAM.
    available_vram_gb: float, optional
        Available VRAM in GB. If None, will use system VRAM.
    use_gpu: bool, optional
        Whether to consider GPU memory constraints. If False, VRAM constraints are ignored

    Returns
    -------
    dict: Recommended settings for DataLoader
    """
    # Get system resources if not provided
    if available_ram_gb is None:
        available_ram_gb = psutil.virtual_memory().total / (1024**3)
    if available_cpu_cores is None:
        available_cpu_cores = psutil.cpu_count(logical=False)

    # Calculate memory per batch
    batch_memory_mb = batch_size * sample_memory_mb

    # Calculate maximum prefetch factor based on RAM
    max_prefetch_memory_mb = (available_ram_gb * 1024) * target_ram_usage
    max_prefetch_factor_ram = int(
        max_prefetch_memory_mb / (batch_memory_mb * available_cpu_cores)
    )

    # Calculate maximum prefetch factor based on VRAM if GPU is being used
    max_prefetch_factor_vram = float("inf")
    if use_gpu:
        if available_vram_gb is None:
            if torch.cuda.is_available():
                available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
            else:
                raise ValueError(
                    "use_gpu is True but no VRAM specified and CUDA is not available"
                )

        max_prefetch_memory_mb_vram = (available_vram_gb * 1024) * target_vram_usage
        max_prefetch_factor_vram = int(
            max_prefetch_memory_mb_vram / (batch_memory_mb * available_cpu_cores)
        )

    # Take the minimum of RAM and VRAM based prefetch factors
    max_prefetch_factor = min(max_prefetch_factor_ram, max_prefetch_factor_vram)
    max_prefetch_factor = max(1, min(max_prefetch_factor, 4))  # Cap between 1 and 4

    # Calculate optimal number of workers
    # Leaving 2 cores for main process and other tasks
    optimal_workers = max(1, min(available_cpu_cores - 2, available_cpu_cores))

    return {
        "num_workers": optimal_workers,
        "prefetch_factor": max_prefetch_factor,
        "persistent_workers": True,
        "pin_memory": True,
        "estimated_memory_usage_mb": batch_memory_mb
        * optimal_workers
        * max_prefetch_factor,
        "estimated_vram_usage_mb": (
            batch_memory_mb * optimal_workers * max_prefetch_factor if use_gpu else 0
        ),
        "available_vram_gb": available_vram_gb if use_gpu else None,
    }
