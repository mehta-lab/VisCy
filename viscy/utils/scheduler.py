from typing import Literal


class ParameterScheduler:
    """
    Generic scheduler for any parameter (beta, temporal_weight, etc.).

    Supports linear warmup, cosine annealing, step warmup, or constant values.

    Parameters
    ----------
    param_name : str
        Name of the parameter being scheduled (for logging/debugging)
    initial_value : float
        Starting value at epoch 0
    target_value : float
        Target value after warmup completes
    warmup_epochs : int
        Number of epochs for warmup phase
    schedule_type : {"linear", "cosine", "warmup", "constant"}
        Type of schedule:
        - "linear": Linear interpolation from initial to target
        - "cosine": Cosine annealing from initial to target
        - "warmup": Stay at initial, then jump to target at warmup_epochs
        - "constant": Always return target_value (no warmup)
    min_value : float, optional
        Minimum clipping value, by default 1e-15

    Examples
    --------
    >>> # Linear warmup from 0.1 to 1.0 over 50 epochs
    >>> scheduler = ParameterScheduler(
    ...     param_name="beta",
    ...     initial_value=0.1,
    ...     target_value=1.0,
    ...     warmup_epochs=50,
    ...     schedule_type="linear"
    ... )
    >>> scheduler.get_value(0)  # Returns ~0.1
    >>> scheduler.get_value(25)  # Returns ~0.55
    >>> scheduler.get_value(50)  # Returns 1.0

    >>> # Constant value (no scheduling)
    >>> scheduler = ParameterScheduler(
    ...     param_name="temporal_weight",
    ...     initial_value=0.0,
    ...     target_value=0.5,
    ...     warmup_epochs=0,
    ...     schedule_type="constant"
    ... )
    >>> scheduler.get_value(0)  # Returns 0.5
    >>> scheduler.get_value(100)  # Returns 0.5
    """

    def __init__(
        self,
        param_name: str,
        initial_value: float,
        target_value: float,
        warmup_epochs: int,
        schedule_type: Literal["linear", "cosine", "warmup", "constant"] = "constant",
        min_value: float = 1e-15,
    ):
        self.param_name = param_name
        self.initial_value = initial_value
        self.target_value = target_value
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.min_value = min_value

        # Validate inputs
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        if schedule_type not in ["linear", "cosine", "warmup", "constant"]:
            raise ValueError(f"Invalid schedule_type: {schedule_type}")

    def get_value(self, current_epoch: int) -> float:
        """
        Get parameter value for the current epoch.

        Parameters
        ----------
        current_epoch : int
            Current training epoch (0-indexed)

        Returns
        -------
        float
            Scheduled parameter value, clipped to min_value
        """
        if self.schedule_type == "constant":
            return max(self.target_value, self.min_value)

        if current_epoch >= self.warmup_epochs:
            return max(self.target_value, self.min_value)

        progress = current_epoch / self.warmup_epochs

        if self.schedule_type == "linear":
            value = (
                self.initial_value + (self.target_value - self.initial_value) * progress
            )

        elif self.schedule_type == "cosine":
            import math

            value = self.initial_value + (
                self.target_value - self.initial_value
            ) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))

        elif self.schedule_type == "warmup":
            value = self.initial_value

        else:
            value = self.target_value

        return max(value, self.min_value)

    def __repr__(self) -> str:
        return (
            f"ParameterScheduler(param_name='{self.param_name}', "
            f"initial_value={self.initial_value}, target_value={self.target_value}, "
            f"warmup_epochs={self.warmup_epochs}, schedule_type='{self.schedule_type}')"
        )
