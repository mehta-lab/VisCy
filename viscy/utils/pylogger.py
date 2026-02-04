"""PyLogger for ranked logging in distributed training."""

import logging

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """Logger that only logs on rank 0 in distributed training.

    This prevents redundant logging when using multiple processes.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: dict | None = None,
    ) -> None:
        """Initialize the RankedLogger.

        Parameters
        ----------
        name : str
            Name of the logger
        rank_zero_only : bool
            If True, only log on rank 0
        extra : dict, optional
            Extra context for the logger
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(
        self, level: int, msg: str, rank: int | None = None, *args, **kwargs
    ) -> None:
        """Log a message at the specified level.

        If rank is specified, only log on that rank.
        If rank_zero_only is True, only log on rank 0.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
