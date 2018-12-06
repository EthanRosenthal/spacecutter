from skorch.callbacks import Callback
from torch.nn import Module

from spacecutter.models import LogisticCumulativeLink


class AscensionCallback(Callback):
    """
    Ensure that each cutpoint is ordered in ascending value.
    e.g.

    .. < cutpoint[i - 1] < cutpoint[i] < cutpoint[i + 1] < ...

    This is done by clipping the cutpoint values at the end of a batch gradient
    update. By no means is this an efficient way to do things, but it works out
    of the box with stochastic gradient descent.

    Parameters
    ----------
    margin : float, (default=0.0)
        The minimum value between any two adjacent cutpoints.
        e.g. enforce that cutpoint[i - 1] + margin < cutpoint[i]
    min_val : float, (default=-1e6)
        Minimum value that the smallest cutpoint may take.
    """
    def __init__(self, margin: float = 0.0, min_val: float = -1.0e6) -> None:
        super().__init__()
        self.margin = margin
        self.min_val = min_val

    def clip(self, module: Module) -> None:
        # NOTE: Only works for LogisticCumulativeLink right now
        # We assume the cutpoints parameters are called `cutpoints`.
        if isinstance(module, LogisticCumulativeLink):
            cutpoints = module.cutpoints.data
            for i in range(cutpoints.shape[0] - 1):
                cutpoints[i].clamp_(self.min_val,
                                    cutpoints[i + 1] - self.margin)

    def on_batch_end(self, net: Module, *args, **kwargs) -> None:
        net.module_.apply(self.clip)
