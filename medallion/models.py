from copy import deepcopy

import torch
from torch import nn


class LogisticCumulativeLink(nn.Module):
    """
    Converts a single number to the proportional odds of belonging to a class.

    Parameters
    ----------
    num_classes : int
        Number of ordered classes to partition the odds into.
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, num_classes: int,
                 init_cutpoints: str = 'ordered') -> None:
        assert num_classes > 2, (
            'Only use this model if you have 3 or more classes'
        )
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == 'ordered':
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == 'random':
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
            raise ValueError(f'{init_cutpoints} is not a valid init_cutpoints '
                             f'type')

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Equation (11) from
        "On the consistency of ordinal regression methods", Pedregosa et. al.
        """
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        return link_mat


class OrdinalLogisticModel(nn.Module):
    """
    "Wrapper" model for outputting proportional odds of ordinal classes.
    Pass in any model that outputs a single prediction value, and this module
    will then pass that model through the LogisticCumulativeLink module.

    Parameters
    ----------
    predictor : nn.Module
        When called, must return a torch.FloatTensor with shape [batch_size, 1]
    init_cutpoints : str (default='ordered')
        How to initialize the cutpoints of the model. Valid values are
        - ordered : cutpoints are initialized to halfway between each class.
        - random : cutpoints are initialized with random values.
    """

    def __init__(self, predictor: nn.Module, num_classes: int,
                 init_cutpoints: str = 'ordered') -> None:
        super().__init__()
        self.num_classes = num_classes
        self.predictor = deepcopy(predictor)
        self.link = LogisticCumulativeLink(self.num_classes,
                                           init_cutpoints=init_cutpoints)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.link(self.predictor(X))
