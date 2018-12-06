import torch
from torch import nn

from spacecutter import callbacks
from spacecutter.models import OrdinalLogisticModel


def test_clip_ensures_sorted_cutpoints():
    predictor = nn.Linear(5, 1)
    model = OrdinalLogisticModel(predictor, 4, init_cutpoints='ordered')
    # model.link.cutpoints = [-1.5, -0.5, 0.5]

    # The following is necessary to be able to manually modify the cutpoints.
    for p in model.parameters():
        p.requires_grad = False
    ascension = callbacks.AscensionCallback()

    # Make cutpoints not in sorted order
    model.link.cutpoints += torch.FloatTensor([0, 5, 0])
    # model.link.cutpoints = [-1.5, 4.5, 0.5]

    # Apply the clipper
    model.apply(ascension.clip)

    assert torch.allclose(model.link.cutpoints.data,
                          torch.FloatTensor([-1.5, 0.5, 0.5]))


def test_margin_is_satisfied():
    predictor = nn.Linear(5, 1)
    model = OrdinalLogisticModel(predictor, 4, init_cutpoints='ordered')
    # model.link.cutpoints = [-1.5, -0.5, 0.5]

    # The following is necessary to be able to manually modify the cutpoints.
    for p in model.parameters():
        p.requires_grad = False
    ascension = callbacks.AscensionCallback(margin=0.5)

    # Make cutpoints not in sorted order
    model.link.cutpoints += torch.FloatTensor([0, 5, 0])
    # model.link.cutpoints = [-1.5, 4.5, 0.5]

    # Apply the clipper
    model.apply(ascension.clip)

    assert torch.allclose(model.link.cutpoints.data,
                          torch.FloatTensor([-1.5, 0.0, 0.5]))
