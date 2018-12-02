import numpy as np
from skorch import NeuralNet
import torch
from torch import nn

from medallion.callbacks import AscensionCallback
from medallion.losses import CumulativeLinkLoss
from medallion.models import OrdinalLogisticModel


SEED = 666


def test_loss_lowers_on_each_epoch():
    torch.manual_seed(SEED)
    num_classes = 5
    num_features = 5
    size = 200
    y = torch.randint(0, num_classes, (size, 1), dtype=torch.long)
    X = torch.rand((size, num_features))

    predictor = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.ReLU(),
        nn.Linear(num_features, 1)
    )

    skorch_model = NeuralNet(
        module=OrdinalLogisticModel,
        module__predictor=predictor,
        module__num_classes=num_classes,
        criterion=CumulativeLinkLoss,
        max_epochs=10,
        optimizer=torch.optim.Adam,
        lr=0.01,
        train_split=None,
        callbacks=[
            ('ascension', AscensionCallback()),
        ],
    )

    skorch_model.fit(X, y)
    losses = [epoch['train_loss'] for epoch in skorch_model.history]
    for idx, loss in enumerate(losses[:-1]):
        # Next epoch's loss is less than this epoch's loss.
        assert losses[idx + 1] < loss, 'Loss lowers on each epoch'
