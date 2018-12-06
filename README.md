# spacecutter

`spacecutter` is a library for implementing ordinal regression models in PyTorch. The library consists of models and loss functions. It is recommended to use [skorch](http://skorch.readthedocs.io/) to wrap the models to make them compatible with scikit-learn.

## Installation

```bash
pip install spacecutter
```

## Usage

### Models

Define any PyTorch model you want that generates a single, scalar prediction value. This will be our `predictor` model. This model can then be wrapped with `spacecutter.models.OrdinalLogisticModel` which will convert the output of the `predictor` from a single number to an array of ordinal class probabilities. The following example shows how to do this for a two layer neural network `predictor` for a problem with three ordinal classes.

```python
import numpy as np
import torch
from torch import nn

from spacecutter.models import OrdinalLogisticModel


X = np.array([[0.5, 0.1, -0.1],
              [1.0, 0.2, 0.6],
              [-2.0, 0.4, 0.8]],
             dtype=np.float32)

y = np.array([0, 1, 2]).reshape(-1, 1)

num_features = X.shape[1]
num_classes = len(np.unique(y))

predictor = nn.Sequential(
    nn.Linear(num_features, num_features),
    nn.ReLU(),
    nn.Linear(num_features, 1)
)

model = OrdinalLogisticModel(predictor, num_classes)

y_pred = model(torch.as_tensor(X))

print(y_pred)

# tensor([[0.2325, 0.2191, 0.5485],
#         [0.2324, 0.2191, 0.5485],
#         [0.2607, 0.2287, 0.5106]], grad_fn=<CatBackward>)

```

### Training

It is recommended to use [skorch](http://skorch.readthedocs.io/) to train `spacecutter` models. The following shows how to train the model from the previous section using cumulative link loss with `skorch`:

```python
from skorch import NeuralNet

from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss

skorch_model = NeuralNet(
    module=OrdinalLogisticModel,
    module__predictor=predictor,
    module__num_classes=num_classes,
    criterion=CumulativeLinkLoss,
    train_split=None,
    callbacks=[
        ('ascension', AscensionCallback()),
    ],
)

skorch_model.fit(X, y)

```

Note that we must add the `AscensionCallback`. This ensures that the ordinal cutpoints stay in ascending order. While ideally this constraint would be factored directly into the model optimization, `spacecutter` currently hacks an SGD-compatible solution by utilizing a post-backwards-pass callback to clip the cutpoint values.
