import numpy as np
import pytest
import torch

from spacecutter import losses


class Test_reduction:

    def test__reduction_mean(self):
        loss = torch.FloatTensor([[6.0], [4.0]])
        output = losses._reduction(loss, 'elementwise_mean')
        assert output.item() == 5.0

    def test__reduction_sum(self):
        loss = torch.FloatTensor([[6.0], [4.0]])
        output = losses._reduction(loss, 'sum')
        assert output.item() == 10.0

    def test__reduction_none(self):
        loss = torch.FloatTensor([[6.0], [4.0]])
        output = losses._reduction(loss, 'none')
        assert torch.allclose(loss, output)

    def test__reduction_invalid_reduction(self):
        loss = torch.FloatTensor([[6.0], [4.0]])
        with pytest.raises(ValueError):
            losses._reduction(loss, 'invalid')


class Test_cumulative_link_loss:

    y_pred = torch.FloatTensor([[0.25, 0.5, 0.25],
                                [0.1, 0.6, 0.3]])

    def test_default_behavior(self):
        y_true = torch.LongTensor([[1], [2]])
        output = losses.cumulative_link_loss(self.y_pred, y_true)
        # pickes out log(0.5) and log(0.3) then takes the average
        expected = pytest.approx(0.94856)
        assert output.item() == expected

    def test_class_weights(self):
        y_true = torch.LongTensor([[1], [2]])
        class_weights = np.array([5, 10, 20])
        output = losses.cumulative_link_loss(self.y_pred, y_true,
                                             class_weights=class_weights)
        # picks out log(0.5) and log(0.3), multiplies by 10 and 20, respectively.
        expected = pytest.approx(15.5054639)
        assert output.item() == expected

    def test_extremely_small_loss_is_clipped(self):
        y_true = torch.LongTensor([[1],])
        y_pred = torch.FloatTensor([[0.25, 1e-20, 0.75]])
        output = losses.cumulative_link_loss(y_pred, y_true)
        assert output.item() < 35
