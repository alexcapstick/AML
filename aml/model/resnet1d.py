"""
These models are still in development!!
"""

import torch
from torch import nn
import numpy as np
import typing
from copy import deepcopy

from .base_model import BaseLightningModule
from .utils import get_function_from_name

# ResNet block that will be used multiple times in the resnet model
class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_channels: int,
        out_channels: int,
        out_dim: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
    ):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.out_dim = out_dim

        self.x1 = nn.Sequential(
            nn.Conv1d(
                input_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=False,
                padding="same",
            ),
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=True,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.x2 = nn.Sequential(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                stride=input_dim // out_dim,
            )
        )

        self.y1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=False,
            )
        )

        self.xy1 = nn.Sequential(
            nn.BatchNorm1d(
                num_features=out_channels,
                affine=False,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        return

    # resizing the skip connection if needed, and then using 1d Convolution
    def _skip_connection(self, y):
        downsample = self.input_dim // self.out_dim
        if downsample > 1:

            same_pad = np.ceil(
                0.5
                * (
                    (y.size(-1) // self.out_dim) * (self.out_dim - 1)
                    - y.size(-1)
                    + downsample
                )
            )
            if same_pad < 0:
                same_pad = 0
            y = nn.functional.pad(y, (int(same_pad), int(same_pad)), "constant", 0)
            y = nn.MaxPool1d(
                kernel_size=downsample,
                stride=downsample,
            )(y)

        elif downsample == 1:
            pass
        else:
            raise ValueError("Size of input should always decrease.")
        y = self.y1(y)

        return y

    def forward(self, inputs):
        x, y = inputs

        # y
        y = self._skip_connection(y)

        # x
        x = self.x1(x)
        same_pad = np.ceil(
            0.5
            * (
                (x.size(-1) // self.out_dim) * (self.out_dim - 1)
                - x.size(-1)
                + self.kernel_size
            )
        )
        if same_pad < 0:
            same_pad = 0
        x = nn.functional.pad(x, (int(same_pad), int(same_pad)), "constant", 0)
        x = self.x2(x)

        # xy
        xy = x + y
        y = x
        xy = self.xy1(xy)

        return [xy, y]


# main resnet model, made of 4 resnet blocks
class ResNet1D(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        input_channels: int = 64,
        n_output: int = 10,
        kernel_size: int = 16,
        dropout_rate: float = 0.2,
    ):
        """
        Model with 4 :code:`ResBlock`s, in which
        the number of channels increases linearly
        and the output dimensions decreases
        exponentially. This model will
        require the input dimension to be of at least
        256 in size. This model is designed for sequences,
        and not images. The expected input is of the type::

            [batch_size, n_filters, sequence_length]


        Examples
        ---------

        .. code-block::

            >>> model = ResNet(
                    input_dim=4096,
                    input_channels=64,
                    kernel_size=16,
                    n_output=5,
                    dropout_rate=0.2,
                    )
            >>> model(
                    torch.rand(1,64,4096)
                    )
            tensor([[0.3307, 0.4782, 0.5759, 0.5214, 0.6116]], grad_fn=<SigmoidBackward0>)


        Arguments
        ---------

        - input_dim: int, optional:
            The input dimension of the input. This
            is the size of the final dimension, and
            the sequence length.
            Defaults to :code:`4096`.

        - input_channels: int, optional:
            The number of channels in the input.
            This is the second dimension. It is the
            number of features for each sequence element.
            Defaults to :code:`64`.

        - n_output: int, optional:
            The number of output classes in
            the prediction.
            Defaults to :code:`10`.

        - kernel_size: int, optional:
            The size of the kernel filters
            that will act over the sequence.
            Defaults to :code:`16`.

        - dropout_rate: float, optional:
            The dropout rate of the ResNet
            blocks. This should be a value
            between :code:`0` and  :code:`1`.
            Defaults to :code:`0.2`.

        """
        super(ResNet1D, self).__init__()

        self.x1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(
                num_features=input_channels,
                affine=False,
            ),
            nn.ReLU(),
        )

        self.x2 = nn.Sequential(
            ResBlock(
                input_dim=input_dim,  # 4096
                input_channels=input_channels,  # 64
                out_channels=2 * input_channels // 1,  # 128
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 4,  # 1024,
                dropout_rate=dropout_rate,
            ),
            ResBlock(
                input_dim=input_dim // 4,  # 1024
                input_channels=2 * input_channels // 1,  # 128
                out_channels=3 * input_channels // 1,  # 192
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 16,  # 256
                dropout_rate=dropout_rate,
            ),
            ResBlock(
                input_dim=input_dim // 16,  # 256
                input_channels=3 * input_channels // 1,  # 192
                out_channels=4 * input_channels // 1,  # 256
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 64,  # 64
                dropout_rate=dropout_rate,
            ),
            ResBlock(
                input_dim=input_dim // 64,  # 64
                input_channels=4 * input_channels // 1,  # 256
                out_channels=5 * input_channels // 1,  # 320
                kernel_size=kernel_size,  # 16
                out_dim=input_dim // 256,  # 16
                dropout_rate=dropout_rate,
            ),
        )

        self.x3 = nn.Flatten()  # flattens the data
        self.x4 = nn.Sequential(
            nn.Linear(
                (input_dim // 256) * (5 * input_channels // 1),
                n_output,
            )
        )

    def forward(self, x):

        x = self.x1(x)
        x, _ = self.x2([x, x])
        x = self.x3(x)
        x = self.x4(x)

        return x


class ResNet1DModel(BaseLightningModule):
    def __init__(
        self,
        input_dim: int = 4096,
        input_channels: int = 64,
        n_output: int = 10,
        kernel_size: int = 16,
        dropout_rate: float = 0.2,
        optimizer: dict = {"adam": {"lr": 0.01}},
        criterion: str = "mseloss",
        n_epochs: int = 10,
        accelerator="auto",
        **kwargs,
    ):
        """


        Model with 4 :code:`ResBlock`s, in which
        the number of channels increases linearly
        and the output dimensions decreases
        exponentially. This model will
        require the input dimension to be of at least
        256 in size. This model is designed for sequences,
        and not images. The expected input is of the type::

            [n_batches, n_filters, sequence_length]


        Examples
        ---------

        .. code-block::

            >>> model = ResNet(
                    input_dim=4096,
                    input_channels=64,
                    kernel_size=16,
                    n_output=5,
                    dropout_rate=0.2,
                    )
            >>> model.fit(torch.rand(1,64,4096))


        Arguments
        ---------

        - input_dim: int, optional:
            The input dimension of the input. This
            is the size of the final dimension, and
            the sequence length.
            Defaults to :code:`4096`.

        - input_channels: int, optional:
            The number of channels in the input.
            This is the second dimension. It is the
            number of features for each sequence element.
            Defaults to :code:`64`.

        - n_output: int, optional:
            The number of output classes in
            the prediction.
            Defaults to :code:`10`.

        - kernel_size: int, optional:
            The size of the kernel filters
            that will act over the sequence.
            Defaults to :code:`16`.

        - dropout_rate: float, optional:
            The dropout rate of the ResNet
            blocks. This should be a value
            between :code:`0` and  :code:`1`.
            Defaults to :code:`0.2`.

        - criterion: str or torch.nn.Module:
            The criterion that is used to calculate the loss.
            If using a string, please use one of :code:`['mseloss', 'celoss']`
            Defaults to :code:`mseloss`.

        - optimizer: dict, optional:
            A dictionary containing the optimizer name as keys and
            a dictionary as values containing the arguments as keys.
            For example: :code:`{'adam':{'lr':0.01}}`.
            The key can also be a :code:`torch.optim` class,
            but not initiated.
            For example: :code:`{torch.optim.Adam:{'lr':0.01}}`.
            Defaults to :code:`{'adam':{'lr':0.01}}`.

        - n_epochs: int, optional:
            The number of epochs to run the training for.
            Defaults to :code:`10`.

        - accelerator: str, optional:
            The device to use for training. Please use
            any of :code:`(“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “auto”)`.
            Defaults to :code:`'auto'`

        - kwargs: optional:
            These keyword arguments will be passed to
            :code:`dcarte_transform.model.base_model.BaseModel`.


        """

        if "model_name" in kwargs:
            if kwargs["model_name"] is None:
                self.model_name = f"ResNet1D-{input_dim}-{n_output}"

        super(ResNet1DModel, self).__init__(
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            accelerator=accelerator,
            **kwargs,
        )

        self.input_dim = input_dim
        self.input_channels = input_channels
        self.n_output = n_output
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.predict_type = "classes"

        return

    def _build_model(self):
        self.resnet = ResNet1D(
            input_dim=self.input_dim,
            input_channels=self.input_channels,
            n_output=self.n_output,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
        )
        return

    def forward(self, X):
        return self.resnet(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.resnet(x)
        loss = self.criterion(z, y)
        self.log("train_loss", float(loss))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.resnet(x)
        loss = self.criterion(z, y)
        self.log("val_loss", float(loss), prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int):
        if type(batch) == list:
            batch = batch[0]
        if self.predict_type == "classes":
            _, predictions = torch.max(self.resnet(batch), dim=1)
            return predictions
        elif self.predict_type == "probabilities":
            return self(batch)

    def fit(
        self,
        X: np.array = None,
        y=None,
        train_loader: torch.utils.data.DataLoader = None,
        X_val: typing.Union[np.array, None] = None,
        y_val=None,
        val_loader: torch.utils.data.DataLoader = None,
        **kwargs,
    ):
        """
        This is used to fit the model. Please either use
        the :code:`train_loader` or :code:`X` and :code:`y`.
        This corresponds to using either a torch DataLoader
        or a numpy array as the training data. If using
        the :code:`train_loader`, ensure each iteration returns
        :code:`[X, X]`.

        Arguments
        ---------

        - X: numpy.array or None, optional:
            The input array to fit the model on.
            Defaults to :code:`None`.

        - train_loader: torch.utils.data.DataLoader or None, optional:
            The training data, which contains the input and the targets.
            Defaults to :code:`None`.

        - X_val: numpy.array or None, optional:
            The validation input to calculate validation
            loss on when training the model.
            Defaults to :code:`None`

        - val_loader: torch.utils.data.DataLoader or None, optional:
            The validation data, which contains the input and the targets.
            Defaults to :code:`None`.

        """

        self._build_model()

        return super(ResNet1DModel, self).fit(
            train_loader=train_loader,
            X=X,
            y=y,
            val_loader=val_loader,
            X_val=X_val,
            y_val=y_val,
            **kwargs,
        )

    def predict(
        self,
        X: np.array = None,
        y: np.array = None,
        test_loader: torch.utils.data.DataLoader = None,
    ):
        """
        Method for making predictions on a test loader.

        Arguments
        ---------

        - X: numpy.array or None, optional:
            The input array to test the model on.
            Defaults to :code:`None`.

        - y: numpy.array or None, optional:
            The target array to test the model on. If set to :code:`None`,
            then :code:`targets_too` will automatically be set to :code:`False`.
            Defaults to :code:`None`.

        - test_loader: torch.utils.data.DataLoader or None, optional:
            A data loader containing the test data.
            Defaults to :code:`None`.


        Returns
        --------

        - output: torch.tensor:
            The resutls from the predictions


        """
        self.predict_type = "classes"
        return super(ResNet1DModel, self).predict(
            X=X,
            y=y,
            test_loader=test_loader,
        )

    def predict_proba(
        self,
        X: np.array = None,
        y: np.array = None,
        test_loader: torch.utils.data.DataLoader = None,
    ):
        """
        Method for making probability predictions on a test loader.

        Arguments
        ---------

        - X: numpy.array or None, optional:
            The input array to test the model on.
            Defaults to :code:`None`.

        - y: numpy.array or None, optional:
            The target array to test the model on. If set to :code:`None`,
            then :code:`targets_too` will automatically be set to :code:`False`.
            Defaults to :code:`None`.

        - test_loader: torch.utils.data.DataLoader or None, optional:
            A data loader containing the test data.
            Defaults to :code:`None`.


        Returns
        --------

        - output: torch.tensor:
            The resutls from the predictions


        """
        self.predict_type = "probabilities"
        return super(ResNet1DModel, self).predict(
            X=X,
            y=y,
            test_loader=test_loader,
        )
