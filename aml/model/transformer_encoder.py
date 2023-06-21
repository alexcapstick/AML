import torch
from torch import nn
import numpy as np
import typing
import math
from copy import deepcopy

from .base_model import BaseLightningModule
from .utils import get_function_from_name


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        seq_len: int,
        n_heads: int = 5,
        dim_feedforward: int = 2048,
        transformer_encoder_dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        max_len: int = 100,
    ):
        """
        This is a simple transformer encoder model.

        Arguments
        ---------

        - n_input: int:
            The size of the input feature dimension.

        - n_output: int:
            The size of the output dimension.

        - seq_len: int:
            The sequence length of the input.

        - n_heads: int, optional:
            The number of heads in the multi-head attention.
            Defaults to :code:`5`.

        - dim_feedforward: int, optional:
            The dimension of the feedforward network model.
            Defaults to :code:`2048`.

        - transformer_encoder_dim_feedforward: int, optional:
            The dimension of the feedforward network model in the transformer encoder.
            Defaults to :code:`2048`.

        - dropout: float, optional:
            The dropout value in each of the layers in the transformer encoder.
            Defaults to :code:`0.1`.

        - activation: str, optional:
            The activation function to be used in the hidden
            layers within the transformer encoder.
            Defaults to :code:`relu`.

        - max_len: int, optional:
            The maximum length of the input sequence.
            Defaults to :code:`100`.


        """
        super().__init__()

        self.positional_encoding = PositionalEncoding(d_model=n_input, max_len=max_len)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=n_input,
            nhead=n_heads,
            dim_feedforward=transformer_encoder_dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.fcs = nn.Sequential(
            nn.Linear(n_input * seq_len, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, n_output),
        )

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x)
        x = self.transformer_layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


class TransformerEncoderModel(BaseLightningModule):
    def __init__(
        self,
        n_output: int,
        n_input: int = None,
        seq_len: int = None,
        n_heads: int = 5,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        dim_feedforward: int = 2048,
        transformer_encoder_dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        max_len: int = 5000,
        optimizer: str = "adam",
        criterion: str = "mseloss",
        n_epochs: int = 10,
        accelerator="auto",
        **kwargs,
    ):
        """
        A simple Transformer Encoder model and
        built to be run similar to sklearn models.

        Examples
        ---------
        .. code-block::

            >>> transformer_model = TransformerEncoderModel(
            ...     n_input=100,
            ...     n_output=2,
            ...     n_heads=5,
            ...     n_epochs = 2,
            ...     verbose=True,
            ...     batch_size=10,
            ...     optimizer={'adam':{'lr':0.01}},
            ...     criterion='mseloss',
            ...     )
            >>> X = torch.tensor(np.random.random((5, 10,100))).float()
            >>> X_val = torch.tensor(np.random.random((5, 10,100))).float()
            >>> training_metrics = transformer_model.fit(X=X, X_val=X_val)
            >>> output = transformer_model.predict(X_test=X)


        Arguments
        ---------

        - n_output: int:
            The size of the output dimension.


        - n_input: int:
            The size of the input feature dimension.
            Defaults to :code:`None`.

        - seq_len: int:
            The sequence length of the input.
            Defaults to :code:`None`.

        - n_heads: int, optional:
            The number of heads in the multi-head attention.
            Defaults to :code:`5`.

        - dim_feedforward: int, optional:
            The dimension of the feedforward network model.
            Defaults to :code:`2048`.

        - transformer_encoder_dim_feedforward: int, optional:
            The dimension of the feedforward network model in the transformer encoder.
            Defaults to :code:`2048`.

        - lr: float, optional:
            The learning rate to be used in the optimizer.
            Defaults to :code:`0.001`.

        - weight_decay: float, optional:
            The weight decay to be used in the optimizer.
            Defaults to :code:`0.0`.

        - dropout: float, optional:
            The dropout value in each of the layers in the transformer encoder.
            Defaults to :code:`0.1`.

        - activation: str, optional:
            The activation function to be used in the hidden
            layers within the transformer encoder.
            Defaults to :code:`relu`.

        - max_len: int, optional:
            The maximum length of the input sequence.
            Defaults to :code:`5000`.

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
            If using a string, please use one of :code:`['adam', 'sgd', 'adagrad']`
            and provide :code:`lr` and :code:`weight_decay`.
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
                self.model_name = (
                    f"TransformerEncoder-{n_input}-{n_output}-{n_heads}"
                    f"-{dim_feedforward}-{dropout}"
                )

        if type(optimizer) != dict:
            optimizer = {optimizer: {"lr": lr, "weight_decay": weight_decay}}

        super().__init__(
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            accelerator=accelerator,
            **kwargs,
        )

        self.n_input = n_input
        self.n_output = n_output
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.transformer_encoder_dim_feedforward = transformer_encoder_dim_feedforward
        self.max_len = max_len
        self.dropout = dropout
        self.activation = activation
        self.predict_type = "classes"

        return

    def _build_model(self):
        self.model = TransformerEncoder(
            n_input=self.n_input,
            n_output=self.n_output,
            seq_len=self.seq_len,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            transformer_encoder_dim_feedforward=self.transformer_encoder_dim_feedforward,
            max_len=self.max_len,
            dropout=self.dropout,
            activation=self.activation,
        )
        return

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.log("train_loss", float(loss))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.model(x)
        loss = self.criterion(z, y)
        self.log("val_loss", float(loss), prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int):
        if type(batch) == list:
            batch = batch[0]
        if self.predict_type == "classes":
            _, predictions = torch.max(self.model(batch), dim=1)
            return predictions
        elif self.predict_type == "probabilities":
            return self.model(batch)

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
        if (self.seq_len is None) or (self.n_input is None):
            if not X is None:
                self.seq_len, self.n_input = X.shape[1], X.shape[2]
            else:
                raise ValueError(
                    "Please provide either X as an array or "
                    "set the seq_len and n_input"
                )

        self._build_model()

        return super().fit(
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
        return super().predict(
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
        return super().predict(
            X=X,
            y=y,
            test_loader=test_loader,
        )
