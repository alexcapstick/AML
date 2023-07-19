# This is an implementation of a Variational Autoencoder (VAE) model.
# It can be used by wrapping modules in the :code:`VAEEncoder` and :code:`VAEDecoder` classes.
# The :code:`VAEEncoder` class is used to encode the data into the mean and log-variance
# of the latent distribution. The :code:`VAEDecoder` class is used to decode the latent
# variables into the output data. The :code:`VAE` class is used to wrap the encoder and
# decoder together into a single model. The :code:`Prior` class is used to define the
# prior distribution over the latent variables.


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as t


# the following are based on functions provided during
# a generative modelling summer school GeMSS Copenhagen 2023

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.0e-5


def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1.0 - EPS))
    if reduction == "mean":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1.0 - EPS)
    log_p = x * torch.log(pp) + (1.0 - x) * torch.log(1.0 - pp)
    if reduction == "mean":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    D = x.shape[1]
    log_p = (
        -0.5 * D * torch.log(2.0 * PI)
        - 0.5 * log_var
        - 0.5 * torch.exp(-log_var) * (x - mu) ** 2.0
    )
    if reduction == "mean":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2.0 * PI) - 0.5 * x**2.0
    if reduction == "mean":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


class VAEEncoder(nn.Module):
    def __init__(self, encode_net: nn.Module):
        """
        The encoder network for the VAE model. This
        can be used to wrap any module for use in a VAE.


        Examples
        ---------

        The following example shows how to wrap a simple
        MLP for use in a VAE. The MLP takes in a tensor
        of shape (batch_size, 32) and outputs a tensor
        of shape (batch_size, 2*latent_dim). The first
        half of the output is used as the mean of the
        latent distribution, and the second half is used
        as the log-variance of the latent distribution.

        .. code-block::

            >>> encoder_net = nn.Sequential(
            ...     nn.Linear(32, 256),
            ...     nn.ReLU(),
            ...     nn.Linear(256, 256),
            ...     nn.ReLU(),
            ...     nn.Linear(256, 32*2),
            ... )
            >>> encoder = Encoder(encoder_net)


        Arguments
        ---------

        - encode_net: nn.Module:
            The network that will be used to encode the data.
            This network should output a tensor of
            shape (batch_size, 2*latent_dim). This is so that
            the first half of the output can be used as the
            mean of the latent distribution, and the second
            half can be used as the log-variance of the latent
            distribution.

        """
        super(VAEEncoder, self).__init__()

        self.encode_net = encode_net

    @staticmethod
    def reparameterization(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from a Gaussian.
        """
        # return z = mu + sigma * epsilon
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)

    # encode x into mu and log_var
    def encode(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input data into the mean and log-variance
        of the latent distribution.

        Arguments
        ---------

        - x: torch.Tensor:
            The input data.


        Returns
        -------

        - mu: torch.Tensor:
            The mean of the latent distribution.

        - log_var: torch.Tensor:
            The log-variance of the latent distribution.

        """
        return self.encode_net(x).chunk(2, dim=-1)

    # sample z from q(z|x)
    def sample(
        self,
        x: t.Union[None, torch.Tensor] = None,
        mu: t.Union[None, torch.Tensor] = None,
        log_var: t.Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from the latent distribution.

        Arguments
        ---------

        - x: torch.Tensor:
            The input data. This is only used if :code:`mu` and
            :code:`log_var` are not provided.
            Defaults to :code:`None`.

        - mu: torch.Tensor:
            The mean of the latent distribution. If this
            is not provided, then the input data will be
            encoded to obtain this value.
            Defaults to :code:`None`.

        - log_var: torch.Tensor:
            The log-variance of the latent distribution.
            If this is not provided, then the input data
            will be encoded to obtain this value.
            Defaults to :code:`None`.


        Returns
        -------

        - z: torch.Tensor:
            The sample from the latent distribution.

        """
        if mu == None or log_var == None:
            mu, log_var = self.encode_net(x).chunk(2, dim=-1)
        return self.reparameterization(mu, log_var)

    # calculate the log probability of z under q(z|x)
    def log_prob(
        self,
        x: t.Union[None, torch.Tensor] = None,
        mu: t.Union[None, torch.Tensor] = None,
        log_var: t.Union[None, torch.Tensor] = None,
        z: t.Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate the probability of the sample under the
        latent distribution.

        Arguments
        ---------

        - x: torch.Tensor:
            The input data. This is only used if :code:`mu`,
            :code:`log_var`, and :code:`x` are not provided.
            Defaults to :code:`None`.

        - mu: torch.Tensor:
            The mean of the latent distribution. If this
            is not provided, then the input data will be
            encoded to obtain this value.
            Defaults to :code:`None`.

        - log_var: torch.Tensor:
            The log-variance of the latent distribution.
            If this is not provided, then the input data
            will be encoded to obtain this value.
            Defaults to :code:`None`.

        - z: torch.Tensor:
            The sample from the latent distribution.
            If this is not provided, then the input data
            will be encoded to obtain this value.
            Defaults to :code:`None`.


        Returns
        -------

        - prob: torch.Tensor:
            The probability of the sample under the
            latent distribution.

        """
        if mu == None or log_var == None or z == None:
            mu, log_var = self.encode_net(x).chunk(2, dim=-1)
            z = self.reparameterization(mu, log_var)
        return log_normal_diag(z, mu, log_var, reduction=None, dim=None)

    # return z, mu, and log_var
    def forward(
        self, x: torch.Tensor
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass of the encoder network.

        Arguments
        ---------

        - x: torch.Tensor:
            The input data.


        Returns
        --------

        - z: torch.Tensor:
            The sample from the latent distribution.

        - mu: torch.Tensor:
            The mean of the latent distribution.

        - log_var: torch.Tensor:
            The log-variance of the latent distribution.


        """
        mu, log_var = self.encode(x)
        return self.sample(mu=mu, log_var=log_var), mu, log_var


list_distributions = ["categorical", "bernoulli", "standard_normal"]


class VAEDecoder(nn.Module):
    def __init__(
        self,
        decoder_net: nn.Module,
        distribution: t.Literal[
            "categorical", "bernoulli", "standard_normal"
        ] = "standard_normal",
        num_values: t.Union[int, None] = None,
        decoder_log_var: bool = False,
        out_shape: t.Union[t.Tuple[int, ...], None] = None,
    ):
        """
        The decoder network that maps the latent sample to the
        parameters of the distribution over the data.
        This is used as part of a VAE model.
        
        
        Examples
        ---------
        
        .. code-block::
        
            >>> decoder_net = nn.Sequential(
            ...     nn.Linear(32, 256),
            ...     nn.ReLU(),
            ...     nn.Linear(256, 256),
            ...     nn.ReLU(),
            ...     nn.Linear(256, 32),
            ... )
            >>> decoder = Decoder(decoder_net, distribution='bernoulli')
        
        
        Arguments
        ---------
        
        - decoder_net: nn.Module: 
            The decoder network that maps the latent sample to the
            parameters of the distribution over the data.

            - categorical distribution: the output \
                of this network should be of shape :code:`(batch_size, data_shape, num_values)`.
            
            - bernoulli distribution: the output \
                of this network should be of shape :code:`(batch_size, data_shape)`.
            
            - standard normal distribution: the output \
                of this network should be of shape :code:`(batch_size, data_shape)`\
                if the :code:`decoder_log_var` argument is :code:`False`, else the output \
                of this network should be of shape :code:`(batch_size, data_shape, 2)`.
        
        - distribution: t.Literal["categorical", "bernoulli", "standard_normal"], optional:
            The distribution of the output. 
            Defaults to :code:`'categorical'`.
        
        - num_values: t.Union[int, None], optional:
            The number of values in the categorical distribution if used. 
            Defaults to :code:`None`.
        
        - decoder_log_var: bool, optional:
            Whether the decoder network outputs the log-variance of the distribution.
            This is only used in the case of the standard normal distribution.
            If it does not, then the network should output shape :code:`(batch_size, data_shape)`.
            Defaults to :code:`False`.
        
        - out_shape: t.Union[t.Tuple[int, ...], None], optional:
            The shape of the output data.
            This is only used in the case of the :code:`decoder_log_var=False`.
            Defaults to :code:`None`.
        
        
        """
        super(VAEDecoder, self).__init__()

        if distribution == "categorical" and num_values == None:
            raise ValueError(
                "num_values must be specified for categorical distribution."
            )

        self.decoder_net = decoder_net
        self.distribution = distribution
        self.num_values = num_values
        self.decoder_log_var = decoder_log_var
        if not decoder_log_var:
            if out_shape == None:
                raise ValueError(
                    "out_shape must be specified if decoder_log_var is True."
                )
            self.log_var = nn.Parameter(torch.zeros(out_shape))
            torch.nn.init.normal_(self.log_var, mean=0, std=0.1)

    # calculates the parameteres of the distribution p(x|z)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent sample to the parameters of the
        distribution over the data.


        Arguments
        ---------

        - z: torch.Tensor:
            The latent sample.


        Returns
        --------

        - out: torch.Tensor:
            The parameters of the distribution over the data.


        """
        x_hat = self.decoder_net(z)

        if self.distribution == "categorical":
            # x_hat will be of shape (batch_size, data_shape, num_values)
            batch_size = x_hat.shape[0]
            data_shape = x_hat.shape[1:-1]
            x_hat = x_hat.reshape(batch_size, *data_shape, self.num_values)
            return torch.softmax(x_hat, dim=-1)

        elif self.distribution == "bernoulli":
            # x_hat will be of shape (batch_size, data_shape)
            return torch.sigmoid(x_hat)

        elif self.distribution == "standard_normal":
            # x_hat will be of shape (batch_size, data_shape, 2)
            if self.decoder_log_var:
                mu, log_var = x_hat[..., 0], x_hat[..., 1]
                return torch.concat([mu, log_var], axis=-1)
            # x_hat will be of shape (batch_size, data_shape)
            else:
                mu = x_hat
                return torch.cat([mu, self.log_var.expand(*mu.shape)], axis=-1)

        else:
            raise ValueError("Distribution not supported")

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        """
        Samples p(x|z).


        Arguments
        ---------

        - z: torch.Tensor:
            The latent sample to decode and use
            to generate a sample of x.


        Returns
        --------

        - out: torch.Tensor:
            The generated sample


        """

        out = self.decode(z)

        if self.distribution == "categorical":
            mu = out
            batch_size, n_pixels = mu.shape
            return torch.multinomial(mu.view(batch_size * n_pixels, -1), 1).view(
                batch_size, n_pixels
            )

        elif self.distribution == "bernoulli":
            mu = out
            return torch.bernoulli(mu)

        elif self.distribution == "standard_normal":
            mu, log_var_d = out.chunk(2, dim=-1)
            return torch.distributions.Normal(mu, torch.exp(0.5 * log_var_d)).sample()

    # calculate the log probability of x under p(x|z)
    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        The log probability of x under p(x|z).


        Arguments
        ---------

        - x: torch.Tensor:
            The data.

        - z: torch.Tensor:
            The latent sample.


        Returns
        --------

        - out: torch.Tensor:
            The log probability of x under p(x|z).


        """
        if self.distribution == "categorical":
            return log_categorical(
                x, self.decode(z), num_classes=self.num_values, reduction="sum", dim=-1
            ).sum(-1)

        elif self.distribution == "bernoulli":
            return log_bernoulli(x, self.decode(z), reduction="sum", dim=-1).sum(-1)

        elif self.distribution == "standard_normal":
            mu, log_var = self.decode(z).chunk(2, dim=-1)
            return log_normal_diag(x, mu, log_var, reduction="sum", dim=-1).sum(-1)

        else:
            raise ValueError("Distribution not supported")

    def forward(
        self,
        z: torch.Tensor,
        x: t.Union[None, torch.Tensor] = None,
        type: t.Literal["log_prob", "sample"] = "log_prob",
    ) -> torch.Tensor:
        """
        The forward pass through the decoder network.


        Arguments
        ---------

        - z: torch.Tensor:
            The latent sample.

        - x: t.Union[None,torch.Tensor], optional:
            The data. No data is required if using
            :code:`type="sample"`.
            Defaults to :code:`None`.

        - type: t.Literal['log_prob', 'sample'], optional:
            Whether to return a sample or the log probability.
            Defaults to :code:`'log_prob'`.


        Returns
        --------

        - out: torch.Tensor:
            A sample or the log probability.


        """
        if type == "log_prob":
            assert x is not None, "x must be specified for log_prob"
            return self.log_prob(x, z)
        elif type == "sample":
            return self.sample(z)


class StandardNormalPrior(nn.Module):
    def __init__(self, L: int):
        """
        A prior that can be used in the VAE.
        This prior is a standard normal distribution.

        Arguments
        ---------

        - L: int:
            The dimensionality of the latent space.

        """
        super(StandardNormalPrior, self).__init__()
        self.type = type
        self.L = L

        self.mu = nn.parameter.Parameter(torch.zeros(size=(L,)), requires_grad=False)
        self.log_var = nn.parameter.Parameter(
            torch.ones(size=(L,)), requires_grad=False
        )

        self.prior = torch.distributions.Normal(self.mu, self.log_var)

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample the prior.

        Arguments
        ---------

        - batch_size: int:
            The number of samples to draw.


        Returns
        --------

        - out: torch.Tensor:
            The samples.

        """
        return self.prior.sample((batch_size,))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        The log probability of a sample.


        Arguments
        ---------

        - z: torch.Tensor:
            The sample


        Returns
        --------

        - out: torch.Tensor:
            The log probability.

        """
        return self.prior.log_prob(z)

    def forward(
        self, z: torch.Tensor, type: t.Literal["log_prob", "sample"] = "log_prob"
    ) -> torch.Tensor:
        """
        The forward function.


        Arguments
        ---------

        - z: torch.Tensor:
            The sample.

        - type: t.Literal['log_prob', 'sample'], optional:
            Whether to sample from the distribution or calculate the log probability.
            Defaults to :code:`'log_prob'`.


        Returns
        --------

        - out: torch.Tensor:
            A sample or the log probability.

        """
        if type == "log_prob":
            return self.log_prob(z)
        elif type == "sample":
            return self.sample(z)


class GaussianMixPrior(nn.Module):
    def __init__(self, L: int, num_components: int):
        """
        Based on code here: https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_priors_example.ipynb


        Arguments
        ---------

        - L: torch.Tensor:
            The number of latent dimensions.

        - num_components: int, optional:
            The number of Gaussian components.
            Defaults to :code:`1`.

        """
        super(GaussianMixPrior, self).__init__()

        assert num_components > 1, "num_components must be greater than 1"

        self.type = type
        self.L = L
        self.num_components = num_components

        self.mus = nn.parameter.Parameter(torch.zeros(size=(num_components, L)))
        self.log_vars = nn.parameter.Parameter(torch.zeros(size=(num_components, L)))
        self.w = nn.parameter.Parameter(torch.zeros(size=(num_components, 1, 1)))

        nn.init.normal_(self.mus, mean=0, std=0.1)
        nn.init.normal_(self.log_vars, mean=0, std=0.1)
        nn.init.normal_(self.w, mean=0, std=0.1)

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample the prior.

        Arguments
        ---------

        - batch_size: int:
            The number of samples to draw.


        Returns
        --------

        - out: torch.Tensor:
            The samples.

        """

        w = F.softmax(self.w, dim=0).squeeze()  # num_components
        indexes = torch.multinomial(w, batch_size, replacement=True)
        mus = self.mus  # num_components x L
        log_vars = self.log_vars  # num_components x L

        eps = torch.randn(batch_size, self.L, device=self.mus.device)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = mus[[indx]] + eps[[i]] * torch.exp(log_vars[[indx]])
            else:
                z = torch.cat(
                    (z, mus[[indx]] + eps[[i]] * torch.exp(log_vars[[indx]])), 0
                )
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        The log probability of a sample.


        Arguments
        ---------

        - z: torch.Tensor:
            The sample


        Returns
        --------

        - out: torch.Tensor:
            The log probability.

        """

        w = F.softmax(self.w, dim=0)
        z = z.unsqueeze(0)  # 1 x B x L
        mus = self.mus.unsqueeze(1)  # num_components x 1 x L
        log_vars = self.log_vars.unsqueeze(1)  # num_components x 1 x L

        log_p = log_normal_diag(z, mus, log_vars) + torch.log(
            w
        )  # num_components x B x L

        return torch.logsumexp(log_p, dim=0, keepdim=False)  # B x L

    def forward(
        self, z: torch.Tensor, type: t.Literal["log_prob", "sample"] = "log_prob"
    ) -> torch.Tensor:
        """
        The forward function.


        Arguments
        ---------

        - z: torch.Tensor:
            The sample.

        - type: t.Literal['log_prob', 'sample'], optional:
            Whether to sample from the distribution or calculate the log probability.
            Defaults to :code:`'log_prob'`.


        Returns
        --------

        - out: torch.Tensor:
            A sample or the log probability.

        """
        if type == "log_prob":
            return self.log_prob(z)
        elif type == "sample":
            return self.sample(z)


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        prior: nn.Module,
        KL_beta: float = 1.0,
    ):
        """
        A Variational Autoencoder (VAE). This class is a wrapper around the
        encoder, decoder, and prior.



        Examples
        ---------

        .. code-block::

            >>> encoder = Encoder(...)
            >>> decoder = Decoder(...)
            >>> prior = Prior(...)
            >>> vae = VAE(encoder, decoder, prior)


        Arguments
        ---------

        - encoder: nn.Module:
            The encoder. Please see the documentation for VAEEncoder
            for more details.

        - decoder: nn.Module:
            The decoder. Please see the documentation for VAEDecoder
            for more details.

        - prior: nn.Module:
            The prior. This should have methods for sampling
            and calculating the log probability. Specifically,
            it should implement the following methods:
            - sample(batch_size: int) -> torch.Tensor[batch_size, latent_dim]
            - log_prob(z: torch.Tensor) -> torch.Tensor[batch_size, latent_dim]

        - KL_beta: float, optional:
            The KL weight used in the loss. This will likely need
            to be tuned for each dataset.
            Defaults to :code:`1.0`.

        """

        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.KL_beta = KL_beta

    def sample(self, batch_size: int = 64) -> torch.Tensor:
        """
        Sample from the VAE.


        Examples
        ---------

        .. code-block::

            >>> vae = VAE(...)
            >>> samples = vae.sample(64)


        Arguments
        ---------

        - batch_size: int, optional:
            The number of samples to generate.
            Defaults to :code:`64`.


        Returns
        --------

        - out: torch.Tensor:
            The samples.


        """
        z = self.prior.sample(batch_size)
        return self.decoder.sample(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input data.


        Examples
        ---------

        .. code-block::

            >>> vae = VAE(...)
            >>> x = torch.randn(64, 784)
            >>> reconstruction = vae.reconstruct(x)


        Arguments
        ---------

        - x: torch.Tensor:
            The input data.


        Returns
        --------

        - out: torch.Tensor:
            The reconstructions.


        """
        z, _, _ = self.encoder(x)
        return self.decoder.sample(z)

    def forward(
        self, x: torch.Tensor, reduction: t.Literal["mean", "sum", None] = "mean"
    ) -> torch.Tensor:
        """
        The forward pass through the VAE. This will return the negative ELBO.


        Examples
        ---------

        The following provides the loss for a batch of data:

        .. code-block::

            >>> vae = VAE(...)
            >>> x = torch.randn(64, 784)
            >>> loss = vae(x)


        Arguments
        ---------

        - x: torch.Tensor:
            The input data.

        - reduction: t.Literal['mean', 'sum'], optional:
            Whether to sum or mean the loss before returning.
            Defaults to :code:`'mean'`.


        Returns
        --------

        - out: torch.Tensor:
            The loss.


        """
        # encode
        z, mu, log_var = self.encoder(x)

        # decode
        reconstruction_term = self.decoder.log_prob(x, z).view(x.shape[0])
        kl_term = (
            (
                self.prior.log_prob(z)
                - self.encoder.log_prob(z=z, mu=mu, log_var=log_var)
            )
            .view(x.shape[0], -1)
            .sum(-1)
        )

        if reduction == "mean":
            return -(reconstruction_term + self.KL_beta * kl_term).mean(-1)
        elif reduction == "sum":
            return -(reconstruction_term + self.KL_beta * kl_term).sum(-1)
        elif reduction is None:
            return -(reconstruction_term + self.KL_beta * kl_term)
        else:
            raise ValueError('reduction must be either "mean" or "sum"')
