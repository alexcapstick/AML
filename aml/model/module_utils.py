import torch
import torch.nn as nn
import typing as t


class LambdaModule(nn.Module):
    def __init__(self, f: t.Callable[[torch.Tensor], torch.Tensor]):
        """
        This is a wrapper class for a function to be
        used as a module in a nn.Sequential.


        Examples
        ---------

        .. code-block::

            >>> from aml.model.module_utils import LambdaModule
            >>> import torch.nn as nn
            >>> import torch
            >>> model = nn.Sequential(
            ...      LambdaModule(lambda x: x**2),
            ...      nn.Linear(10, 1)
            ... )
            >>> x = torch.randn(10)
            >>> model(x)


        Arguments
        ---------

        - f: t.Callable[[torch.Tensor], torch.Tensor]
            The function to be wrapped as a module. All 
            arguments and keyword arguments passed using the forward
            method will be passed to the function.


        """
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)
