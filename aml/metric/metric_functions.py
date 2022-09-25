import numpy as np



def countna(array:np.ndarray, normalise:bool=True,) -> float:
    '''
    Function to calculate the number of NAs in
    an array.

    Arguments
    ---------

    - array: numpy.ndarray:
        The array to calculate the number of missing
        values on.
    
    - normalise: bool, optional:
        Whether to return the values as a percentage.
        Defaults to :code:`True`.
    
    Returns
    ---------

    - countna: float:
        A :code:`float` equal to the number or proportion
        of missing values in an array.

    '''
    count_na = np.count_nonzero(np.isnan(array))
    if normalise:
        count_na *= 1/array.shape[0]
    return count_na


def interquartile_range(values:np.ndarray, lower:float=25, upper:float=75) -> float:
    '''
    Function to calculate the interquartile
    range of an array.

    Arguments
    ---------

    - array: numpy.ndarray:
        The array to calculate the IQR of.
    
    - lower: float, optional:
        The percentile of the lower quartile.
        Defaults to :code:`25`.
    
    - upper: float, optional:
        The percentile of the upper quartile.
        Defaults to :code:`75`.

    Returns
    ---------

    - iqr: float:
        A :code:`float` equal to the interquartile range.

    '''
    return np.subtract(*np.nanpercentile(values, [upper, lower]))


def format_mean_iqr_missing(
    values:np.ndarray, 
    string:str="{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))",
    ) -> str:
    '''
    A function useful for formatting a table with information 
    on the mean, IQR and missing rate of an attribute.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_iqr_missing(
                values=np.array([1,2,3,4,5]),
                string="{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))",
                )
        '3.00 (2) ((0%))'


    Arguments
    ---------

    - values: numpy.ndarray:
        The array to calculate the values on.
    
    - string: str, optional:
        The string that dictates the output.
        This should include somewhere :code:`{mean}`,
        :code:`{iqr}`, and :code:`{count_na}`.
        Defaults to :code:`"{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))"`.
    
    Returns
    ---------

    - stats: str:
        A string of the desired format with the 
        statistics included.

    
    '''
    mean = np.mean(values)
    iqr = interquartile_range(values)
    count_na = countna(values)*100
    return string.format(mean=mean, iqr=iqr, count_na=count_na)



def format_mean_std(
    values:np.ndarray, 
    string:str="{mean:.2f} ({std:.2f})",
    ) -> str:
    '''
    A function useful for formatting a table with information 
    on the mean and standard deviation an attribute.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_std(
                values=np.array([1,2,3,4,5]),
                string="{mean:.2f} ({std:.2f})",
                )
        '3.00 (1.41)'


    Arguments
    ---------

    - values: numpy.ndarray:
        The array to calculate the values on.
    
    - string: str, optional:
        The string that dictates the output.
        This should include somewhere :code:`{mean}` and
        :code:`{std}`.
        Defaults to :code:`"{mean:.2f} ({std:.2f})"`.
    
    Returns
    ---------

    - stats: str:
        A string of the desired format with the 
        statistics included.

    
    '''
    mean = np.mean(values)
    std = np.std(values)
    return string.format(mean=mean, std=std)