import numpy as np
import pandas as pd
import typing
from sklearn.metrics._classification import (
    _check_set_wise_labels,
    multilabel_confusion_matrix,
    _prf_divide,
)
from sklearn.metrics import precision_recall_curve, auc
from scipy import stats


def countna(
    array: np.ndarray,
    normalise: bool = True,
) -> float:
    """
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

    """
    count_na = np.count_nonzero(pd.isna(array))
    if normalise:
        count_na *= 1 / array.shape[0]
    return count_na


def interquartile_range(
    values: np.ndarray, lower: float = 25, upper: float = 75
) -> float:
    """
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

    """
    return np.subtract(*np.nanpercentile(values, [upper, lower]))


def format_mean_iqr_missing(
    values: np.ndarray,
    string: str = "{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))",
) -> str:
    """
    A function useful for formatting a table with information
    on the mean, IQR and missing rate of an attribute.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_iqr_missing(
        ...     values=np.array([1,2,3,4,5]),
        ...     string="{mean:.2f} ({iqr:.2f}) (({count_na:.0f}%))",
        ...     )
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


    """
    mean = np.mean(values)
    iqr = interquartile_range(values)
    count_na = countna(values) * 100
    return string.format(mean=mean, iqr=iqr, count_na=count_na)


def format_mean_std(
    values: np.ndarray,
    string: str = "{mean:.2f} ({std:.2f})",
) -> str:
    """
    A function useful for formatting a table with information
    on the mean and standard deviation an attribute.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_std(
        ...     values=np.array([1,2,3,4,5]),
        ...     string="{mean:.2f} ({std:.2f})",
        ...     )
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


    """
    mean = np.mean(values)
    std = np.std(values)
    return string.format(mean=mean, std=std)


def format_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    string: str = "{mean:.2f} ({confidence_l:.2f} - {confidence_u:.2f})",
    distribution: typing.Literal["norm", "t"] = "t",
) -> str:
    """
    A function useful for formatting a table with information
    on the mean and confidence intervals of a set of values.
    You may use the normal distribution or the t-distribution
    to calculate the confidence intervals.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> format_mean_confidence(
        ...     values=np.array([1,2,3,4,5]),
        ...     confidence=0.95,
        ...     string="{mean:.2f} ({confidence_l:.2f} - {confidence_u:.2f})",
        ...     distribution='t',
        ... )
        '3.00 (1.04 - 4.96)'


    Arguments
    ---------

    - values: numpy.ndarray:
        The array to calculate the values on.

    - confidence: float, optional:
        The confidence interval to calculate.
        Defaults to :code:`0.95`.

    - string: str, optional:
        The string that dictates the output.
        This should include somewhere :code:`{mean}`,
        :code:`{confidence_l}`, and :code:`{confidence_u}`.
        Defaults to :code:`"{mean:.2f} ({confidence_l:.2f} - {confidence_u:.2f})"`.

    - distribution: str, optional:
        The distribution to use for calculating the confidence
        intervals. Can be either :code:`'norm'` or :code:`'t'`.
        Defaults to :code:`'t'`.

    Returns
    ---------

    - stats: str:
        A string of the desired format with the
        statistics included.


    """

    mean = np.mean(values)

    if distribution == "norm":
        confidence_l, confidence_u = stats.norm.interval(
            alpha=confidence,
            loc=np.mean(values),
            scale=stats.sem(values),
        )
    elif distribution == "t":
        confidence_l, confidence_u = stats.t.interval(
            alpha=confidence,
            df=len(values) - 1,
            loc=np.mean(values),
            scale=stats.sem(values),
        )

    return string.format(
        mean=mean, confidence_l=confidence_l, confidence_u=confidence_u
    )


def sensitivity_specificity_ppv_npv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: typing.Union[np.ndarray, None] = None,
    pos_label: typing.Union[str, int] = 1,
    average: typing.Union[str, None] = None,
    warn_for: typing.Union[str, typing.Tuple[str]] = ("sensitivity", "specificity"),
    sample_weight: typing.Union[np.ndarray, None] = None,
    zero_division: typing.Union[int, str] = "warn",
) -> typing.Tuple[float]:
    """
    A function that calculates the sensitivity, 
    specificity, ppv, and npv between two arrays. 
    This is modelled on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> sensitivity_specificity_ppv_npv(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_pred=np.array([0,0,0,1,0]),
        ...     )
        (0.5, 1.0, 1.0, 0.75)


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_pred: np.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:

        - :code:`'binary'`: \
        Only report results for the class specified by :code:`pos_label`. \
        This is applicable only if targets (`y_{true,pred}`) are binary.

        - :code:`'micro'`: \
        Calculate metrics globally by counting the total true positives, \
        false negatives and false positives.

        - :code:`'macro'`: \
        Calculate metrics for each label, and find their unweighted \
        mean.  This does not take label imbalance into account.

        - :code:`'weighted'`: \
        Calculate metrics for each label, and find their average weighted \
        by support (the number of true instances for each label). This \
        alters 'macro' to account for label imbalance; it can result in an \
        F-score that is not between precision and recall.

        - :code:`'samples'`: \
        Calculate metrics for each instance, and find their average (only \
        meaningful for multilabel classification where this differs from \
        :code:`accuracy_score`).

        Defaults to :code:`None`.

    - warn_for: typing.Union[str, typing.Tuple[str]], optional:
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.
        Defaults to :code:`("sensitivity", "specificity")`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:

        - sensitivity: when there are no positive labels

        - specificity: when there are no negative labels

        - ppv: when there are no positive predictions

        - npv: when there are no negative predictions


        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - sensitivity: float:
        The sensitivity score.

    - specificity: float:
        The specificity score.

    - ppv: float:
        The positive predictive value score.

    - npv: float:
        The negative predictive value score.
    
    """

    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )

    tp_sum = MCM[:, 1, 1]
    tn_sum = MCM[:, 0, 0]
    pos_pred_sum = tp_sum + MCM[:, 0, 1]
    pos_true_sum = tp_sum + MCM[:, 1, 0]
    neg_true_sum = tn_sum + MCM[:, 0, 1]
    neg_pred_sum = tn_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])
        pos_pred_sum = np.array([pos_pred_sum.sum()])
        pos_true_sum = np.array([pos_true_sum.sum()])
        neg_true_sum = np.array([neg_true_sum.sum()])

    sensitivity = _prf_divide(
        tp_sum,
        pos_true_sum,
        "sensitivity",
        "true",
        average,
        warn_for,
        zero_division=zero_division,
    )

    specificity = _prf_divide(
        tn_sum,
        neg_true_sum,
        "specificity",
        "true",
        average,
        warn_for,
        zero_division=zero_division,
    )

    ppv = _prf_divide(
        tp_sum,
        pos_pred_sum,
        "ppv",
        "predicted",
        average,
        warn_for,
        zero_division=zero_division,
    )

    npv = _prf_divide(
        tn_sum,
        neg_pred_sum,
        "npv",
        "predicted",
        average,
        warn_for,
        zero_division=zero_division,
    )

    sensitivity_zero_division = False
    specificity_zero_division = False
    ppv_zero_division = False
    npv_zero_division = False

    if average == "weighted":
        weights_sensitivity = pos_true_sum
        weights_specificity = neg_true_sum
        weights_ppv = pos_pred_sum
        weights_npv = neg_pred_sum

        if weights_sensitivity.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # sensitivity is zero_division if there are no positive labels
            sensitivity = zero_division_value
            sensitivity_zero_division = True

        if weights_specificity.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # specificity is zero_division if there are no negative labels
            specificity = zero_division_value
            specificity_zero_division = True

        if weights_ppv.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # specificity is zero_division if there are no negative labels
            ppv = zero_division_value
            ppv_zero_division = True

        if weights_npv.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # specificity is zero_division if there are no negative labels
            npv = zero_division_value
            npv_zero_division = True

    elif average == "samples":
        weights_sensitivity = sample_weight
        weights_specificity = sample_weight
        weights_ppv = sample_weight
        weights_npv = sample_weight
    else:
        weights_sensitivity = None
        weights_specificity = None
        weights_ppv = None
        weights_npv = None

    if average is not None:
        assert average != "binary" or len(sensitivity) == 1
        if not sensitivity_zero_division:
            sensitivity = np.average(sensitivity, weights=weights_sensitivity)
        if not specificity_zero_division:
            specificity = np.average(specificity, weights=weights_specificity)
        if not ppv_zero_division:
            ppv = np.average(ppv, weights=weights_ppv)
        if not npv_zero_division:
            npv = np.average(npv, weights=weights_npv)

    return sensitivity, specificity, ppv, npv


def sensitivity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: typing.Union[np.ndarray, None] = None,
    pos_label: typing.Union[str, int] = 1,
    average: typing.Union[str, None] = "binary",
    sample_weight: typing.Union[np.ndarray, None] = None,
    zero_division: typing.Union[int, str] = "warn",
) -> float:
    """
    A function that calculates the sensitivity
    between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> sensitivity_score(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_pred=np.array([0,0,0,1,0]),
        ...     )
        0.5


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_pred: np.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:

        - :code:`'binary'`: \
        Only report results for the class specified by :code:`pos_label`. \
        This is applicable only if targets (`y_{true,pred}`) are binary.

        - :code:`'micro'`: \
        Calculate metrics globally by counting the total true positives, \
        false negatives and false positives.

        - :code:`'macro'`: \
        Calculate metrics for each label, and find their unweighted \
        mean.  This does not take label imbalance into account.

        - :code:`'weighted'`: \
        Calculate metrics for each label, and find their average weighted \
        by support (the number of true instances for each label). This \
        alters 'macro' to account for label imbalance; it can result in an \
        F-score that is not between precision and recall.

        - :code:`'samples'`: \
        Calculate metrics for each instance, and find their average (only \
        meaningful for multilabel classification where this differs from \
        :code:`accuracy_score`).

        Defaults to :code:`'binary'`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:

        - specificity: when there are no negative labels

        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - sensitivity: float:
        The sensitivity score.
    
    """

    warn_for = ("sensitivity",)
    s, _, _, _ = sensitivity_specificity_ppv_npv(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    return s


def specificity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: typing.Union[np.ndarray, None] = None,
    pos_label: typing.Union[str, int] = 1,
    average: typing.Union[str, None] = "binary",
    sample_weight: typing.Union[np.ndarray, None] = None,
    zero_division: typing.Union[int, str] = "warn",
):
    """
    A function that calculates the sensitivity
    between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> specificity_score(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_pred=np.array([0,0,0,1,0]),
        ...     )
        1.0


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_pred: np.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:

        - :code:`'binary'`: \
        Only report results for the class specified by :code:`pos_label`. \
        This is applicable only if targets (`y_{true,pred}`) are binary.

        - :code:`'micro'`: \
        Calculate metrics globally by counting the total true positives, \
        false negatives and false positives.

        - :code:`'macro'`: \
        Calculate metrics for each label, and find their unweighted \
        mean.  This does not take label imbalance into account.

        - :code:`'weighted'`: \
        Calculate metrics for each label, and find their average weighted \
        by support (the number of true instances for each label). This \
        alters 'macro' to account for label imbalance; it can result in an \
        F-score that is not between precision and recall.

        - :code:`'samples'`: \
        Calculate metrics for each instance, and find their average (only \
        meaningful for multilabel classification where this differs from \
        :code:`accuracy_score`).

        Defaults to :code:`'binary'`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:
        
        - specificity: when there are no negative labels
        
        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - sensitivity: float:
        The sensitivity score.
    
    """

    warn_for = ("specificity",)
    _, s, _, _ = sensitivity_specificity_ppv_npv(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    return s


def ppv_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: typing.Union[np.ndarray, None] = None,
    pos_label: typing.Union[str, int] = 1,
    average: typing.Union[str, None] = "binary",
    sample_weight: typing.Union[np.ndarray, None] = None,
    zero_division: typing.Union[int, str] = "warn",
):
    """
    A function that calculates the positive
    predictive value between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> ppv_score(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_pred=np.array([0,0,0,1,0]),
        ...     )
        1.0


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_pred: np.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:

        - :code:`'binary'`: \
        Only report results for the class specified by :code:`pos_label`. \
        This is applicable only if targets (`y_{true,pred}`) are binary.

        - :code:`'micro'`: \
        Calculate metrics globally by counting the total true positives, \
        false negatives and false positives.

        - :code:`'macro'`: \
        Calculate metrics for each label, and find their unweighted \
        mean.  This does not take label imbalance into account.

        - :code:`'weighted'`: \
        Calculate metrics for each label, and find their average weighted \
        by support (the number of true instances for each label). This \
        alters 'macro' to account for label imbalance; it can result in an \
        F-score that is not between precision and recall.

        - :code:`'samples'`: \
        Calculate metrics for each instance, and find their average (only \
        meaningful for multilabel classification where this differs from \
        :code:`accuracy_score`).

        Defaults to :code:`'binary'`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:
        
        - ppv: when there are no positive predictions
        
        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - ppv: float:
        The positive predictive value score.
    
    """

    warn_for = ("ppv",)
    _, _, s, _ = sensitivity_specificity_ppv_npv(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    return s


def npv_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: typing.Union[np.ndarray, None] = None,
    pos_label: typing.Union[str, int] = 1,
    average: typing.Union[str, None] = "binary",
    sample_weight: typing.Union[np.ndarray, None] = None,
    zero_division: typing.Union[int, str] = "warn",
):
    """
    A function that calculates the negative
    predictive value between two arrays. This is modelled 
    on the Scikit-Learn :code:`recall_score`, 
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> npv_score(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_pred=np.array([0,0,0,1,0]),
        ...     )
        0.75


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_pred: np.ndarray:
        The array of predicted values.
    
    - labels: typing.Union[np.ndarray, None], optional:
        The set of labels to include when 
        :code:`average != 'binary'`, and their order 
        if average is :code:`None`. Labels present in 
        the data can be excluded, for example to calculate a 
        multiclass average ignoring a majority 
        negative class, while labels not present 
        in the data will result in 0 components in a 
        macro average. For multilabel targets, labels are 
        column indices. By default, all labels in 
        :code:`y_true` and :code:`y_pred` are used in sorted order.
        Defaults to :code:`None`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and 
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - average: typing.Union[str, None], optional:
        If :code:`None`, the scores for each class 
        are returned. Otherwise, this determines the type 
        of averaging performed on the data:

        - :code:`'binary'`: \
        Only report results for the class specified by :code:`pos_label`. \
        This is applicable only if targets (`y_{true,pred}`) are binary.

        - :code:`'micro'`: \
        Calculate metrics globally by counting the total true positives, \
        false negatives and false positives.

        - :code:`'macro'`: \
        Calculate metrics for each label, and find their unweighted \
        mean.  This does not take label imbalance into account.

        - :code:`'weighted'`: \
        Calculate metrics for each label, and find their average weighted \
        by support (the number of true instances for each label). This \
        alters 'macro' to account for label imbalance; it can result in an \
        F-score that is not between precision and recall.

        - :code:`'samples'`: \
        Calculate metrics for each instance, and find their average (only \
        meaningful for multilabel classification where this differs from \
        :code:`accuracy_score`).

        Defaults to :code:`'binary'`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    - zero_division: typing.Union[int, str], optional:
        Sets the value to return when there is a zero division:
        
        - npv: when there are no negative predictions
        
        If set to :code:`"warn"`, this acts as :code:`0`, 
        but warnings are also raised.
        Defaults to :code:`"warn"`.
    
    Returns
    ---------

    - npv: float:
        The positive predictive value score.
    
    """

    warn_for = ("npv",)
    _, _, _, s = sensitivity_specificity_ppv_npv(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        warn_for=warn_for,
        zero_division=zero_division,
    )
    return s


def auc_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pos_label=None,
    sample_weight=None,
) -> float:
    """
    A function that calculates the area under
    the precision-recall curve
    between two arrays. This is modelled
    on the Scikit-Learn :code:`recall_score`,
    :code:`precision_score`, :code:`accuracy_score`,
    and :code:`f1_score`.

    Examples
    ---------

    .. code-block::

        >>> import numpy as np
        >>> auc_precision_recall_curve(
        ...     y_true=np.array([0,1,0,1,0]),
        ...     y_proba=np.array([0,0,0,1,0]),
        ...     )
        0.85


    Arguments
    ---------

    - y_true: np.ndarray:
        The array of true values.

    - y_proba: np.ndarray:
        The array of predicted score values. If :code:`y_pred`
        has shape :code:`(N,2)`, and :code:`y_true` has two unique
        values, then the probability of a positive class will
        be assumed to be :code:`y_proba[:,1]`.

    - pos_label: typing.Union[str, int], optional:
        The class to report if :code:`average='binary'` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting :code:`labels=[pos_label]` and
        :code:`average != 'binary'` will report
        scores for that label only.
        Defaults to :code:`1`.

    - sample_weight: typing.Union[np.ndarray, None], optional:
        Sample weights.
        Defualts to :code:`None`.

    Returns
    ---------

    - auc: float:
        The area under the precision-recall curve.

    """

    if len(y_proba.shape) == 2:
        if len(np.unique(y_true)) == 2:
            y_proba = y_proba[:, 1]

    y, x, _ = precision_recall_curve(
        y_true,
        y_proba,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )

    return auc(x, y)
