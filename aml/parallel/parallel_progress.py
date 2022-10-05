import typing
from joblib import Parallel
import tqdm

from ..progress import tqdm_style

class ProgressParallel(Parallel):
    def __init__(
        self, 
        tqdm_bar:typing.Union[tqdm.tqdm, None]=None, 
        verbose:bool=True,
        desc:str='In Parallel',
        total:int=None,
        *args, 
        **kwargs,
        ):
        '''
        This is a wrapper for the joblib Parallel
        class that allows for a progress bar to be passed into
        the :code:`__init__` function so that the progress 
        can be viewed.

        Recall that using :code:`backend='threading'`
        allows for shared access to variables!
        
        
        
        Examples
        ---------
        
        .. code-block:: 
        
            >>> pbar = tqdm.tqdm(total=5)
            >>> result = ProgressParallel(
                    tqdm_bar=pbar,
                    n_jobs=10,
                    )(
                        joblib.delayed(f_parallel)(i)
                        for i in range(5)
                    )
        
        Alternatively, you do not need to pass a :code:`tqdm` bar:

        .. code-block:: 
        
            >>> result = ProgressParallel(
                    n_jobs=10,
                    total=20,
                    desc='In Parallel',
                    )(
                        joblib.delayed(f_parallel)(i)
                        for i in range(20)
                    )
        
        
        Arguments
        ---------
        
        - tqdm_bar: typing.Union[tqdm.tqdm, None]: 
            The tqdm bar that will be used in the
            progress updates.
            Every time progress is displayed, 
            :code:`tqdm_bar.update(n)` will be called,
            where :code:`n` is the number of updates made.
            If :code:`None`, then a bar is created 
            inside this class.
            Defaults to :code:`None`.
        
        - verbose: bool: 
            If :code:`tqdm_bar=None`, then this
            argument allows the user to stop the 
            progress bar from printing at all.
            Defaults to :code:`True`.
        
        - desc: str: 
            If :code:`tqdm_bar=None`, then this
            argument allows the user to add 
            a description to the progress bar.
            Defaults to :code:`'In Parallel'`.
        
        - total: str: 
            If :code:`tqdm_bar=None`, then this
            argument allows the user to add 
            a total to the progress bar, rather
            than let the bar automatically update it
            as it finds new tasks. If :code:`None`, then
            the total might update multiple times as the 
            parallel process queues jobs.
            Defaults to :code:`None`.

        
        '''
        if tqdm_bar is None:
            self.tqdm_bar = tqdm.tqdm(desc=desc, total=total, disable=not verbose, **tqdm_style)
            self.total=total
            self.build_pbar_each_time=True
        else:
            self.tqdm_bar = tqdm_bar
            self.build_pbar_each_time=False
        self.previously_completed = 0
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        
        if self.build_pbar_each_time:
            if self.total is None:
                self.tqdm_bar.total = self.n_dispatched_tasks
            self.tqdm_bar.n = self.n_completed_tasks
            self.tqdm_bar.refresh()
            if self.n_dispatched_tasks == self.n_completed_tasks:
                self.tqdm_bar.close()
        else:
            difference = self.n_completed_tasks - self.previously_completed
            self.tqdm_bar.update(difference)
            self.tqdm_bar.refresh()
            self.previously_completed += difference