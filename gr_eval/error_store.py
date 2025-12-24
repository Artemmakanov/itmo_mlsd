from collections import defaultdict
from typing import List, Dict


class ErrorStore:
    """
    Stores FP/FN statistics across iterations.
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.fn_by_iter: Dict[int, List[str]] = defaultdict(list)
        self.fp_by_iter: Dict[int, List[str]] = defaultdict(list)

    def add(self, iteration: int, prompt: str, label: int, pred: int):
        if label == 1 and pred == 0:
            self.fn_by_iter[iteration].append(prompt)
        elif label == 0 and pred == 1:
            self.fp_by_iter[iteration].append(prompt)

    def get_prev_iteration(self, current_iteration: int):
        """
        Collect errors from previous iterations only.
        """
        
        if current_iteration > 0:
            fn = self.fn_by_iter[current_iteration - 1]
            fp = self.fp_by_iter[current_iteration - 1]
        else:
            fn, fp = [], []
        return fn, fp
