import random
from typing import List, Tuple


class FewShotSelector:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select(
        self,
        fn_prompts: List[str],
        fp_prompts: List[str],
        num_fn: int,
        num_fp: int,
    ) -> List[Tuple[str, str]]:
        shots = []

        fn_sample = fn_prompts[:num_fn]
        fp_sample = fp_prompts[:num_fp]

        for p in fn_sample:
            shots.append((p, "refuse"))

        for p in fp_sample:
            shots.append((p, "answer"))

        return shots
