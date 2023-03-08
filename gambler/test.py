import numpy as np
import math
from typing import List


WINDOW_SIZE = 20

if __name__ == '__main__':
    rand = np.random.RandomState(seed=78362)

    collection_rates = [i/100 for i in range(10, 100)]

    seen = set()
    for collection_rate in collection_rates:
        target_samples = int(collection_rate * WINDOW_SIZE)
        iter_size = WINDOW_SIZE

        if target_samples in seen:
            continue
        seen.add(target_samples)

        skip = max(1.0 / collection_rate, 1)
        frac_part = skip - math.floor(skip)

        skip_indices: List[int] = []

        # static const uint16_t COLLECT_INDICES[NUM_INDICES] = {0,3,6,9,11,13,16,18,21,23,25,28,30,32,34,37,40,43,45,48};

        index = 0
        while index < iter_size:
            skip_indices.append(index)
            if (target_samples - len(skip_indices)) == (iter_size - index - 1):
                index += 1
            else:
                r = rand.uniform()
                if r > frac_part:
                    index += int(math.floor(skip))
                else:
                    index += int(math.ceil(skip))    

        skip_indices = skip_indices[:target_samples]

        str_rep = ','.join(str(idx) for idx in skip_indices)
        length = len(skip_indices)
        print(f'(uint16_t[]) {{{str_rep}}},')

tmp = ''
for i in range(4, 19):
    tmp += f'{i},'
print(tmp)
