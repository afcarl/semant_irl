import numpy as np

class BatchSampler(object):
    def __init__(self, data):
        self.data = np.array(data, dtype=np.object_)
        self.num_data = len(data)

    def with_replacement(self, batch_size=5):
        while True:
            batch_idx = np.random.randint(0, self.num_data, size=batch_size)
            yield self.data[batch_idx]


