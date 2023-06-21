import torch
import numpy as np


class UniformMasking:
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_ratio = mask_prob

    def __call__(self, instance):

        # 1. get binary-encoded masked indexes and masked positions
        uniform_vec = np.random.rand(len(instance))
        uniform_vec = uniform_vec <= self.mask_ratio
        masked_vec = uniform_vec.astype(int)

        # 2. get real and random binary-encoded masked indexes
        uniform_vec2 = np.random.rand(len(instance))

        random_vec = np.zeros(len(instance))
        same_vec = np.zeros(len(instance))

        random_vec[(masked_vec == 1) & (uniform_vec2 <= 0.1)] = 1
        same_vec[(masked_vec == 1) & (uniform_vec2 >= 0.9)] = 1
        real_vec = abs(masked_vec - random_vec - same_vec)
        random_vec = np.array(random_vec).astype(bool)
        real_vec = np.array(real_vec).astype(bool)

        # 3. masking with all zeros.
        instance[real_vec, :] = [0, 0, 0, 0]

        # 4. masking with random one-hot encode
        instance[random_vec, :] = np.eye(4)[np.random.choice(4, 1)]

        return instance, masked_vec


class BatchMaking:
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_ratio = mask_prob

    def __call__(self, batch):
        mask = UniformMasking(self.mask_ratio)
        new_batch = np.zeros_like(batch)
        new_vec = np.zeros([batch.shape[0], batch.shape[1]])

        for index, instance in enumerate(batch):
            mask_instance, mask_vec = mask(instance)
            new_batch[index] = mask_instance
            new_vec[index] = mask_vec

        return new_batch, new_vec


def pretrain_loss(loss, batch_pretrain, batch_mask_pred, batch_vec):
    batch_vec_re = batch_vec.repeat(4, axis=1).reshape(batch_pretrain.shape)
    batch_vec_re = torch.tensor(batch_vec_re)

    batch_pretrain_pick = batch_pretrain[batch_vec_re == 1]
    batch_mask_pred_pick = batch_mask_pred[batch_vec_re == 1]

    return loss(batch_mask_pred_pick, batch_pretrain_pick)
