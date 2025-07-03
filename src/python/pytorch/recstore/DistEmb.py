import torch
from .DistTensor import DistTensor

class _DistEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, dist_tensor, dummy_param):
        ctx.save_for_backward(ids)
        ctx.dist_tensor = dist_tensor
        embs = dist_tensor[ids]
        return embs

    @staticmethod
    def backward(ctx, grad_output):
        ids, = ctx.saved_tensors
        dist_tensor = ctx.dist_tensor
        dist_tensor[ids] = grad_output.contiguous()
        return None, None, None

class DistEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, name, init_func=None):
        super(DistEmbedding, self).__init__()
        if not name:
            raise ValueError("DistEmb requires a unique 'name'.")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self._tensor = DistTensor(
            shape=(num_embeddings, embedding_dim),
            dtype=torch.float32,
            name=name,
            init_func=init_func
        )
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def forward(self, ids):
        return _DistEmbFunction.apply(ids, self._tensor, self.dummy_param)

    @property
    def weight(self):
        return self._tensor

    def __repr__(self):
        return (f"DistEmb(name='{self.weight.name}', num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim})")
