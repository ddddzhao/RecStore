import torch
import os
from typing import Optional, Tuple, List

class RecStoreClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RecStoreClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, library_path: Optional[str] = None):
        if self._initialized:
            return
        
        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(os.path.join(script_dir, '../../../../build/lib/lib_recstore_ops.so'))
            if not os.path.exists(default_lib_path):
                 raise ImportError(
                    f"Could not find Recstore library at default path: {default_lib_path}\n"
                    "Please provide the correct path or ensure your project is built correctly."
                )
            library_path = default_lib_path
        
        torch.ops.load_library(library_path)
        self.ops = torch.ops.recstore_ops
        
        # Metadata store for named tensors
        self._tensor_meta = {}
        self._initialized = True
        print(f"RecStoreClient initialized. Loaded library from: {library_path}")

    def init_data(self, name: str, shape: Tuple[int, int], dtype: torch.dtype, init_func=None):
        """
        Initializes a named tensor in the backend store.
        """
        if name in self._tensor_meta:
            print(f"Tensor '{name}' already exists. Skipping initialization.")
            return

        print(f"Initializing tensor '{name}' with shape {shape} and dtype {dtype}.")
        self._tensor_meta[name] = {'shape': shape, 'dtype': dtype}
        
        # In a real distributed system, this would initialize data across servers.
        # Here, we simulate it by pushing initial values to the backend.
        # For simplicity, we initialize with zeros if no init_func is provided.
        if init_func:
            initial_data = init_func(shape, dtype)
        else:
            initial_data = torch.zeros(shape, dtype=dtype)
        
        # The entire tensor is initialized, so we create keys for all rows.
        all_keys = torch.arange(shape[0], dtype=torch.int64)
        self.push(name, all_keys, initial_data)

    def pull(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        """
        Pulls data for the given IDs from a named tensor.
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        
        meta = self._tensor_meta[name]
        embedding_dim = meta['shape'][1]
        return self.ops.emb_read(ids, embedding_dim)

    def push(self, name: str, ids: torch.Tensor, data: torch.Tensor):
        """
        Pushes data to the given IDs of a named tensor.
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self.ops.emb_write(ids, data)

    def update(self, name: str, ids: torch.Tensor, grads: torch.Tensor):
        """
        Pushes gradients to update the given IDs of a named tensor.
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' has not been initialized.")
        self.ops.emb_update(ids, grads)

    def get_data_meta(self, name: str) -> Tuple[torch.dtype, Tuple[int, int]]:
        """
        Returns the metadata for a named tensor.
        """
        if name not in self._tensor_meta:
            raise RuntimeError(f"Tensor '{name}' does not exist.")
        meta = self._tensor_meta[name]
        return meta['dtype'], meta['shape']

    def data_name_list(self) -> List[str]:
        """
        Returns a list of all initialized tensor names.
        """
        return list(self._tensor_meta.keys())

    def num_servers(self) -> int:
        """
        Returns the number of servers. In our mock setup, this is always 1.
        """
        return 1

def get_kv_client() -> RecStoreClient:
    """
    Factory function to get the singleton instance of the RecStoreClient.
    """
    return RecStoreClient()
