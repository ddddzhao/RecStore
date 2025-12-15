import threading
import queue
from typing import Dict, Tuple, Optional
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class PrefetchingIterator:
    def __init__(self, dataloader, ebc_module, prefetch_count: int = 2):
        """
        dataloader: base DataLoader yielding (dense, KJT, labels)
        ebc_module: RecStoreEmbeddingBagCollection instance
        prefetch_count: queue depth
        NOTE: Must call restart() at each new epoch.
        """
        self._dataloader = dataloader
        self._prefetch_count = prefetch_count
        self._kv = ebc_module.kv_client
        self._feature_names = list(ebc_module._config_names.keys())
        self._ebc = ebc_module
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Optional[Tuple[torch.Tensor, KeyedJaggedTensor, torch.Tensor, Dict[str, object]]]]" = queue.Queue(maxsize=self._prefetch_count)
        self._stop = False
        self._exhausted = False
        self._iter = iter(self._dataloader)
        self._start_thread()

    def _producer(self):
        try:
            while not self._stop:
                batch = next(self._iter)
                dense, sparse, labels = batch
                if self._ebc._enable_fusion and self._ebc._master_config is not None:
                    handle, num_ids, issue_ts, fused_ids_cpu, inverse = self._ebc.issue_fused_prefetch(sparse, record_handle=False)
                    # Pass fused prefetch metadata alongside the batch to avoid race on shared state.
                    self._queue.put((dense, sparse, labels, {"__fused_handle": (handle, num_ids, issue_ts, fused_ids_cpu, inverse)}))
                else:
                    handles: Dict[str, int] = {}
                    import time
                    batch_issue_ts = time.time()
                    for key in sparse.keys():
                        kjt = sparse[key]
                        ids = kjt.values()
                        if ids.numel() == 0:
                            continue
                        h = self._kv.prefetch(ids)
                        handles[key] = (h, int(ids.numel()), batch_issue_ts)
                    self._queue.put((dense, sparse, labels, handles))
        except StopIteration:
            self._exhausted = True
            self._queue.put(None)
        except Exception as e:
            print(f"[Prefetcher] Error: {e}")
            self._queue.put(None)

    def _start_thread(self):
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def restart(self):
        """Restart iteration for a new epoch."""
        # Stop existing thread if alive
        self.stop(join=True)
        # Reset state
        self._queue = queue.Queue(maxsize=self._prefetch_count)
        self._stop = False
        self._exhausted = False
        self._iter = iter(self._dataloader)
        self._start_thread()

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            # If exhausted, ensure subsequent iterations also stop quickly
            if not self._exhausted:
                # Unexpected None (error path), mark exhausted to avoid hang
                self._exhausted = True
            raise StopIteration

        dense, sparse, labels, handles = item
        # Deliver fused prefetch handle for this batch right before consumption to avoid being overwritten by later prefetches.
        fused_key = "__fused_handle"
        if fused_key in handles:
            h, num_ids, issue_ts, fused_ids_cpu, inverse = handles.pop(fused_key)
            self._ebc.set_fused_prefetch_handle(h, num_ids=num_ids, issue_ts=issue_ts, record_stats=True, fused_ids_cpu=fused_ids_cpu, fused_inverse=inverse)
        return dense, sparse, labels, handles

    def stop(self, join: bool = False):
        self._stop = True
        # Drain queue swiftly
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass
        if self._thread and self._thread.is_alive() and join:
            self._thread.join(timeout=1)
