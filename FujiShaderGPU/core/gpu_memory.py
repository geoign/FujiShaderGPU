"""
FujiShaderGPU/core/gpu_memory.py
"""
import contextlib
import threading
import cupy as cp

# Thread-local GPU context management (improved)
_thread_local = threading.local()

def get_gpu_context():
    """Get a thread-local GPU memory pool (improved)."""
    if not hasattr(_thread_local, 'mempool'):
        try:
            import rmm
            try:
                from rmm.allocators.cupy import rmm_cupy_allocator
            except ImportError:
                rmm_cupy_allocator = getattr(rmm, "rmm_cupy_allocator", None)
            if rmm_cupy_allocator:
                cp.cuda.set_allocator(rmm_cupy_allocator)
        except ImportError:
            pass

        _thread_local.mempool = cp.get_default_memory_pool()
        _thread_local.pinned_mempool = cp.get_default_pinned_memory_pool()

    return _thread_local.mempool, _thread_local.pinned_mempool

@contextlib.contextmanager
def gpu_memory_pool(release: bool = False, synchronize: bool = False):
    """Context manager for the GPU memory pool (improved)."""
    mempool, pinned_mempool = get_gpu_context()
    try:
        yield
    finally:
        # Keep allocator caches hot on the tile backend. Full cleanup is still
        # available for explicit low-memory recovery paths.
        if synchronize:
            cp.cuda.Stream.null.synchronize()
        if release:
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
