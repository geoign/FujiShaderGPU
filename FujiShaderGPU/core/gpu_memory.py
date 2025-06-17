"""
FujiShaderGPU/core/gpu_memory.py
"""
import contextlib, threading
import cupy as cp

# スレッドローカルなGPUコンテキスト管理（改良版）
_thread_local = threading.local()

def get_gpu_context():
    """スレッドローカルなGPUメモリプールを取得（改良版）"""
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

        # 無制限ではなく VRAM の 90 % を上限にしてリークを抑制
        try:
            import GPUtil
            gpu = GPUtil.getGPUs()[0]
            _thread_local.mempool.set_limit(size=int(gpu.memoryTotal * 0.9 * 1024**2))
        except Exception:
            pass
    return _thread_local.mempool, _thread_local.pinned_mempool

@contextlib.contextmanager
def gpu_memory_pool():
    """GPUメモリプールを管理するコンテキストマネージャー（改良版）"""
    mempool, pinned_mempool = get_gpu_context()
    try:
        yield
    finally:
        # より効率的なメモリクリーンアップ
        cp.cuda.Stream.null.synchronize()  # GPU処理完了を確実に待機
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
