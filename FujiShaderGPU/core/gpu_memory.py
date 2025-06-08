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
        _thread_local.mempool = cp.get_default_memory_pool()
        _thread_local.pinned_mempool = cp.get_default_pinned_memory_pool()
        # より攻撃的なメモリプール設定
        _thread_local.mempool.set_limit(size=None)  # メモリ制限解除
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
