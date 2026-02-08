"""Bridge tile backend algorithms to shared Dask algorithm implementations."""
from __future__ import annotations

import cupy as cp


class DaskSharedTileAdapter:
    """Run a DaskAlgorithm implementation on a single tile array."""

    dask_algorithm_cls = None

    def __init__(self):
        if self.dask_algorithm_cls is None:
            raise ValueError("dask_algorithm_cls must be set in subclass")
        self._algo = self.dask_algorithm_cls()

    def get_default_params(self):
        return self._algo.get_default_params()

    def process(self, dem_gpu: cp.ndarray, **params):
        try:
            import dask.array as da
        except Exception as exc:
            raise RuntimeError(
                "Dask is required for this algorithm on tile backend. "
                "Install with: pip install dask[array]"
            ) from exc

        gpu_da = da.from_array(dem_gpu, chunks=dem_gpu.shape, asarray=False)
        result_da = self._algo.process(gpu_da, **params)
        result = result_da.compute()
        if isinstance(result, cp.ndarray):
            return result
        return cp.asarray(result)
