# FujiShaderGPU 富士シェーダーGPU🌋
- Lightning fast terrain shader for a big Cloud Optimized GeoTIFF
- Cloud Optimized GeoTIFFの為の電光石火DEM地形可視化シェーダー

GPUを使って処理することでCPUの数百倍の速度で計算できます。

## Install インストール
- Requires CUDA environment (nVidia GPU). <br>See below if you are non-Linux user.
- CUDA実行環境が必要です(nVidia社のGPU)。<br>非Linuxユーザーは後半のセクションを参照してください。

```pip install git+https://github.com/geoign/FujiShaderGPU.git```

## Usage 使い方
- More than >10 algorithms are available.
- 現在のバージョンでは、10個以上のアルゴリズムをサポートしています。

```bash
fujishadergpu infile.tif outfile.tif --algo [See below for the supported algorithms]
```

⭐[Try at Google Colab.](https://colab.research.google.com/drive/1IbIGtaoKM9e1OsdxdnzNN7KeO1W_gRwZ?usp=sharing)⭐ <br>
↑ Google Colabで試すことができます。Google Driveから読み込み書き出します。<br>
Colab Notebook Last Updated at: 2025/06/09.

## Algorithms アルゴリズム
### Ridge Valley Index (RV) 尾根谷度
```bash
fujishader DEM.tif RVI.tif --algo rvi
```
| Optional Parameters | Example | Description |
| :-------- | :------- | :-------- |
| --radius | TBD | TBD    |



## Limitations 注意事項
- FujiShaderGPU is designed for the Cartesian DEMs.<br>It can process LatLon DEMs too but the result is not accurate.
- 富士シェーダーは直交座標系のDEMの処理を想定しています。<br>緯度経度座標系のDEMも処理できますが、結果は正確ではありません。

## Benchmark ベンチマーク
### nVidia A100 GPU (Google Colab)
- 70,000 x 70,000 pixels: 5~10 min. (Processing) + 10 min. (COG packaging)
- 220,000 x 240,000 pixels: 60 min. (Processing) + ? min. (COG packaging)

### nVidia RTX4070 Laptop (Windows)
- 60,000 x 30,000 pixels: 5 min. (Processing) + 10 min. (COG packaging)

## For Windows users:
- FujiShaderGPU has two pipelines: "dask-cuda" and "tile". There is no compatibility. <br>The "dask-cuda" only work for Linux and WSL (Windows Subsystem for Linux).<br>The "tile" pipeline is an original ad-hoc routine and is not well maintained at the moment.
- 富士シェーダーGPUは２つのパイプラインを持っています: "dask-cuda"と"tile"です。互換性はありません。<br>"dask-cuda"はLinuxとWSL (Windows Subsystem for Linux)のみサポートしています。<br>"tile"パイプラインは自作のアドホックルーチンであり、現状ではあまりメンテナンスされていません。

## For Mac users:
- Mac is not supported because there is no nVidia GPU on Mac.

## ChangeLog
- 2025/06/09 0.1.4 Fixed problems. dask-based algorithms were implemented.<br>Original "tile" based algorithms were moved to backup.
- 2025/06/08 0.1.4 Broke the repository by an accident T_T.
- 2025/06/07 0.1.0 Initial upload. Only RVI support.

## Maintainer 作成者
池上郁彦 (Fumihiko IKEGAMI) / Ikegami GeoResearch

## Acknowledgements 謝辞
ChatGPT o3 & Claude Sonnet 4
