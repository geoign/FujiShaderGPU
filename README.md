# FujiShaderGPU 富士シェーダーGPU🌋
- Lightning fast terrain shader for a big Cloud Optimized GeoTIFF
- Cloud Optimized GeoTIFFの為の電光石火DEM地形可視化シェーダー

GPUを使って処理することでCPUの数百倍の速度で計算できます。

## Install インストール
- Requires CUDA environment (nVidia GPU)
- CUDA実行環境が必要です。(nVidia社のGPU + WindowsであればCUDA tookit)
- Currently, Linux is the only platform fully supported.
- 今のところ、Linux環境以外は実装やバグが怪しいです。

```pip install git+https://github.com/geoign/FujiShaderGPU.git```

## Usage 使い方
- Only pseudo- Ridge Valley Index (RVI) is available at the moment.
- 現在のバージョンでは、疑似的な尾根谷度の計算のみサポートしています。

```bash
fujishadergpu infile.tif outfile.tif --algo [See below for the supported algorithms]
```

⭐[Try at Google Colab.](https://colab.research.google.com/drive/1IbIGtaoKM9e1OsdxdnzNN7KeO1W_gRwZ?usp=sharing)⭐ <br>
↑ Google Colabで試すことができます。Google Driveから読み込み書き出します。<br>
Colab Notebook Last Updated at: 2025/06/09.

## Limitations 注意事項
- FujiShaderGPU is designed for the Cartesian DEMs.<br>It can process LatLon DEMs too but the result is not accurate.
- 富士シェーダーは直交座標系のDEMの処理を想定しています。<br>緯度経度座標系のDEMも処理できますが、結果は正確ではありません。

## Benchmark ベンチマーク
### nVidia A100 GPU (Google Colab)
- 70,000 x 70,000 pixels: 7 min. (Processing) + 7 min. (COG packaging)
- 220,000 x 240,000 pixels: 60 min. (Processing) + ? min. (COG packaging)

### nVidia RTX4070 Laptop (Windows)
- 60,000 x 30,000 pixels: 5 min. (Processing) + 10 min. (COG packaging)

## Maintainer 作成者
池上郁彦 (Fumihiko IKEGAMI) / Ikegami GeoResearch

## Acknowledgements 謝辞
ChatGPT o3 & Claude Sonnet 4
