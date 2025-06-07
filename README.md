# FujiShaderGPU
- Lightning fast terrain shader for a big Cloud Optimized GeoTIFF
- Cloud Optimized GeoTIFFの為の電光石火DEM地形可視化シェーダー

GPUを使って処理することで数分～10分以内に計算が終わります。

## Install インストール
- Requires CUDA environment (nVidia GPU)
- CUDA実行環境が必要です。(nVidia社のGPU)
```pip install git+https://github.com/geoign/FujiShaderGPU.git```

## Usage 使い方
- Only pseudo- Ridge Valley Index (RVI) is available at the moment.
- 現在のバージョンでは、尾根谷度の計算のみサポートしています。

```bash
FujiShaderGPU infile.tif outfile.tif
```

![Try at Google Colab.](https://colab.research.google.com/drive/1IbIGtaoKM9e1OsdxdnzNN7KeO1W_gRwZ?usp=sharing) ←Google Colabで試すことができます。Google Driveから読み込み書き出します。 

## Limitations 注意事項
FujiShaderGPU is designed for the Cartesian DEMs. It can process LatLon DEMs too but the result is not accurate. 

## Benchmark ベンチマーク
### nVidia A100 GPU (Google Colab)
- 70,000 x 70,000 pixels: 7 min. (Processing) + 7 min. (COG packaging)
- 20,000 x 240,000 pixels: 60 min. (Processing) + ? min. (COG packaging)

### nVidia RTX4070 Laptop
- 60,000 x 30,000 pixels: 5 min. (Processing) + 10 min. (COG packaging)

## Maintainer 作成者
池上郁彦 (Fumihiko IKEGAMI) / Ikegami GeoResearch

## Acknowledgements 謝辞
ChatGPT o3 & Claude Sonnet 4
