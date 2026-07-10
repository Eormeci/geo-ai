# GeoAI

A geospatial AI platform with a Streamlit-based interactive web app and 90+ Jupyter notebooks covering remote sensing, computer vision, and geospatial machine learning workflows.

## Features

### Streamlit Web App (`app.py`)

An all-in-one geospatial demo interface with:

- **Multi-format support** — GeoTIFF, COG, Shapefile, GeoJSON, GeoPackage, KML/KMZ, GPX, Sentinel-2, Landsat, NAIP, Planet (via STAC)
- **Location-based imagery** — Search any location and download satellite imagery
- **Interactive maps** — Folium-based with fullscreen mode, HTML export, and download
- **ArcGIS integration** — Search public feature services, auto-detect point layers, query features
- **LLM-powered map agent** — Ask questions about mapped data in natural language; the agent uses function-calling tools to query, filter, and analyze the dataset
- **Automated field inference** — Auto-detects crime type, description, city, district, and date fields from arbitrary ArcGIS schemas

### Notebooks (90+)

Organized by topic:

| Range | Topic |
|-------|-------|
| 01–05 | Data download (Sentinel-2, NAIP, Planetary Computer, metadata) |
| 06–09 | Vector operations, image chips, tiling |
| 10–13 | Training data creation, augmentation (FlipNSlide) |
| 14–18 | Building footprints (USA, Africa, China), regularization |
| 19–23 | Object detection (cars, ships, solar panels, parking) |
| 24–29 | Visualization, model training (detection, segmentation, landcover) |
| 30–39 | Loss functions, building footprints training, water/wetland detection |
| 41–46 | Globe projection, water detection, SAM, Grounded SAM |
| 51–59 | Instance segmentation, change detection, JPEG2000, DINOv3 |
| 60–64 | AI agents, STAC agents, image recognition, AutoModel |
| 66–79 | Self-supervised learning, timm training, ONNX, smooth inference |
| 81–90 | TorchGeo embeddings, super resolution, image captioning, GeoDeep |

## Tech Stack

- **Web App**: Streamlit, Folium, ArcGIS REST API
- **LLM Agent**: OpenAI-compatible API with function calling
- **Geospatial**: Rasterio, GeoPandas, Shapely, STAC
- **ML**: TensorFlow, PyTorch, KerasNLP, timm, DINOv3, SAM, Grounded SAM
- **Satellite Data**: Microsoft Planetary Computer, Sentinel-2, Landsat, NAIP

## Getting Started

```bash
pip install streamlit folium rasterio geopandas requests numpy

streamlit run app.py
```

For LLM agent features, configure the API endpoint in `app.py`:

```python
LLM_API_URL = "http://your-llm-endpoint/v1/chat/completions"
LLM_MODEL = "your-model-name"
```

## Notebooks

All notebooks are in `notebooks/geoai_notebooks/`. Browse by number — each is self-contained with inline explanations.
