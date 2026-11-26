# AetherMap Road Candidate Explorer

A browser-based image-processing prototype that turns visible linear features in aerial imagery into **unverified** geospatial line candidates. Built as a small proof of work from Randall Morales's AetherMap prototype.

**[Open the live demo](https://stltech9-coder.github.io/aethermap-road-candidate-explorer/)**

## Try it

1. Open `index.html` through a static web server or the GitHub Pages demo.
2. The map displays USGS NAIP aerial imagery in the contiguous United States. Pan or zoom to an area, then click the location to analyze. The default example is downtown Phoenix, Arizona.
3. Select **Find Line Candidates** to request an analysis image from the same USGS service and preview its extracted edges.
4. Select **Vectorize Candidates**, review the green overlay, then **Add to Layer**. In the local version, **SAM-assisted Trace** uses the installed Segment Anything checkpoint and draws candidate centerlines as each result is returned.
5. Export GeoJSON, KML, WKT, or OSM XML for further review. Exported features are marked `review_status: unverified`; OSM XML omits a `highway` classification.

## What the prototype demonstrates

- A live aerial map and georeferenced analysis image from the [USGS NAIP ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer).
- Browser-side grayscale conversion, Gaussian blur, Sobel edge extraction, morphology, skeleton tracing, simplification, and vector export.
- A map overlay and basic feature/length statistics. OpenStreetMap is the fallback map while USGS imagery loads or outside coverage.

The public GitHub Pages version is **classical image processing**, not a trained AI model. The optional local backend runs SAM ViT-H to create generic segmentation masks. It applies the original QGIS scanner's elongated-mask heuristic and traces mask skeletons into lines. SAM does **not** classify roads: rooftops, shadows, and other elongated objects may be included. Neither mode has measured road precision/recall; every line needs human review before use.

## Verification and limits

The original prototype sorted connected pixels by X/Y position and reported approximately 387 km of line length in a 0.26 km² area. This version follows adjacent skeleton pixels in path order. A local browser run on the default USGS sample produced 59 candidates totaling about 1.26 km in a 0.065 km² area. That is a geometry sanity check, **not an accuracy benchmark**.

USGS imagery is fetched on demand and is not stored in this repository. Coverage and response time depend on the USGS service and internet access. The application currently supports analysis in the contiguous United States. The [USGS describes NAIP imagery as public domain](https://www.usgs.gov/programs/national-geospatial-program/supporting-themes), but users should check the provenance of imagery they substitute. Leaflet is loaded from a public CDN.

## Run locally

From this folder, use any static HTTP server, for example:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`. No API key, backend, training dataset, or model weights are required.

### Local SAM-assisted tracing

Double-click `start-local-sam.bat` on this Windows machine, then use the browser tab at `http://127.0.0.1:8766/`. This launcher uses the installed QGIS 3.44.1 Python runtime and the existing `~/.cache/sam/sam_vit_h_4b8939.pth` checkpoint. The backend binds only to `127.0.0.1`. It loads the 2.5 GB SAM ViT-H checkpoint on **CPU**, because the installed GTX 1650 has 4 GB of VRAM. First load and inference can take several minutes. No checkpoint is committed to GitHub.

Click **SAM-assisted Trace (Local)**. The app requests a fresh USGS NAIP image of the selected area, sends it to the local backend, and displays each returned line candidate on the map. Select **Add to Layer** to include them in an export. The source field records `usgs_naip_sam_mask_heuristic`, and exported features remain `unverified`. This is a local model-assisted workflow; the GitHub Pages demo cannot run the Python model.

## Source and attribution

- Prototype and browser application: Randall Morales.
- Analysis imagery: USGS National Map NAIP imagery service.
- Background map: © OpenStreetMap contributors.
- Map interface: [Leaflet](https://leafletjs.com/).

No license is granted for this repository's application code. Please contact the author about reuse.
