# AetherMap Road Candidate Explorer

A browser-based image-processing prototype that turns visible linear features in aerial imagery into **unverified** geospatial line candidates. Built as a small proof of work from Randall Morales's AetherMap prototype.

## Try it

1. Open `index.html` through a static web server or the GitHub Pages demo.
2. Click a location in the contiguous United States. The default example is downtown Phoenix, Arizona.
3. Select **Find Line Candidates** to request a USGS NAIP image and preview its extracted edges.
4. Select **Vectorize Candidates**, review the green overlay, then **Add to Layer**.
5. Export GeoJSON, KML, WKT, or OSM XML for further review. Exported features are marked `review_status: unverified`; OSM XML omits a `highway` classification.

## What the prototype demonstrates

- A georeferenced image request from the [USGS NAIP ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer).
- Browser-side grayscale conversion, Gaussian blur, Sobel edge extraction, morphology, skeleton tracing, simplification, and vector export.
- A map overlay and basic feature/length statistics. OpenStreetMap supplies the background map with attribution.

This is **classical image processing**, not a trained AI model. It does not provide PyTorch inference, visual search, SKU recognition, validated road classification, or measured precision/recall. Building and shadow edges can become candidates; every line needs human review before use.

## Verification and limits

The original prototype sorted connected pixels by X/Y position and reported approximately 387 km of line length in a 0.26 km² area. This version follows adjacent skeleton pixels in path order. A local browser run on the default USGS sample produced 59 candidates totaling about 1.26 km in a 0.065 km² area. That is a geometry sanity check, **not an accuracy benchmark**.

USGS imagery is fetched on demand and is not stored in this repository. Coverage and response time depend on the USGS service and internet access. The application currently supports analysis in the contiguous United States. The [USGS describes NAIP imagery as public domain](https://www.usgs.gov/programs/national-geospatial-program/supporting-themes), but users should check the provenance of imagery they substitute. Leaflet is loaded from a public CDN.

## Run locally

From this folder, use any static HTTP server, for example:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`. No API key, backend, training dataset, or model weights are required.

## Source and attribution

- Prototype and browser application: Randall Morales.
- Analysis imagery: USGS National Map NAIP imagery service.
- Background map: © OpenStreetMap contributors.
- Map interface: [Leaflet](https://leafletjs.com/).

No license is granted for this repository's application code. Please contact the author about reuse.
