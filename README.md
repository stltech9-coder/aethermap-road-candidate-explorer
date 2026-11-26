# AetherMap Feature Tracer

A prototype for tracing visible roads and building footprints from aerial imagery. The [public demo](https://stltech9-coder.github.io/aethermap-road-candidate-explorer/) supports manual tracing and review. Automatic road detection and SAM prompts run through the local Python service.

## Run locally

Double-click `start-local-sam.bat` and open `http://127.0.0.1:8767/`. Port **8767** keeps this app separate from the other local service on 8766. The launcher uses the installed QGIS 3.44.1 Python runtime. The service binds only to `127.0.0.1`.

1. Pan and zoom to a contiguous U.S. location. The map and analysis image use [USGS NAIP aerial imagery](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer).
2. Select **Road centerline**, then **Find Roads in Imagery (Local)**. Purple lines are road-specific model predictions from the aerial image. Click each line to keep or reject it. Select a kept line to edit its vertices.
3. For a roof, select **Building footprint**. Trace its boundary on the map, or click it with the local SAM prompt tool. The SAM result needs visual review because SAM itself has no road/building class label.
4. The separate red **Show Mapped Roads Over Imagery** action displays OpenStreetMap centerlines as reference proposals. These come from map data, not image detection. A blue OSM reference layer can also be loaded for manual snapping.
5. Export reviewed features to GeoJSON, KML, WKT, or OSM XML. Exports retain source and review status. OSM XML does not assign a `highway` tag to model predictions.

The **Generic image-line diagnostics** section holds the older edge and automatic SAM mask scans. Those scans are unclassified and should not be interpreted as road detection.

## Models and limits

The road detector uses the [MIT-licensed `manish2607/road_extrcation_model_BAH` TorchScript model](https://huggingface.co/manish2607/road_extrcation_model_BAH). The local service downloads its 57 MB checkpoint into `~/.cache/aethermap/` on first use and verifies its SHA-256. It predicts a road probability mask from the selected NAIP image; the backend skeletonizes that mask into centerline proposals. The model card has no measured IoU or Dice score. Our Phoenix trial followed several main streets and avoided most roofs, but missed a shaded street segment and has not been benchmarked on labeled road data. Proposals remain unverified until reviewed.

The local generic SAM checkpoint is `~/.cache/sam/sam_vit_h_4b8939.pth`. SAM ViT-H runs on CPU here and may take several minutes on first use. A click prompts a mask at that location; road mode extracts a line skeleton and building mode extracts a polygon boundary. It does not classify the clicked object. Tree tracing is disabled until a class-specific workflow is implemented.

The supplied `BUILDING_FOOTPRINT_EXTRACTION_-_INDIA.dlpk` is an ArcGIS package trained on India imagery and is **not loaded by this web app**. A separate local `usa_building_footprints.dlpk` was found, but automatic building extraction still needs an adapter and validation. [SAM-Road++](https://github.com/earth-insights/samroadplus) is a possible future road-graph benchmark for improving connections and junctions; it is **not used by this version**.

USGS imagery is fetched on demand and not committed to GitHub. The model weights are cached outside the repository. The public GitHub Pages site cannot run the Python endpoints; manual road and building tracing remain available there. Leaflet loads from a public CDN, with OpenStreetMap as the fallback map while NAIP loads or outside coverage.

## Attribution

- Prototype and application: Randall Morales.
- Aerial imagery: USGS / USDA NAIP via The National Map.
- Map and road reference: © OpenStreetMap contributors.
- Map interface: [Leaflet](https://leafletjs.com/).
- Road segmentation model: [manish2607 on Hugging Face](https://huggingface.co/manish2607/road_extrcation_model_BAH), MIT.

No license is granted for this repository's application code. Contact the author about reuse.
