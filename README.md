# AetherMap Feature Tracer

A prototype for tracing visible roads and building footprints from aerial imagery. The [public demo](https://stltech9-coder.github.io/aethermap-road-candidate-explorer/) supports manual tracing and review. Automatic road detection and SAM prompts run through the local Python service.

## Run locally

Double-click `start-local-sam.bat` or the **AetherMap Local** desktop shortcut. The launcher starts the installed QGIS 3.44.1 Python runtime, waits for the local service, then opens `http://127.0.0.1:8767/`. Keep the launcher window open while using local models; errors appear there. If AetherMap is already running, the launcher opens the existing service. Port **8767** keeps this app separate from the other local service on 8766. The service binds only to `127.0.0.1`. Double-clicking `index.html` directly does not start the models; it displays a link to the running app and launcher guidance.

1. Choose **Imagery** or **Road map** from the map display control. The global road map uses [OpenFreeMap](https://openfreemap.org/). In the contiguous U.S., imagery and analysis use [USGS NAIP aerial imagery](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer). Elsewhere, choose **My georeferenced image**, select a PNG/JPEG you have rights to analyze, and enter its WGS84 west, south, east, and north bounds. The image is held in the browser for this session; reload it after refreshing the page.
2. Select **Road centerline**, then **Find Roads with SAM-Road (Local)**. Its road graph is predicted from aerial imagery. Solid purple proposals appear as the image and graph tiles finish; click each line to keep or reject it. **Quick Road Preview (Lower Detail)** runs a smaller 256-pixel segmentation model and shows dashed amber proposals. Select a kept line to edit its vertices.
3. For a roof, select **Building footprint**. **Click One Building for SAM Suggestion** outlines one clicked object; review and save it. **Find Buildings in Area (Local)** scans the selected 512 or 1024 pixel image at zoom 18 or closer using the local U.S. building Mask R-CNN checkpoint. Orange polygons are unreviewed proposals; click each to keep or reject it. You can also trace a roof boundary by hand. The area model is trained on U.S. imagery, so results elsewhere need especially careful review.
4. **Compare with Census Roads** displays red U.S. Census TIGERweb centerlines as reference proposals. These come from map data, not image detection. A blue Census reference layer can also be loaded for manual snapping. Census roads may be offset from the imagery and need review.
5. **Save Trace** stores a hand-drawn line or footprint in the reviewed layer. **Keep road** and **Save Building to Reviewed Layer** do the same for model proposals. The layer count shows saved roads and buildings. **Export All Traces (GeoJSON)** downloads saved features and unreviewed road and building proposals, each marked with source and review status. The other export buttons download reviewed features only in GeoJSON, KML, WKT, or OSM XML. OSM XML does not assign a `highway` tag to model predictions.

Reviewed features and unreviewed road and building proposals autosave in this browser and reappear after refresh, including when you return to an earlier map area. A scan of the same area and model uses its saved proposals. Browser storage is scoped to the page address and is not a cloud backup; export GeoJSON for a portable copy. The progress panel reports completed image tiles, with an indeterminate bar while a model loads. The right preview shows the captured image and suggested outlines after a trace or building scan.

The **Generic image-line diagnostics** section holds the older edge and automatic SAM mask scans. Those scans are unclassified and should not be interpreted as road detection.

## Models and limits

The road detector uses the [MIT-licensed `manish2607/road_extrcation_model_BAH` TorchScript model](https://huggingface.co/manish2607/road_extrcation_model_BAH). The local service downloads its 57 MB checkpoint into `~/.cache/aethermap/` on first use and verifies its SHA-256. It predicts a road probability mask from the selected imagery; the backend skeletonizes that mask into centerline proposals. The model card has no measured IoU or Dice score. Our Phoenix trial followed several main streets and avoided most roofs, but missed a shaded street segment and has not been benchmarked on labeled road data. Proposals remain unverified until reviewed. Performance on imagery outside its training domain has not been validated.

The local generic SAM checkpoint is `~/.cache/sam/sam_vit_h_4b8939.pth`. SAM ViT-H runs on CPU here and may take several minutes on first use. A click prompts a mask at that location; road mode extracts a line skeleton and building mode extracts a polygon boundary. It does not classify the clicked object. Tree tracing is disabled until a class-specific workflow is implemented.

The optional graph mode uses the [original SAM-Road code](https://github.com/htcr/sam_road), the [CityScale ViT-B trained checkpoint](https://huggingface.co/congrui/sam_road/blob/main/cityscale_vitb_512_e10.ckpt), and [Meta's SAM ViT-B base checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints). For this local setup, the source with submodules is at `~/.cache/aethermap/sam_road`, the CityScale checkpoint is in the application folder, and ViT-B is at `~/.cache/aethermap/sam_vit_b_01ec64.pth`. The adapter verifies both weights by SHA-256, then runs 512-pixel tiles and predicts road graph links. It uses CUDA if available, otherwise CPU. This is the original SAM-Road, **not SAM-Road++**; the SAM-Road++ authors have not published trained checkpoints in their repository. The CityScale model has not been validated on Phoenix ground truth and can still make gaps, duplicate lanes, or false connections.

For inference, the app prefers `~/.cache/aethermap/cityscale_inference_state.pt`, a 351 MB copy containing the same 229 trained tensors as the original 1.05 GB Lightning checkpoint. The copy omits optimizer and training state; tensor equality was verified before saving. The original checkpoint remains a fallback. Upstream model initialization still loads the separate 375 MB SAM ViT-B base checkpoint before the trained CityScale weights replace it.

The supplied `BUILDING_FOOTPRINT_EXTRACTION_-_INDIA.dlpk` is an ArcGIS package trained on India imagery and is **not loaded by this web app**. The local `usa_building_footprints.dlpk` is loaded by `building_model.py` for area scans. Its checkpoint is a Mask R-CNN ResNet50 model. A Phoenix imagery tile produced multiple proposals in a local trial; accuracy has not been benchmarked against labeled footprints or validated globally. [SAM-Road++](https://github.com/earth-insights/samroadplus) remains a possible future benchmark if the authors release compatible trained weights.

USGS imagery is fetched on demand and not committed to GitHub. The model weights are cached outside the repository. The public GitHub Pages site cannot run the Python endpoints; manual road and building tracing remain available there. Leaflet and MapLibre load from public CDNs. OpenFreeMap serves the global road map using OpenStreetMap data on its own infrastructure, avoiding the blocked volunteer tile servers. Census road reference uses [TIGERweb](https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_PhysicalFeatures/MapServer/layers) through the local service and is limited to the U.S.

## Attribution

- Prototype and application: Randall Morales.
- Aerial imagery: USGS / USDA NAIP via The National Map.
- Road map: OpenFreeMap, © OpenMapTiles, and OpenStreetMap contributors.
- Map interface: [Leaflet](https://leafletjs.com/).
- Road segmentation model: [manish2607 on Hugging Face](https://huggingface.co/manish2607/road_extrcation_model_BAH), MIT.

No license is granted for this repository's application code. Contact the author about reuse.
