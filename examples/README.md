# Reviewable AetherMap examples

These GeoJSON files are outputs from the application, not reference data or measured accuracy results.

- `reviewed_building.geojson`: one polygon selected from ten unreviewed proposals produced by a 512 × 512 pixel USGS NAIP area scan near downtown Phoenix. The local Mask R-CNN service produced the proposals; one outline was visually kept and exported. Its `review_status` is `visually_kept_unverified`.
- `reviewed_road.geojson`: one centerline drawn by hand over USGS NAIP imagery in the [public demo](https://stltech9-coder.github.io/aethermap-road-candidate-explorer/), saved, reloaded, and exported. Its `review_status` is `user_traced_unverified`.

The public demo runs manual tracing and export. Automated building and road inference require the local Python service and model checkpoints described in the project README. The model outputs have not been benchmarked against labeled ground truth.
