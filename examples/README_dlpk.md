# Running an Esri `.dlpk` in Braein.ai — reproducible walkthrough

This is exactly how the car-detection result was produced, as a script you
can run rather than screenshots to copy.

**Result:** 107 cars detected on live Esri World Imagery over Woodfield Mall,
Schaumburg IL, scores 0.506–0.952, written as an annotated PNG and a WGS84
GeoJSON.

---

## 1. Run it

```bash
cd Breainai/BreainAI_UNet_Original_Polygon_Annotation_Fix

python -X utf8 examples/dlpk_detect_demo.py \
    --dlpk "../../CarDetection_USA.dlpk" \
    --lat 42.0450 --lon -88.0370 --zoom 19 --tiles 4 \
    --out out/cars_woodfield
```

Needs `torch`, `torchvision`, `pillow`, `numpy` — all already installed here.
No ArcGIS, no `arcgis.learn`, no `fastai`.

Against your own imagery instead of fetched tiles:

```bash
python -X utf8 examples/dlpk_detect_demo.py \
    --dlpk CarDetection_USA.dlpk --image my_ortho.tif --out out/mine
```

Against an IGA image service, by pointing `--service` at it (note the
`{z}/{y}/{x}` order — ArcGIS serves `row/col`, not `x/y`):

```bash
--service "https://<server>/arcgis/rest/services/<svc>/MapServer/tile/{z}/{y}/{x}"
```

---

## 2. What it prints

```
 1. READ THE PACKAGE   (dlpk.describe)
   framework              arcgis.learn.models._inferencing
   model_configuration    _maskrcnn_inferencing
   esri_model_type        InstanceDetection
   model_type             mask_rcnn
   inference_task         DetectObjectsUsingDeepLearning
   chip_metadata_format   RCNN_Masks
   num_classes            1
   tile_size              400
   inference_function     ArcGISInstanceDetector.py

 2. IMPORT IT        (dlpk.import_dlpk)
   NOT EXTRACTED ArcGISInstanceDetector.py   <- third-party executable Python

 3. LOAD THE WEIGHTS
   STRICT load OK       307 tensors, zero adaptation
   inference size       800 px  (chip 400 x 2.00 upscale)

 5. DETECT
   chips processed      9
   raw detections       164
   after NMS            107
   score range          0.506 - 0.952
```

Steps 1 and 2 are **Braein.ai's own code** — `dlpk.describe()` and
`dlpk.import_dlpk()`, the same functions the web UI calls. This is not a
parallel implementation.

---

## 3. The two settings that decide whether it works at all

Get either wrong and you get **zero** detections, not poor ones — which
reads like a broken model when it is a mismatched input.

### Scale — the important one

`CarDetection_USA` was trained on imagery finer than Esri World Imagery at
z=19. The model has a fixed idea of how many pixels a car occupies.

| Setting | Result on one chip |
|---|---|
| `min_size=400` (chip at native size) | 6 above 0.5 |
| **`min_size=800` (2× upscale)** | **30 above 0.5, peak 0.952** |
| z=20 imagery instead | **0** |

Hence `--upscale 2.0` is the default. If you get nothing, try
`--upscale 1.0` and `--upscale 3.0` **before** concluding the model is wrong.

### Normalisation

torchvision's detection transform applies ImageNet mean/std internally. Feed
it raw 0–1 pixels. Normalising yourself first double-normalises and
detections collapse to near zero.

---

## 4. Why the weights load with no adaptation

A real Esri **inferencing** package stores a bare torchvision `state_dict`:

```
backbone.body.conv1.weight          <- real Esri, 307 tensors
```

so `maskrcnn_resnet50_fpn(num_classes=2)` + `load_state_dict(strict=True)`
accepts all 307 with no key remapping.

> An earlier note in `CLAUDE.md` said the barrier was a `model.` prefix.
> **That was wrong** — it came from a package built to *imitate* Esri's.
> An arcgis.learn *training* checkpoint may nest under `model.`, which
> `app.load_checkpoint()` handles with a strict-retry; an *inferencing*
> package does not.

**What this means:** an Esri-trained and a Braein-trained Mask R-CNN are the
same torchvision network with different packaging. Interop is an adapter,
not a rewrite.

---

## 5. Outputs

- `out/cars_woodfield.png` — annotated image
- `out/cars_woodfield.geojson` — WGS84 polygons with a `score` per feature,
  loadable straight into ArcGIS or QGIS

Georeferencing checked: boxes land at −88.038…−88.036, 42.044…42.046, and
the first box measures ≈4.5 m — car-sized. (Boxes are axis-aligned, so
angled parking reads slightly taller than a car's 1.8 m width.)

---

## 6. Limits — state these before anyone over-reads the result

- **`arcgis` and `fastai` are not installed**, so Esri's own inference
  function cannot run here. This is torchvision reproducing what that
  function would do, not Esri's code executing.
- **Esri's bundled Python is deliberately not extracted.** A `.dlpk`
  contains executable Python that ArcGIS would run; Braein treats a model
  package as data, not code.
- **Recall is not measured.** The image clearly contains cars with no box.
  No ground truth was labelled, so there is no recall or precision figure —
  only that detections are correctly placed and correctly sized.
- **Not run against ArcGIS Enterprise.** Nothing here creates a portal item.
- **Export the other way is weaker**: verified for U-Net, `structurally_valid`
  only for detection, because `vectorize()` refuses to guess NMS thresholds
  and mask tracing rather than emit wrong geometry.
