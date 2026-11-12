"""
test_dlpk.py - round-trip test of .dlpk export and import
==========================================================
Evidence for the "can a Braein.ai model enter the ArcGIS estate"
question. Train-free: it builds a small checkpoint in memory, exports it,
re-opens the zip, asserts the .emd fields, re-imports it and asserts the
model loads - then does the same for a foreign, ArcGIS-shaped package.

    python test_dlpk.py .

What this test CANNOT prove: that an ArcGIS Image Server will execute the
package. There is no ArcGIS Enterprise in this environment. Everything here
is structural - the zip, the EMD, the class table, the import path and the
route gating. See dlpk.py's docstring for the (a)/(b) distinction.

Uses a throwaway users.json in a temp directory and a scratch directory
under models/, so it never touches the deployment's accounts or models.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile

APPDIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__))
os.chdir(APPDIR)
sys.path.insert(0, APPDIR)

# Deterministic test environment, set BEFORE app import (same as test_auth).
tmp_users = os.path.join(tempfile.mkdtemp(), "users.json")
os.environ["USERS_FILE"] = tmp_users
os.environ["SESSION_SECRET"] = "test-secret-not-for-production"
os.environ["API_KEY"] = "test-api-key-123"
os.environ["PBKDF2_ITERATIONS"] = "1000"
os.environ["MAX_LOGIN_ATTEMPTS"] = "50"
os.environ["AUTH_BACKEND"] = "local"

import torch

import auth
import dlpk
import deployment_config as depcfg
import app as appmod
from fastapi.testclient import TestClient

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (("   " + str(detail)) if detail and not cond else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Scratch space INSIDE models/ so the path-confinement rules apply normally.
WORK = os.path.join(APPDIR, "models", "_test_dlpk")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK, exist_ok=True)

CLASS_NAMES = ["background", "building"]


# =========================================================================
print("\n--- 1. the project-type cascade (CLAUDE.md section 2) ---")
# =========================================================================
expected = {
    "unet":        ("ImageClassification",  "ClassifyPixelsUsingDeepLearning",  "Classified_Tiles"),
    "rcnn":        ("ObjectDetection",      "DetectObjectsUsingDeepLearning",   "PASCAL_VOC_rectangles"),
    "classic_cnn": ("ObjectClassification", "ClassifyObjectsUsingDeepLearning", "Labeled_Tiles"),
    "mask_rcnn":   ("ObjectDetection",      "DetectObjectsUsingDeepLearning",   "RCNN_Masks"),
}
for model_type, (etype, task, chips) in expected.items():
    got = dlpk.task_for(model_type)
    check("cascade %s -> %s / %s / %s" % (model_type, etype, task, chips),
          got[0] == etype and got[1] == task and got[2] == chips, got)

emd_det = dlpk.build_emd(
    {"model_type": "rcnn", "class_names": CLASS_NAMES, "tile_size": 256},
    model_file="x.pth",
)
check("detection EMD drops the background class",
      [c["Name"] for c in emd_det["Classes"]] == ["building"],
      emd_det["Classes"])
emd_seg = dlpk.build_emd(
    {"model_type": "unet", "class_names": CLASS_NAMES, "tile_size": 256},
    model_file="x.pth",
)
check("pixel-classification EMD keeps the background class",
      [c["Name"] for c in emd_seg["Classes"]] == CLASS_NAMES,
      emd_seg["Classes"])


# =========================================================================
print("\n--- 2. build a small checkpoint (no training) ---")
# =========================================================================
model = appmod.build_model(
    mode="binary",
    num_classes=2,
    backbone="unet",
    pretrained=False,          # air-gap guard: nothing is downloaded
    model_type="unet",
    tile_size=64,
)
architecture = getattr(model, "geoai_architecture", type(model).__name__)
CKPT = os.path.join(WORK, "bsdi_buildings.pth")
torch.save({
    # Exactly the fields the torch.save sites in app.py write.
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": {},
    "scheduler_state_dict": None,
    "epoch": 7,
    "best_iou": 0.8123,
    "model_type": "unet",
    "architecture": architecture,
    "backbone": "unet",
    "num_classes": 2,
    "class_names": CLASS_NAMES,
    "tile_size": 64,
    "target_size": 64,
    "mode": "binary",
    "model_name": "bsdi_buildings",
}, CKPT)
CKPT_SHA = sha256(CKPT)
check("checkpoint written", os.path.isfile(CKPT))

meta = dlpk.read_checkpoint_meta(CKPT)
check("read_checkpoint_meta finds model_type", meta.get("model_type") == "unet", meta.get("model_type"))
check("read_checkpoint_meta finds class_names", meta.get("class_names") == CLASS_NAMES, meta.get("class_names"))
check("read_checkpoint_meta reports a state_dict", meta.get("has_state_dict") is True)
check("read_checkpoint_meta carries no tensors",
      all(not torch.is_tensor(v) for v in meta.values()))


# =========================================================================
print("\n--- 3. export to .dlpk ---")
# =========================================================================
def builder(_meta):
    loaded, _ = appmod.load_checkpoint(CKPT, torch.device("cpu"))
    return loaded


OUT = os.path.join(WORK, "bsdi_buildings.dlpk")
result = dlpk.export_dlpk(
    CKPT, OUT, meta=meta, model_name="bsdi_buildings",
    torchscript_builder=builder,
)
check("export returned success", result.get("success") is True, result)
check(".dlpk file exists", os.path.isfile(result["dlpk_path"]))
check("native .pth still on disk (export is additional, not a move)",
      os.path.isfile(CKPT) and sha256(CKPT) == CKPT_SHA)
if result.get("warnings"):
    print("         export warnings: " + "; ".join(result["warnings"]))


# =========================================================================
print("\n--- 4. re-open the zip ---")
# =========================================================================
with zipfile.ZipFile(result["dlpk_path"], "r") as package:
    names = package.namelist()
    emd_raw = package.read("bsdi_buildings.emd").decode("utf-8")
    pth_bytes = package.read("bsdi_buildings.pth")
    fn_source = package.read("BraeinAIImageClassifier.py").decode("utf-8")

check(".dlpk is a readable zip", len(names) >= 4, names)
check("zip holds the .emd", "bsdi_buildings.emd" in names, names)
check("zip holds the native .pth", "bsdi_buildings.pth" in names, names)
check("zip holds the inference function", "BraeinAIImageClassifier.py" in names, names)
check("zip holds a README for the reviewer", "README.txt" in names, names)
check("packaged .pth is byte-identical to the checkpoint",
      hashlib.sha256(pth_bytes).hexdigest() == CKPT_SHA)
def _ast_ok(text):
    import ast
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


check("bundled inference function parses as Python", _ast_ok(fn_source))
check("bundled inference function declares the Esri contract",
      all(m in fn_source for m in
          ("def initialize", "def getParameterInfo", "def getConfiguration",
           "def updatePixels")))

# The TorchScript module is the whole reason the package could run anywhere
# without the Braein.ai source tree. Prove it is a real, loadable module and
# that it agrees with the eager model - that part IS verifiable here, even
# though the ArcGIS plumbing around it is not.
ts_name = result.get("torchscript")
check("TorchScript module was bundled", bool(ts_name) and ts_name in names,
      result.get("warnings"))
if ts_name:
    with zipfile.ZipFile(result["dlpk_path"], "r") as package:
        ts_bytes = package.read(ts_name)
    scripted = torch.jit.load(io.BytesIO(ts_bytes), map_location="cpu")
    scripted.eval()
    probe = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        eager_out = reloaded_for_ts = None
        eager_model, _ = appmod.load_checkpoint(CKPT, torch.device("cpu"))
        eager_out = eager_model(probe)
        ts_out = scripted(probe)
    check("TorchScript module loads standalone (no Braein.ai source needed)",
          ts_out is not None)
    check("TorchScript output matches the eager model",
          torch.allclose(eager_out, ts_out, atol=1e-5),
          float((eager_out - ts_out).abs().max()))
    check("EMD ModelFile points at the TorchScript module",
          json.loads(emd_raw).get("ModelFile") == ts_name,
          json.loads(emd_raw).get("ModelFile"))


# =========================================================================
print("\n--- 5. the .emd ---")
# =========================================================================
emd = json.loads(emd_raw)
check("EMD is valid JSON object", isinstance(emd, dict))
check("EMD Framework = PyTorch", emd.get("Framework") == "PyTorch", emd.get("Framework"))
check("EMD has ModelConfiguration", bool(emd.get("ModelConfiguration")), emd.get("ModelConfiguration"))
check("EMD ModelFile names a file inside the package",
      emd.get("ModelFile") in names, emd.get("ModelFile"))
check("EMD ModelType = ImageClassification (pixel classification)",
      emd.get("ModelType") == "ImageClassification", emd.get("ModelType"))
check("EMD ImageHeight = 64", emd.get("ImageHeight") == 64, emd.get("ImageHeight"))
check("EMD ImageWidth = 64", emd.get("ImageWidth") == 64, emd.get("ImageWidth"))
check("EMD ExtractBands = [0,1,2]", emd.get("ExtractBands") == [0, 1, 2], emd.get("ExtractBands"))
check("EMD InferenceFunction points at the bundled file",
      emd.get("InferenceFunction") in names, emd.get("InferenceFunction"))
classes = emd.get("Classes") or []
check("EMD Classes has one entry per class", len(classes) == 2, classes)
check("EMD Classes carry Value, Name and Color",
      all(set(("Value", "Name", "Color")).issubset(c) for c in classes), classes)
check("EMD Classes colours are RGB triples",
      all(isinstance(c["Color"], list) and len(c["Color"]) == 3 for c in classes), classes)
check("EMD class names match the checkpoint",
      [c["Name"] for c in classes] == CLASS_NAMES, classes)
check("EMD MetaDataMode = Classified_Tiles",
      emd.get("MetaDataMode") == "Classified_Tiles", emd.get("MetaDataMode"))
ext = emd.get("BraeinAI") or {}
check("EMD provenance names the producer",
      str(ext.get("Producer", "")).startswith("Braein.ai"), ext.get("Producer"))
check("EMD provenance records the inference task",
      ext.get("InferenceTask") == "ClassifyPixelsUsingDeepLearning", ext.get("InferenceTask"))
check("EMD provenance records epoch and IoU from the checkpoint",
      ext.get("Epoch") == 7 and abs(float(ext.get("BestIoU")) - 0.8123) < 1e-6, ext)
check("EMD never claims the package is verified-runnable",
      ext.get("ArcGISRunnable") in ("unverified", "structurally_valid"),
      ext.get("ArcGISRunnable"))


# =========================================================================
print("\n--- 6. describe() ---")
# =========================================================================
info = dlpk.describe(result["dlpk_path"])
check("describe reports the framework", info["framework"] == "PyTorch", info["framework"])
check("describe reports the Esri model type",
      info["esri_model_type"] == "ImageClassification", info["esri_model_type"])
check("describe reports the inference task",
      info["inference_task"] == "ClassifyPixelsUsingDeepLearning", info["inference_task"])
check("describe reports the chip metadata format",
      info["chip_metadata_format"] == "Classified_Tiles", info["chip_metadata_format"])
check("describe reports the Braein.ai model type", info["model_type"] == "unet", info["model_type"])
check("describe reports the tile size", info["tile_size"] == 64, info["tile_size"])
check("describe reports the classes", info["class_names"] == CLASS_NAMES, info["class_names"])
check("describe confirms the model file is present", info["model_file_present"] is True)
check("describe flags Braein.ai origin", info["braein_origin"] is True)
check("describe lists bundled Python files",
      info["bundled_python_files"] == ["BraeinAIImageClassifier.py"],
      info["bundled_python_files"])


# =========================================================================
print("\n--- 7. import the package back ---")
# =========================================================================
IMPORT_DIR = os.path.join(WORK, "imported")
imported = dlpk.import_dlpk(result["dlpk_path"], IMPORT_DIR)
check("import returned success", imported.get("success") is True)
check("import extracted the .pth",
      imported["model_path"] and imported["model_path"].lower().endswith(".pth"),
      imported["model_path"])
check("import did NOT extract the bundled Python by default",
      imported["skipped"] == ["BraeinAIImageClassifier.py"]
      and not any(p.lower().endswith(".py") for p in imported["extracted"]),
      imported["skipped"])
check("import reports a loadable checkpoint",
      bool(imported["loadable_checkpoint"]), imported["loadable_reason"])
check("round-tripped .pth is byte-identical to the original",
      sha256(imported["model_path"]) == CKPT_SHA)

reloaded, rmeta = appmod.load_checkpoint(
    imported["loadable_checkpoint"], torch.device("cpu")
)
check("round-tripped model loads through app.load_checkpoint", reloaded is not None)
check("reloaded model_type survives the round trip", rmeta["model_type"] == "unet", rmeta["model_type"])
check("reloaded class_names survive the round trip", rmeta["class_names"] == CLASS_NAMES, rmeta["class_names"])
check("reloaded best_iou survives the round trip", abs(rmeta["best_iou"] - 0.8123) < 1e-6, rmeta["best_iou"])
with torch.no_grad():
    out = reloaded(torch.zeros(1, 3, 64, 64))
check("reloaded model runs a forward pass", tuple(out.shape)[0] == 1, tuple(out.shape))


# =========================================================================
print("\n--- 8. a foreign, ArcGIS-shaped package ---")
# =========================================================================
# No arcgis.learn here to produce a real one, so this is an EMD in the shape
# arcgis.learn writes, around a bare state_dict. It exercises the import
# path that a Deep Learning Studio package would take: no BraeinAI block, so
# the task has to come from ModelConfiguration / ModelType.
FOREIGN = os.path.join(WORK, "landcover_from_arcgis.dlpk")
bare = os.path.join(WORK, "bare_state_dict.pth")
torch.save(model.state_dict(), bare)
foreign_emd = {
    "Framework": "arcgis.learn.models._inferencing",
    "ModelConfiguration": "_unet",
    "ModelFile": "model.pth",
    "ModelType": "ImageClassification",
    "InferenceFunction": "ArcGISImageClassifier.py",
    "ImageHeight": 64,
    "ImageWidth": 64,
    "ExtractBands": [0, 1, 2],
    "Classes": [
        {"Value": 0, "Name": "background", "Color": [0, 0, 0]},
        {"Value": 1, "Name": "building", "Color": [255, 0, 0]},
    ],
}
with zipfile.ZipFile(FOREIGN, "w", zipfile.ZIP_DEFLATED) as package:
    package.writestr("landcover.emd", json.dumps(foreign_emd))
    package.write(bare, "model.pth")
    package.writestr("ArcGISImageClassifier.py", "raise SystemExit('pwned')\n")

finfo = dlpk.describe(FOREIGN)
check("foreign package: task derived from the EMD, not from keywords",
      finfo["inference_task"] == "ClassifyPixelsUsingDeepLearning", finfo["inference_task"])
check("foreign package: mapped to a Braein.ai model type",
      finfo["model_type"] == "unet", finfo["model_type"])
check("foreign package: not flagged as Braein.ai origin", finfo["braein_origin"] is False)
check("foreign package: classes read from the EMD",
      finfo["class_names"] == CLASS_NAMES, finfo["class_names"])

fimported = dlpk.import_dlpk(FOREIGN, IMPORT_DIR)
check("foreign package imports", fimported["success"] is True)
check("foreign inference function is NOT written to disk",
      fimported["skipped"] == ["ArcGISImageClassifier.py"]
      and not any(p.endswith(".py") for p in fimported["extracted"]),
      fimported["skipped"])
check("foreign bare state_dict is wrapped into a loadable checkpoint",
      bool(fimported["loadable_checkpoint"])
      and fimported["loadable_checkpoint"].endswith("_from_dlpk.pth"),
      fimported["loadable_reason"])
wrapped = torch.load(fimported["loadable_checkpoint"], map_location="cpu", weights_only=True)
check("wrapper carries model_state_dict", "model_state_dict" in wrapped)
check("wrapper maps EMD classes onto checkpoint metadata",
      wrapped.get("class_names") == CLASS_NAMES, wrapped.get("class_names"))
check("wrapper maps EMD ImageHeight onto tile_size",
      wrapped.get("tile_size") == 64, wrapped.get("tile_size"))
check("import warns that foreign layer names may not match",
      any("foreign framework" in n for n in fimported["notes"]), fimported["notes"])
fmodel, fmeta = appmod.load_checkpoint(
    fimported["loadable_checkpoint"], torch.device("cpu")
)
check("wrapped foreign checkpoint loads through app.load_checkpoint "
      "(weights here are ours, dressed as a foreign package)", fmodel is not None)


# =========================================================================
print("\n--- 9. archive safety (zip-slip, zip bomb, junk) ---")
# =========================================================================
EVIL = os.path.join(WORK, "evil.dlpk")
with zipfile.ZipFile(EVIL, "w") as package:
    package.writestr("evil.emd", json.dumps(foreign_emd))
    package.writestr("../../../../escaped.pth", b"not a model")
try:
    dlpk.import_dlpk(EVIL, IMPORT_DIR)
    check("zip-slip member is rejected", False, "import succeeded")
except dlpk.DlpkError as exc:
    check("zip-slip member is rejected", "escape" in str(exc).lower(), str(exc))
check("nothing was written outside the extraction directory",
      not os.path.exists(os.path.join(APPDIR, "..", "escaped.pth")))

EVIL2 = os.path.join(WORK, "absolute.dlpk")
with zipfile.ZipFile(EVIL2, "w") as package:
    package.writestr("evil.emd", json.dumps(foreign_emd))
    package.writestr("/etc/cron.d/evil", b"x")
try:
    dlpk.import_dlpk(EVIL2, IMPORT_DIR)
    check("absolute member path is rejected", False, "import succeeded")
except dlpk.DlpkError as exc:
    check("absolute member path is rejected", "absolute" in str(exc).lower(), str(exc))

try:
    dlpk.import_dlpk(result["dlpk_path"], IMPORT_DIR, max_extract_bytes=16)
    check("expansion-size cap refuses an oversized package", False, "import succeeded")
except dlpk.DlpkError as exc:
    check("expansion-size cap refuses an oversized package", "limit" in str(exc).lower(), str(exc))

try:
    dlpk.import_dlpk(result["dlpk_path"], IMPORT_DIR, max_members=1)
    check("member-count cap refuses a package with too many files", False, "import succeeded")
except dlpk.DlpkError as exc:
    check("member-count cap refuses a package with too many files", "limit" in str(exc).lower(), str(exc))

NOEMD = os.path.join(WORK, "no_emd.dlpk")
with zipfile.ZipFile(NOEMD, "w") as package:
    package.writestr("model.pth", b"x")
try:
    dlpk.describe(NOEMD)
    check("a package with no .emd is refused", False, "describe succeeded")
except dlpk.DlpkError as exc:
    check("a package with no .emd is refused", "emd" in str(exc).lower(), str(exc))

JUNK = os.path.join(WORK, "junk.dlpk")
with open(JUNK, "wb") as handle:
    handle.write(b"this is not a zip file")
try:
    dlpk.describe(JUNK)
    check("a non-zip .dlpk is refused", False, "describe succeeded")
except dlpk.DlpkError as exc:
    check("a non-zip .dlpk is refused", "zip" in str(exc).lower(), str(exc))


# =========================================================================
print("\n--- 10. routes: authentication and capability gating ---")
# =========================================================================
anon = TestClient(appmod.app, follow_redirects=False)
r = anon.post("/api/models/bsdi_buildings.pth/export_dlpk", json={})
check("unauthenticated export -> 401", r.status_code == 401, r.status_code)
r = anon.get("/api/models/bsdi_buildings/dlpk_info")
check("unauthenticated dlpk_info -> 401", r.status_code == 401, r.status_code)

admin_pw = auth.FIRST_RUN_PASSWORD
admin = TestClient(appmod.app, follow_redirects=False)
admin.post("/login", data={"username": "admin", "password": admin_pw, "next": "/"})
who = admin.get("/api/whoami").json()
check("admin holds can_publish_model",
      "can_publish_model" in who.get("capabilities", []), who.get("capabilities"))

auth.set_password("lab2", "a-very-long-password", role="labeler", full_name="Lab Two")
auth.set_password("ds2", "a-very-long-password", role="data_scientist", full_name="DS Two")
lab = TestClient(appmod.app, follow_redirects=False)
lab.post("/login", data={"username": "lab2", "password": "a-very-long-password", "next": "/"})
ds = TestClient(appmod.app, follow_redirects=False)
ds.post("/login", data={"username": "ds2", "password": "a-very-long-password", "next": "/"})

r = lab.post("/api/models/bsdi_buildings.pth/export_dlpk", json={})
check("labeler export_dlpk -> 403", r.status_code == 403, r.status_code)
r = ds.post("/api/models/bsdi_buildings.pth/export_dlpk", json={})
check("data scientist export_dlpk -> 403 (publishing is an admin act)",
      r.status_code == 403, r.status_code)

r = admin.post("/api/models/bsdi_buildings.pth/export_dlpk",
               json={"save_location": WORK, "model_name": "route_export"})
check("admin export_dlpk -> 200", r.status_code == 200, "%s %s" % (r.status_code, r.text[:200]))
body = r.json() if r.status_code == 200 else {}
check("route export produced a .dlpk on disk",
      bool(body.get("dlpk_path")) and os.path.isfile(body["dlpk_path"]), body.get("error"))
check("route export reports the EMD ModelType",
      (body.get("emd") or {}).get("ModelType") == "ImageClassification",
      (body.get("emd") or {}).get("ModelType"))
check("route export states an honest runnability verdict",
      body.get("arcgis_runnable") in ("unverified", "structurally_valid"),
      body.get("arcgis_runnable"))
print("         route export verdict: %s - %s"
      % (body.get("arcgis_runnable"), body.get("arcgis_runnable_reason")))

r = admin.get("/api/models/route_export/dlpk_info")
check("dlpk_info -> 200", r.status_code == 200, "%s %s" % (r.status_code, r.text[:200]))
check("dlpk_info reports the inference task",
      r.json().get("inference_task") == "ClassifyPixelsUsingDeepLearning"
      if r.status_code == 200 else False, r.text[:200])
r = lab.get("/api/models/route_export/dlpk_info")
check("labeler may read dlpk_info (read-only, not over-blocked)",
      r.status_code == 200, r.status_code)

with open(result["dlpk_path"], "rb") as handle:
    payload = handle.read()
r = lab.post("/api/models/import_dlpk",
             files={"file": ("bsdi_buildings.dlpk", payload, "application/octet-stream")})
check("labeler import_dlpk -> 403", r.status_code == 403, r.status_code)
r = ds.post("/api/models/import_dlpk",
            files={"file": ("bsdi_buildings.dlpk", payload, "application/octet-stream")})
check("data scientist import_dlpk -> 200", r.status_code == 200, "%s %s" % (r.status_code, r.text[:200]))
ibody = r.json() if r.status_code == 200 else {}
check("route import reports a loadable checkpoint",
      bool(ibody.get("loadable_checkpoint")), ibody.get("loadable_reason"))
check("route import skipped the bundled Python",
      ibody.get("skipped_python_files") == ["BraeinAIImageClassifier.py"],
      ibody.get("skipped_python_files"))

r = ds.get("/api/models")
listed = r.json().get("models", []) if r.status_code == 200 else []
check("an imported .dlpk checkpoint appears in /api/models",
      any(m.get("source") == "dlpk" for m in listed),
      [m.get("source") for m in listed])
check("the listed .dlpk model carries its metadata",
      any(m.get("source") == "dlpk" and m.get("model_type") == "unet"
          and int(m.get("tile_size") or 0) == 64 for m in listed),
      [m for m in listed if m.get("source") == "dlpk"])

r = ds.post("/api/models/import_dlpk",
            files={"file": ("model.pth", b"not a package", "application/octet-stream")})
check("import of a non-.dlpk is refused -> 400", r.status_code == 400, r.status_code)
r = ds.post("/api/models/import_dlpk",
            files={"file": ("junk.dlpk", b"not a zip", "application/octet-stream")})
check("import of a corrupt package is refused -> 400", r.status_code == 400, r.status_code)


# =========================================================================
print("\n--- 11. routes: path confinement and the licence gate ---")
# =========================================================================
r = admin.post("/api/models/..%2F..%2F..%2Fdeployment_config.py/export_dlpk", json={})
check("traversal in the model name is refused (not 200)", r.status_code != 200, r.status_code)
r = admin.post("/api/models/bsdi_buildings.pth/export_dlpk",
               json={"save_location": "C:/Windows/Temp"})
check("export to a path outside the declared roots -> 400",
      r.status_code == 400, "%s %s" % (r.status_code, r.text[:160]))
r = admin.get("/api/models/nosuchpackage/dlpk_info")
check("dlpk_info for an unknown package -> 404", r.status_code == 404, r.status_code)

# .pth and .pt stay separate: the YOLO licence gate must not be dragged into
# the U-Net workflow, and a .pt must not be packaged behind its back.
FAKE_PT = os.path.join(WORK, "fake_yolo.pt")
torch.save({"model_state_dict": {}}, FAKE_PT)
r = admin.post("/api/models/fake_yolo.pt/export_dlpk", json={})
check("export of a .pt is refused with the licence reason",
      r.status_code == 400 and "licence" in r.text.lower(),
      "%s %s" % (r.status_code, r.text[:200]))
check("YOLO stayed disabled throughout", depcfg.ENABLE_YOLO is False)

check("native .pth untouched at the end of the run",
      os.path.isfile(CKPT) and sha256(CKPT) == CKPT_SHA)


# =========================================================================
shutil.rmtree(WORK, ignore_errors=True)
shutil.rmtree(os.path.join(APPDIR, "models", "dlpk_imported"), ignore_errors=True)

print("\n" + "=" * 62)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("  FAILED: " + "; ".join(FAIL))
print("=" * 62)
sys.exit(1 if FAIL else 0)
