"""End-to-end test of the IGA imagery path against a real ArcGIS service.

Uses Esri's public Toronto sample ImageServer as a stand-in for the IGA
Image Server: it is a genuine ArcGIS Image Server with 4 bands of uint16,
which is the shape a governed orthophoto service will have.
"""
import os, sys, tempfile

APPDIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
os.chdir(APPDIR)
sys.path.insert(0, APPDIR)

SERVICE = "https://sampleserver6.arcgisonline.com/arcgis/rest/services/Toronto/ImageServer"

os.environ["USERS_FILE"] = os.path.join(tempfile.mkdtemp(), "users.json")
os.environ["SESSION_SECRET"] = "imagery-test-secret-not-for-production"
os.environ["PBKDF2_ITERATIONS"] = "1000"
os.environ["IGA_IMAGE_SERVER"] = SERVICE
os.environ["IGA_IMAGE_SERVER_NAME"] = "Test Ortho (stands in for IGA)"
os.environ["ARCGIS_IMAGE_SERVICE"] = SERVICE
os.environ["ALLOW_PUBLIC_BASEMAPS"] = "false"

import auth
import app as appmod
from fastapi.testclient import TestClient

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (("   " + str(detail)) if detail and not cond else ""))


c = TestClient(appmod.app, follow_redirects=False)
auth.ensure_bootstrap_admin()
auth.set_password("tester", "a-very-long-password", role="admin")
r = c.post("/login", data={"username": "tester",
                           "password": "a-very-long-password", "next": "/"})
assert r.status_code == 303, r.status_code

print("\n--- 1. basemap list points at the proxy, not the imagery host ---")
svcs = c.get("/api/map-services").json()
iga = svcs["services"].get("iga_imagery", {})
check("IGA basemap is present", bool(iga), list(svcs["services"]))
check("url is the local proxy", iga.get("url") == "/api/basemap/tile/{z}/{x}/{y}", iga.get("url"))
check("imagery host is NOT in the browser url",
      "sampleserver6" not in str(iga.get("url")), iga.get("url"))
check("public basemaps removed", set(svcs["services"]) == {"iga_imagery"}, list(svcs["services"]))
check("residency reports IGA", svcs["residency"]["status"] == "IGA", svcs["residency"]["status"])

print("\n--- 2. the tile proxy actually returns pixels ---")
# Toronto, zoom 12
import arcgis_service
tx, ty = arcgis_service.tile_for_lonlat(-79.3832, 43.6532, 12)
r = c.get("/api/basemap/tile/12/%d/%d" % (tx, ty))
check("proxy tile -> 200", r.status_code == 200, r.status_code)
check("proxy returned an image", r.headers.get("content-type", "").startswith("image/"),
      r.headers.get("content-type"))
check("proxy tile has real bytes", len(r.content) > 2000, len(r.content))
check("cached privately (licensed imagery)",
      "private" in r.headers.get("cache-control", ""), r.headers.get("cache-control"))

print("\n--- 3. proxy requires a session ---")
anon = TestClient(appmod.app, follow_redirects=False)
r = anon.get("/api/basemap/tile/12/%d/%d" % (tx, ty))
check("unauthenticated proxy tile -> 401", r.status_code == 401, r.status_code)

print("\n--- 4. diagnostic explains the service ---")
d = c.get("/api/arcgis/diagnostic").json()
check("diagnostic ok", d.get("ok") is True, d)
check("probe tile is over Bahrain",
      49 < d.get("probe_bbox", [0])[0] < 52 if d.get("probe_bbox") else False,
      d.get("probe_bbox"))

print("\n--- 5. capture uses exportImage, not stitched tiles ---")
# A small AOI over Toronto, deliberately NOT aligned to tile boundaries.
bounds = {"west": -79.3850, "south": 43.6500, "east": -79.3750, "north": 43.6560}
r = c.post("/api/map/capture", json={"bounds": bounds, "zoom": 17,
                                     "service_id": "iga_imagery"})
check("capture -> 200", r.status_code == 200, "%s %s" % (r.status_code, r.text[:200]))
j = r.json() if r.status_code == 200 else {}
check("capture succeeded", j.get("success") is not False, str(j)[:200])

sb = (j.get("bounds") or j.get("snapped_bounds") or
      (appmod.training_state.image_bounds_latlon or {}))
if sb:
    exact = (abs(sb["west"] - bounds["west"]) < 1e-9 and
             abs(sb["east"] - bounds["east"]) < 1e-9 and
             abs(sb["north"] - bounds["north"]) < 1e-9 and
             abs(sb["south"] - bounds["south"]) < 1e-9)
    check("bounds are the EXACT AOI (no tile snapping)", exact,
          {k: sb.get(k) for k in ("west", "south", "east", "north")})
    check("image has real size", sb.get("image_width", 0) > 50, sb.get("image_width"))
else:
    check("capture returned bounds", False, str(j)[:200])

p = appmod.training_state.image_path
check("captured file exists on disk", bool(p) and os.path.exists(p), p)
if p and os.path.exists(p):
    from PIL import Image
    with Image.open(p) as im:
        print("        captured %dx%d %s -> %s" % (im.size[0], im.size[1], im.mode, p))

print("\n" + "=" * 62)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("  FAILED: " + "; ".join(FAIL))
print("=" * 62)
sys.exit(1 if FAIL else 0)
