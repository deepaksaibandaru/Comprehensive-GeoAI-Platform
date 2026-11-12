"""
test_capabilities.py - the role/capability matrix, proved route by route
=========================================================================
Evidence for the IGA review, and the answer to "what can each role do?"
asked as something executable rather than a table in a document.

Two halves, and the second matters as much as the first:

  PART A  Every capability-gated route x every role. A role that holds the
          capability must NOT get 403; a role that does not MUST get 403.
          The check must also fire BEFORE input validation, so a denied
          user cannot probe for valid model paths or usernames.

  PART B  Which declared capabilities are NOT enforced anywhere. The matrix
          names eight; the server currently enforces three. Writing that
          down in prose lets it rot. Asserting it here means the day someone
          wires up can_infer, this test fails and has to be updated
          deliberately.

    python test_capabilities.py .

Uses a throwaway users.json in a temp directory - never touches the
deployment's own accounts.
"""
import os, sys, tempfile

APPDIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__))
os.chdir(APPDIR)
sys.path.insert(0, APPDIR)

os.environ["USERS_FILE"] = os.path.join(tempfile.mkdtemp(), "users.json")
os.environ["SESSION_SECRET"] = "capability-test-secret-not-for-production"
os.environ["PBKDF2_ITERATIONS"] = "1000"
os.environ["AUTH_BACKEND"] = "local"

import auth
import app as appmod
from fastapi.testclient import TestClient

PASS, FAIL = [], []
PW = "a-very-long-test-password"


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (("   " + str(detail)) if detail and not cond else ""))


def client_for(role):
    """A signed-in client holding exactly `role`."""
    user = "u_" + role
    auth.set_password(user, PW, role=role)
    c = TestClient(appmod.app, follow_redirects=False)
    r = c.post("/login", data={"username": user, "password": PW, "next": "/"})
    assert r.status_code == 303, (role, r.status_code)
    return c


# Every route that carries require_capability(), and which one.
# (method, path, capability, how to call it)
GATED = [
    ("GET",    "/api/auth/users",                        "can_admin",         {}),
    ("POST",   "/api/auth/users",                        "can_admin",
     {"data": {"username": "probe", "password": "x" * 14, "role": "labeler"}}),
    ("DELETE", "/api/auth/users/no-such-user-exists",    "can_admin",         {}),
    ("POST",   "/train",                                 "can_train",
     {"json": {}}),
    ("POST",   "/api/train/stop",                        "can_train",         {}),
    ("POST",   "/api/train/overfit_test",                "can_train",
     {"json": {}}),
    ("POST",   "/api/models/no-such-model.pth/export_dlpk", "can_publish_model",
     {"json": {}}),
    ("POST",   "/api/models/import_dlpk",                "can_train",         {}),
]

ROLES = ["labeler", "data_scientist", "reviewer", "admin"]


def call(c, method, path, how):
    kw = dict(how)
    if path.endswith("import_dlpk") and "files" not in kw:
        # A deliberately invalid archive: an allowed role should get a 4xx
        # about the FILE, never a 403.
        kw["files"] = {"file": ("x.dlpk", b"not a zip", "application/zip")}
    return c.request(method, path, **kw)


print("\n" + "=" * 66)
print(" PART A - every gated route x every role")
print("=" * 66)

for role in ROLES:
    caps = set(auth.capabilities_for(role))
    c = client_for(role)
    print("\n  %s  (%s)" % (auth.ROLES[role]["title"], ", ".join(sorted(caps)) or "no capabilities"))
    for method, path, cap, how in GATED:
        r = call(c, method, path, how)
        allowed = cap in caps
        short = "%-6s %-46s needs %s" % (method, path, cap)
        if allowed:
            # Must not be refused for lacking the capability. Anything else
            # (400 bad input, 404 missing) is the route doing its real job.
            check(short + "  -> permitted", r.status_code != 403, r.status_code)
        else:
            check(short + "  -> 403", r.status_code == 403,
                  "%s %s" % (r.status_code, r.text[:70]))
    # A training POST may have set state; put it back.
    if "can_train" in caps:
        c.post("/api/train/stop")

print("\n" + "=" * 66)
print(" PART A2 - the gate fires BEFORE input validation")
print("=" * 66)
# A Labeler asking for a model that does not exist must get 403, not 404.
# A 404 would confirm or deny the path and turn the route into a probe.
c = client_for("labeler")
r = c.post("/api/models/definitely-not-a-real-model.pth/export_dlpk", json={})
check("denied role gets 403, not 404, for a bogus model path",
      r.status_code == 403, r.status_code)
r = c.request("DELETE", "/api/auth/users/definitely-not-a-user")
check("denied role gets 403, not 404, for a bogus username",
      r.status_code == 403, r.status_code)

print("\n" + "=" * 66)
print(" PART B - declared capabilities that NOTHING enforces")
print("=" * 66)

import re
src = open(os.path.join(APPDIR, "app.py"), encoding="utf-8").read()
enforced = set(re.findall(r'require_capability\("([a-z_]+)"\)', src))
declared = set(auth.CAPABILITIES)
unenforced = sorted(declared - enforced)

print("  declared in auth.py : %d  (%s)" % (len(declared), ", ".join(sorted(declared))))
print("  enforced in app.py  : %d  (%s)" % (len(enforced), ", ".join(sorted(enforced))))
print()

# This is the honest state of the system, asserted so it cannot drift
# unnoticed. If you enforce one of these, update this list in the same
# commit - that is the point.
EXPECTED_UNENFORCED = [
    "can_export_chips", "can_infer", "can_label",
    "can_publish_results", "can_review",
]
check("the unenforced set is exactly what we think it is",
      unenforced == EXPECTED_UNENFORCED,
      "got %s, expected %s" % (unenforced, EXPECTED_UNENFORCED))

for cap in unenforced:
    print("     NOT ENFORCED  %s" % cap)

print()
print("  Consequence, stated plainly: any signed-in user - a Labeler")
print("  included - can run inference, export results and save/load models.")
print("  Those routes require a SESSION, not a capability. The three that")
print("  are enforced are the ones that spend GPU budget, publish a model,")
print("  or administer accounts.")

# Prove the consequence rather than only asserting the absence.
c = client_for("labeler")
r = c.post("/api/predict", json={})
check("a Labeler reaches /api/predict (not 403) - documents the gap",
      r.status_code != 403, r.status_code)
r = c.post("/api/export/geojson", json={})
check("a Labeler reaches /api/export/geojson (not 403) - documents the gap",
      r.status_code != 403, r.status_code)

print("\n" + "=" * 66)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("  FAILED: " + "; ".join(FAIL))
print("=" * 66)
sys.exit(1 if FAIL else 0)
