"""
test_auth.py - end-to-end test of login, sessions and capability gating
========================================================================
Evidence for the IGA / NCSC review: these assertions are what "the
application requires authentication" actually means, run against the real
route table rather than read off the source.

Covers the finding the previous audit could not close - that /uploads,
/chips and /outputs were served without authentication - and the role
control, that a Labeler cannot start a GPU training run even by posting to
/train directly with the button hidden.

    python test_auth.py .

Uses a throwaway users.json in a temp directory, so it never touches the
deployment's own accounts.
"""
import os, sys, json, tempfile

APPDIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__))
os.chdir(APPDIR)
sys.path.insert(0, APPDIR)

# Deterministic test environment, set BEFORE app import.
tmp_users = os.path.join(tempfile.mkdtemp(), "users.json")
os.environ["USERS_FILE"] = tmp_users
os.environ["SESSION_SECRET"] = "test-secret-not-for-production"
os.environ["API_KEY"] = "test-api-key-123"
os.environ["PBKDF2_ITERATIONS"] = "1000"      # keep the test fast
os.environ["MAX_LOGIN_ATTEMPTS"] = "3"
os.environ["AUTH_BACKEND"] = "local"

import auth
import app as appmod
from fastapi.testclient import TestClient

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + str(detail)) if detail and not cond else ""))


client = TestClient(appmod.app, follow_redirects=False)

print("\n--- route table ---")
routes = [r for r in appmod.app.routes if hasattr(r, "path")]
print("  %d routes registered" % len(routes))
paths = {r.path for r in routes}
check("/login registered", "/login" in paths)
check("/logout registered", "/logout" in paths)
check("/api/whoami registered", "/api/whoami" in paths)

print("\n--- 1. unauthenticated access ---")
r = client.get("/", headers={"accept": "text/html"})
check("GET / redirects to login", r.status_code == 303 and "/login" in r.headers.get("location", ""),
      "%s %s" % (r.status_code, r.headers.get("location")))

r = client.get("/api/projects")
check("GET /api/projects -> 401", r.status_code == 401, r.status_code)

# THE residual finding from the audit: static mounts were open.
for mount in ("/chips/x.png", "/uploads/x.png", "/outputs/x.png", "/static/app.js"):
    r = client.get(mount)
    check("mount %s is protected" % mount, r.status_code in (401, 303), r.status_code)

r = client.get("/api/health")
check("GET /api/health stays open (load balancer)", r.status_code == 200, r.status_code)

print("\n--- 2. login page ---")
r = client.get("/login")
check("GET /login -> 200", r.status_code == 200, r.status_code)
check("login form present", 'name="password"' in r.text and 'name="username"' in r.text)
check("login page loads no external asset", "unpkg.com" not in r.text and "cdnjs" not in r.text)

print("\n--- 3. bootstrap admin ---")
users = auth.load_users()["users"]
check("admin created on first run", "admin" in users, list(users))
check("admin password is hashed, not stored", users.get("admin", {}).get("password", "").startswith("pbkdf2_sha256$"))
admin_pw = auth.FIRST_RUN_PASSWORD
check("first-run password was generated", bool(admin_pw) and len(admin_pw) >= 12)

print("\n--- 4. bad credentials ---")
r = client.post("/login", data={"username": "admin", "password": "wrong", "next": "/"})
check("wrong password -> 401", r.status_code == 401, r.status_code)
check("no session cookie issued", auth.SESSION_COOKIE not in r.cookies)
r2 = client.post("/login", data={"username": "nosuchuser", "password": "wrong", "next": "/"})
check("unknown user gives same message as wrong password",
      ("Incorrect username or password" in r.text) and ("Incorrect username or password" in r2.text))

print("\n--- 5. lockout ---")
for i in range(3):
    client.post("/login", data={"username": "locktest", "password": "x", "next": "/"})
r = client.post("/login", data={"username": "locktest", "password": "x", "next": "/"})
check("locked out after 3 failures -> 429", r.status_code == 429, r.status_code)

print("\n--- 6. successful login ---")
c = TestClient(appmod.app, follow_redirects=False)
r = c.post("/login", data={"username": "admin", "password": admin_pw, "next": "/"})
check("correct password -> 303", r.status_code == 303, r.status_code)
cookie_hdr = r.headers.get("set-cookie", "")
check("session cookie set", auth.SESSION_COOKIE in cookie_hdr, cookie_hdr[:120])
check("cookie is HttpOnly", "httponly" in cookie_hdr.lower(), cookie_hdr[:120])
check("cookie is SameSite=Lax", "samesite=lax" in cookie_hdr.lower(), cookie_hdr[:120])

r = c.get("/", headers={"accept": "text/html"})
check("GET / with session -> 200", r.status_code == 200, r.status_code)
check("page no longer embeds an API key", 'name="geoai-api-key"' not in r.text)
check("page carries capabilities on body", 'data-capabilities="' in r.text)

r = c.get("/api/whoami")
who = r.json() if r.status_code == 200 else {}
check("whoami authenticated", who.get("authenticated") is True, r.status_code)
check("whoami role is admin", who.get("role") == "admin", who.get("role"))
check("admin has can_train", "can_train" in who.get("capabilities", []), who.get("capabilities"))

r = c.get("/chips/nothing.png")
check("signed-in chip request is no longer 401", r.status_code != 401, r.status_code)

print("\n--- 7. capability gating (the GPU control) ---")
auth.set_password("labeler1", "a-very-long-password", role="labeler", full_name="Lab One")
auth.set_password("ds1", "a-very-long-password", role="data_scientist", full_name="DS One")

cl = TestClient(appmod.app, follow_redirects=False)
cl.post("/login", data={"username": "labeler1", "password": "a-very-long-password", "next": "/"})
w = cl.get("/api/whoami").json()
check("labeler role resolved", w.get("role") == "labeler", w.get("role"))
check("labeler has NO can_train", "can_train" not in w.get("capabilities", []), w.get("capabilities"))
r = cl.post("/train", json={})
check("labeler POST /train -> 403", r.status_code == 403, "%s %s" % (r.status_code, r.text[:120]))
r = cl.post("/api/train/stop")
check("labeler POST /api/train/stop -> 403", r.status_code == 403, r.status_code)
r = cl.get("/api/auth/users")
check("labeler cannot list users -> 403", r.status_code == 403, r.status_code)
r = cl.get("/api/projects")
check("labeler CAN read projects (not over-blocked)", r.status_code == 200, r.status_code)

cd = TestClient(appmod.app, follow_redirects=False)
cd.post("/login", data={"username": "ds1", "password": "a-very-long-password", "next": "/"})
w = cd.get("/api/whoami").json()
check("data scientist has can_train", "can_train" in w.get("capabilities", []), w.get("capabilities"))
r = cd.post("/train", json={})
check("data scientist POST /train is NOT 403", r.status_code != 403, r.status_code)
r = cd.get("/api/auth/users")
check("data scientist cannot list users -> 403", r.status_code == 403, r.status_code)

print("\n--- 8. open redirect ---")
c2 = TestClient(appmod.app, follow_redirects=False)
for evil in ("https://evil.example/x", "//evil.example/x"):
    r = c2.post("/login", data={"username": "ds1", "password": "a-very-long-password", "next": evil})
    loc = r.headers.get("location", "")
    check("next=%s is neutralised" % evil, "evil.example" not in loc, loc)

print("\n--- 9. API key still works for scripts ---")
cs = TestClient(appmod.app, follow_redirects=False)
r = cs.get("/api/projects", headers={"X-API-Key": "test-api-key-123"})
check("script with API key -> 200", r.status_code == 200, r.status_code)
r = cs.get("/api/projects", headers={"X-API-Key": "wrong-key"})
check("script with wrong key -> 401", r.status_code == 401, r.status_code)

print("\n--- 10. logout ---")
r = c.get("/logout")
check("logout -> 303", r.status_code == 303, r.status_code)
c.cookies.clear()
r = c.get("/api/whoami")
check("after logout, whoami -> 401", r.status_code == 401, r.status_code)

print("\n" + "=" * 62)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("  FAILED: " + "; ".join(FAIL))
print("=" * 62)
sys.exit(1 if FAIL else 0)
