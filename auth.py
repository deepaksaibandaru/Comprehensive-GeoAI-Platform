"""
auth.py - user login, sessions and capabilities for Braein.ai
==============================================================
Replaces "anyone who can load / gets the API key" with a real login.

WHY A COOKIE AND NOT A HEADER
-----------------------------
The previous design put an API key in a meta tag and attached it as an
X-API-Key header from JavaScript. That left a hole the audit flagged and
could not close: a browser cannot attach a custom header to

    <img src="/chips/...">            Leaflet tiles, chip thumbnails
    new WebSocket("/ws/training")     training progress
    window.location = "/outputs/..."  downloads

so /uploads, /chips and /outputs had to stay open, and the page that handed
out the key had to stay open too - which meant the key was open.

A signed session cookie is sent automatically on every one of those. That
is what makes it possible to authenticate the static mounts, and it is the
reason this is a cookie and not a bearer token.

The API key still works for scripts and machine callers (section 7 of
deployment_config). Humans get sessions; machines get keys.

IDENTITY SOURCES
----------------
  local    users.json beside this file; PBKDF2-SHA256 password hashes.
           For development, demos, and standalone use.

  arcgis   ArcGIS Portal generateToken, then /community/self to read the
           user's group membership. Roles come from Portal groups, not
           from ArcGIS org roles, which are too coarse to express
           "may spend GPU budget". This is the IGA production path.

CAPABILITIES, NOT ROLES
-----------------------
Screens and routes test capabilities (can_train), never roles
(role == "admin"). Adding a role must not mean auditing every call site.
The matrix mirrors docs/01_Solution_Architecture.md section 9.1.

NO NEW DEPENDENCY
-----------------
Standard library only - hmac, hashlib, secrets, base64, json. The NCSC
supply-chain register does not grow because we added a login.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.environ.get("USERS_FILE", os.path.join(BASE_DIR, "users.json"))

# =========================================================================
# Configuration
# =========================================================================

# How a user proves who they are: "local", "arcgis", or "both".
AUTH_BACKEND = os.environ.get("AUTH_BACKEND", "local").strip().lower()

# Signs session cookies. Set this in production: without it sessions are
# signed with a per-process key, so every restart logs everyone out and two
# workers cannot validate each other's cookies.
SESSION_SECRET = os.environ.get("SESSION_SECRET", "").strip()
SESSION_SECRET_IS_EPHEMERAL = False
if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_urlsafe(48)
    SESSION_SECRET_IS_EPHEMERAL = True

SESSION_COOKIE = os.environ.get("SESSION_COOKIE", "geoai_session")
SESSION_HOURS = int(os.environ.get("SESSION_HOURS", "8"))

# Secure cookies require HTTPS. Off by default because the dev server is
# plain http on loopback; MUST be on behind the IGA load balancer.
SESSION_COOKIE_SECURE = os.environ.get(
    "SESSION_COOKIE_SECURE", ""
).strip().lower() in ("1", "true", "yes", "on")

# Lockout after repeated failures, per username+address.
MAX_LOGIN_ATTEMPTS = int(os.environ.get("MAX_LOGIN_ATTEMPTS", "5"))
LOCKOUT_SECONDS = int(os.environ.get("LOCKOUT_SECONDS", "300"))

# PBKDF2 cost. 600k is the OWASP 2023 figure for SHA-256.
PBKDF2_ITERATIONS = int(os.environ.get("PBKDF2_ITERATIONS", "600000"))

# ArcGIS Portal, for AUTH_BACKEND=arcgis
PORTAL_URL = os.environ.get("ARCGIS_PORTAL_URL", "").rstrip("/")
PORTAL_REFERER = os.environ.get("ARCGIS_REFERER", "https://localhost")
PORTAL_VERIFY_TLS = os.environ.get(
    "ARCGIS_VERIFY_TLS", "true"
).strip().lower() not in ("0", "false", "no")


# Portal group title (or id) -> role. Set ROLE_GROUPS as
#   "GeoAI Data Scientists=data_scientist,GeoAI Reviewers=reviewer"
def _parse_role_groups(raw):
    out = {}
    for pair in (raw or "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip().lower()
            if k and v:
                out[k.lower()] = v
    return out


ROLE_GROUPS = _parse_role_groups(os.environ.get("ROLE_GROUPS", ""))

# Role granted to a Portal user in none of the mapped groups. "labeler" is
# deliberate: the safe default can label, and cannot spend GPU budget.
DEFAULT_PORTAL_ROLE = os.environ.get("DEFAULT_PORTAL_ROLE", "labeler").lower()


# =========================================================================
# Capability matrix  (docs/01_Solution_Architecture.md section 9.1)
# =========================================================================
# can_train is the one that costs money. Training runs on the p4d Spot
# fleet against roughly 80 GPU-hours a month; one careless run can take a
# large share of it, and a model that underpins published national data is
# an accountable act. It is not a convenience toggle.

CAPABILITIES = (
    "can_label",
    "can_export_chips",
    "can_train",
    "can_infer",
    "can_review",
    "can_publish_results",
    "can_publish_model",
    "can_admin",
)

ROLES = {
    "labeler": {
        "title": "Labeler",
        "caps": {"can_label", "can_export_chips"},
    },
    "data_scientist": {
        "title": "Data Scientist",
        "caps": {"can_label", "can_export_chips", "can_train", "can_infer"},
    },
    "reviewer": {
        "title": "Reviewer",
        "caps": {"can_review", "can_publish_results"},
    },
    "admin": {
        "title": "Administrator",
        "caps": set(CAPABILITIES),
    },
}

# Accept the long form used in the docs.
ROLE_ALIASES = {
    "administrator": "admin",
    "datascientist": "data_scientist",
    "data-scientist": "data_scientist",
    "scientist": "data_scientist",
}


def normalize_role(role):
    r = (role or "").strip().lower().replace(" ", "_")
    r = ROLE_ALIASES.get(r, r)
    return r if r in ROLES else DEFAULT_PORTAL_ROLE


def capabilities_for(role):
    return sorted(ROLES[normalize_role(role)]["caps"])


def has_capability(user, cap):
    if not user:
        return False
    return cap in set(user.get("capabilities") or [])


# =========================================================================
# Password hashing
# =========================================================================

def hash_password(password, salt=None, iterations=None):
    """PBKDF2-HMAC-SHA256. Returns 'pbkdf2_sha256$iters$salt$hash'."""
    if salt is None:
        salt = secrets.token_hex(16)
    iterations = iterations or PBKDF2_ITERATIONS
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
    )
    return "pbkdf2_sha256$%d$%s$%s" % (iterations, salt, dk.hex())


def verify_password(password, stored):
    """Constant-time check of a password against a stored hash."""
    try:
        algo, iters, salt, _rest = (stored or "").split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        candidate = hash_password(password, salt=salt, iterations=int(iters))
    except Exception:
        return False
    return hmac.compare_digest(candidate, stored)


# =========================================================================
# Local user store
# =========================================================================

_users_lock = threading.Lock()
FIRST_RUN_PASSWORD = None     # printed once at startup, never persisted


def _write_users(data):
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, USERS_FILE)
    try:
        os.chmod(USERS_FILE, 0o600)
    except Exception:
        # Windows ACLs do not map onto POSIX modes; noted, not fatal.
        pass


def load_users():
    if not os.path.exists(USERS_FILE):
        return {"users": {}}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "users" not in data:
            return {"users": {}}
        return data
    except Exception:
        return {"users": {}}


def ensure_bootstrap_admin():
    """Create an admin on first run with a generated password.

    Never ships a default password. The password is printed once to the
    console and is not recoverable - reset it with set_password().
    """
    global FIRST_RUN_PASSWORD
    with _users_lock:
        data = load_users()
        if data["users"]:
            return None
        pw = secrets.token_urlsafe(12)
        username = os.environ.get("BOOTSTRAP_ADMIN", "admin").strip() or "admin"
        data["users"][username] = {
            "password": hash_password(pw),
            "role": "admin",
            "full_name": "Administrator",
            "created": int(time.time()),
            "must_change_password": True,
        }
        _write_users(data)
        FIRST_RUN_PASSWORD = pw
        return username, pw


def set_password(username, password, role=None, full_name=None):
    """Create or update a local user."""
    with _users_lock:
        data = load_users()
        rec = data["users"].get(username, {})
        rec["password"] = hash_password(password)
        rec["role"] = normalize_role(role or rec.get("role") or "labeler")
        if full_name is not None:
            rec["full_name"] = full_name
        rec.setdefault("created", int(time.time()))
        rec["must_change_password"] = False
        data["users"][username] = rec
        _write_users(data)
        return rec


def delete_user(username):
    with _users_lock:
        data = load_users()
        if username in data["users"]:
            del data["users"][username]
            _write_users(data)
            return True
        return False


def list_users():
    data = load_users()
    return [
        {
            "username": u,
            "role": r.get("role", "labeler"),
            "full_name": r.get("full_name", ""),
            "capabilities": capabilities_for(r.get("role")),
            "must_change_password": bool(r.get("must_change_password")),
        }
        for u, r in sorted(data["users"].items())
    ]


# =========================================================================
# Login throttling
# =========================================================================

_attempts = {}
_attempts_lock = threading.Lock()


def _throttle_key(username, client):
    return "%s|%s" % ((username or "").lower(), client or "?")


def lockout_remaining(username, client):
    with _attempts_lock:
        rec = _attempts.get(_throttle_key(username, client))
        if not rec:
            return 0
        count, first = rec
        if count < MAX_LOGIN_ATTEMPTS:
            return 0
        elapsed = time.time() - first
        return max(0, int(LOCKOUT_SECONDS - elapsed))


def record_failure(username, client):
    k = _throttle_key(username, client)
    with _attempts_lock:
        count, first = _attempts.get(k, (0, time.time()))
        if time.time() - first > LOCKOUT_SECONDS:
            count, first = 0, time.time()
        _attempts[k] = (count + 1, first)


def clear_failures(username, client):
    with _attempts_lock:
        _attempts.pop(_throttle_key(username, client), None)


# =========================================================================
# Session tokens  -  payload.signature, both base64url
# =========================================================================

def _b64e(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(txt):
    pad = "=" * (-len(txt) % 4)
    return base64.urlsafe_b64decode(txt + pad)


def _sign(payload_b64):
    return _b64e(
        hmac.new(
            SESSION_SECRET.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).digest()
    )


def issue_session(username, role, source="local", full_name=""):
    now = int(time.time())
    payload = {
        "u": username,
        "r": normalize_role(role),
        "s": source,
        "n": full_name or username,
        "iat": now,
        "exp": now + SESSION_HOURS * 3600,
    }
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return body + "." + _sign(body)


def read_session(token):
    """Validate a session cookie. Returns a user dict or None.

    The signature is checked before the payload is parsed, so a forged
    cookie never reaches json.loads.
    """
    if not token or "." not in token:
        return None
    body, _, sig = token.rpartition(".")
    if not hmac.compare_digest(_sign(body), sig):
        return None
    try:
        payload = json.loads(_b64d(body))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < time.time():
        return None

    role = normalize_role(payload.get("r"))
    return {
        "username": payload.get("u"),
        "full_name": payload.get("n") or payload.get("u"),
        "role": role,
        "role_title": ROLES[role]["title"],
        "source": payload.get("s", "local"),
        "capabilities": capabilities_for(role),
        "expires": int(payload.get("exp", 0)),
    }


# =========================================================================
# Authentication backends
# =========================================================================

def authenticate_local(username, password):
    data = load_users()
    rec = data["users"].get(username)
    if not rec:
        # Hash anyway so a missing user and a wrong password take the same
        # time; otherwise response timing enumerates valid usernames.
        hash_password(password)
        return None
    if not verify_password(password, rec.get("password", "")):
        return None
    return {
        "username": username,
        "role": normalize_role(rec.get("role")),
        "full_name": rec.get("full_name") or username,
        "source": "local",
        "must_change_password": bool(rec.get("must_change_password")),
    }


def authenticate_arcgis(username, password):
    """Sign in against ArcGIS Portal, deriving the role from group membership.

    The user's own credentials are used, never a service account: the point
    is to establish who this person is and which groups they are in.
    """
    if not PORTAL_URL:
        raise RuntimeError(
            "ARCGIS_PORTAL_URL is not set; cannot use the arcgis backend."
        )

    import requests  # local import: the local backend must not need it

    try:
        r = requests.post(
            PORTAL_URL + "/sharing/rest/generateToken",
            data={
                "username": username,
                "password": password,
                "client": "referer",
                "referer": PORTAL_REFERER,
                "expiration": SESSION_HOURS * 60,
                "f": "json",
            },
            timeout=30,
            verify=PORTAL_VERIFY_TLS,
        )
        r.raise_for_status()
        tok = r.json()
    except Exception as exc:
        raise RuntimeError("Portal unreachable: %s" % exc) from exc

    if "token" not in tok:
        return None   # bad credentials - not an error condition

    try:
        s = requests.get(
            PORTAL_URL + "/sharing/rest/community/self",
            params={"f": "json", "token": tok["token"]},
            timeout=30,
            verify=PORTAL_VERIFY_TLS,
        )
        s.raise_for_status()
        me = s.json()
    except Exception:
        me = {}

    role = DEFAULT_PORTAL_ROLE
    titles = []
    for g in (me.get("groups") or []):
        titles.append(g.get("title", ""))
        for key in (g.get("title", ""), g.get("id", "")):
            mapped = ROLE_GROUPS.get((key or "").lower())
            if mapped and mapped in ROLES:
                # Most privileged mapped group wins.
                if len(ROLES[mapped]["caps"]) > len(ROLES[normalize_role(role)]["caps"]):
                    role = mapped

    return {
        "username": me.get("username") or username,
        "role": normalize_role(role),
        "full_name": me.get("fullName") or username,
        "source": "arcgis",
        "portal_groups": titles,
        "must_change_password": False,
    }


def authenticate(username, password):
    """Try the configured backend(s). Returns a user dict or None."""
    if AUTH_BACKEND in ("arcgis", "both") and PORTAL_URL:
        try:
            u = authenticate_arcgis(username, password)
            if u:
                return u
        except RuntimeError:
            if AUTH_BACKEND == "arcgis":
                raise
            # "both": fall through to the local store when the portal is down.
    if AUTH_BACKEND in ("local", "both"):
        return authenticate_local(username, password)
    return None


# =========================================================================
# Startup banner
# =========================================================================

def summary():
    lines = [
        "  Login              : enabled (%s backend)" % AUTH_BACKEND,
        "  Session cookie     : %s, %dh, HttpOnly, SameSite=Lax%s"
        % (SESSION_COOKIE, SESSION_HOURS,
           ", Secure" if SESSION_COOKIE_SECURE else ""),
        "  Local users        : %d in %s"
        % (len(load_users()["users"]), USERS_FILE),
    ]
    if AUTH_BACKEND in ("arcgis", "both"):
        lines.append("  Portal             : "
                     + (PORTAL_URL or "(ARCGIS_PORTAL_URL not set)"))
        lines.append("  Role groups        : "
                     + (", ".join(ROLE_GROUPS)
                        or "(none mapped - every user gets %s)" % DEFAULT_PORTAL_ROLE))
    if SESSION_SECRET_IS_EPHEMERAL:
        lines.append("  !! SESSION_SECRET not set: sessions end at restart and")
        lines.append("     multiple workers will reject each other's cookies.")
    if not SESSION_COOKIE_SECURE:
        lines.append("  !! SESSION_COOKIE_SECURE is off - set it behind HTTPS.")
    return "\n".join(lines)
