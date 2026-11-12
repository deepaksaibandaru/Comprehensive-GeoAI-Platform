import os
import sys
import threading
import time
import webbrowser

APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

# app.py's startup banner contains emoji. On a Windows console (cp1252)
# printing them raises UnicodeEncodeError before the server ever binds,
# which looks like the application failing to start. Harmless on Linux,
# where the deployment runs, but this makes local runs behave the same.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# =========================================================================
# .env  -  loaded BEFORE deployment_config / auth, which read os.environ
#          at import time.
# =========================================================================

ENV_FILE = os.environ.get("BRAEIN_ENV_FILE", os.path.join(APP_DIR, ".env"))


def load_env_file(path):
    """Minimal KEY=VALUE loader. Existing environment variables win.

    Deliberately not python-dotenv: this is twenty lines of stdlib and
    adding a dependency to read a text file is not a trade the supply-chain
    register should have to carry.

    Supports:  KEY=value, quoted values, # comments, blank lines,
               a leading "export ".
    Does not support: interpolation, multi-line values. Keep secrets on
    one line.
    """
    if not os.path.exists(path):
        return 0, []
    loaded, skipped = 0, []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not key:
                continue
            if key in os.environ:
                # The real environment is authoritative: IIS, a service
                # unit or an operator's shell must be able to override a
                # file that was shipped with the application.
                skipped.append(key)
                continue
            os.environ[key] = value
            loaded += 1
    return loaded, skipped


_loaded, _overridden = load_env_file(ENV_FILE)
if _loaded or _overridden:
    print("[env] %s: %d variable(s) loaded%s"
          % (ENV_FILE, _loaded,
             (", %d already set in the environment and left alone"
              % len(_overridden)) if _overridden else ""))
elif not os.path.exists(ENV_FILE):
    print("[env] no .env beside run_server.py - using the environment only "
          "(see .env.example)")


# IIS HttpPlatformHandler assigns the child process a private loopback port
# and passes it in HTTP_PLATFORM_PORT. Honour it unless PORT was set
# explicitly, so the same launcher works under IIS and standalone.
if os.environ.get("HTTP_PLATFORM_PORT") and not os.environ.get("PORT"):
    os.environ["PORT"] = os.environ["HTTP_PLATFORM_PORT"]
    print("[env] HTTP_PLATFORM_PORT -> PORT=" + os.environ["PORT"])


# =========================================================================
# Pre-flight  -  fail closed before anything binds
# =========================================================================

LOOPBACK = {"127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1"}


def _is_truthy(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def preflight():
    """Refuse to start a network-reachable server with broken sessions."""
    host = os.environ.get("HOST", "127.0.0.1").strip()
    loopback = host in LOOPBACK or host.startswith("127.")

    errors, warnings = [], []

    if not loopback:
        if not os.environ.get("SESSION_SECRET", "").strip():
            errors.append(
                "SESSION_SECRET is not set.\n"
                "      Session cookies would be signed with a key generated per\n"
                "      process: every restart signs everyone out, and two workers\n"
                "      reject each other's cookies. Generate one with\n"
                "          python -c \"import secrets;print(secrets.token_urlsafe(48))\""
            )
        elif len(os.environ.get("SESSION_SECRET", "").strip()) < 32:
            errors.append(
                "SESSION_SECRET is shorter than 32 characters.\n"
                "      It is the only thing standing between a forged cookie and\n"
                "      an administrator session."
            )

        if not _is_truthy("SESSION_COOKIE_SECURE"):
            errors.append(
                "SESSION_COOKIE_SECURE is off while HOST is not loopback.\n"
                "      The session cookie would be sent over plain HTTP and can be\n"
                "      read off the wire. Terminate TLS at IIS and set\n"
                "          SESSION_COOKIE_SECURE=true"
            )

        if not os.environ.get("API_KEY", "").strip():
            warnings.append(
                "API_KEY is not set: a key is generated per process, so every "
                "restart breaks script and machine callers.")

        origins = os.environ.get("CORS_ALLOW_ORIGINS", "")
        if not origins:
            warnings.append(
                "CORS_ALLOW_ORIGINS is not set: it defaults to localhost "
                "origins, which will not match the published hostname.")
        elif "localhost" in origins or "127.0.0.1" in origins:
            warnings.append(
                "CORS_ALLOW_ORIGINS still contains a localhost origin.")

        if _is_truthy("ENABLE_API_DOCS"):
            warnings.append(
                "ENABLE_API_DOCS is on: /docs publishes the whole route "
                "surface. Off in production.")

    if not os.environ.get("IGA_IMAGE_SERVER", "").strip():
        warnings.append(
            "IGA_IMAGE_SERVER is not set: basemap imagery of Bahrain will be "
            "requested from a PUBLIC service. This is the finding IGA raised "
            "on 2026-10-22 (34:05).")

    if _is_truthy("ENABLE_YOLO"):
        warnings.append(
            "ENABLE_YOLO is on: ultralytics is AGPL-3.0 and this needs legal "
            "sign-off, not an environment variable.")

    if _is_truthy("ALLOW_PRETRAINED_DOWNLOAD"):
        warnings.append(
            "ALLOW_PRETRAINED_DOWNLOAD is on: this process may call "
            "download.pytorch.org during training.")

    if warnings:
        print("\n" + "=" * 70)
        print("  PRE-FLIGHT WARNINGS")
        print("=" * 70)
        for w in warnings:
            print("  !  " + w)
        print("=" * 70 + "\n")

    if errors:
        print("\n" + "=" * 70)
        print("  REFUSING TO START - HOST=%s is not loopback" % host)
        print("=" * 70)
        for e in errors:
            print("  X  " + e)
        print("-" * 70)
        print("  These are fatal, not advisory: a reachable server with")
        print("  unsigned or interceptable sessions is worse than no server.")
        print("  Set them in .env (see .env.example) or in the IIS")
        print("  environmentVariables block, then start again.")
        print("=" * 70 + "\n")
        sys.exit(2)

    return host


HOST_CHECKED = preflight()


# =========================================================================
# Browser auto-open helper
# =========================================================================

def _open_browser_when_ready(host, port, timeout_s=20.0):
    """Poll the health endpoint, then open the app in the default browser.

    Runs in a daemon thread so it never blocks the server. Silently does
    nothing if the browser cannot be launched (e.g. headless server) or if
    the health check never succeeds. Controlled by BRAEIN_NO_BROWSER=1.
    """
    if os.environ.get("BRAEIN_NO_BROWSER", "").strip().lower() in ("1", "true", "yes", "on"):
        print("[browser] BRAEIN_NO_BROWSER is set - not opening a browser.")
        return

    # When HOST is 0.0.0.0 or ::, the browser should hit loopback.
    browser_host = host
    if host in ("0.0.0.0", "::", ""):
        browser_host = "127.0.0.1"
    url = "http://%s:%d" % (browser_host, port)

    # Try IPv4 loopback for the health check if the bind address is a wildcard.
    health_host = "127.0.0.1" if browser_host == "localhost" else browser_host
    health_url = "http://%s:%d/api/health" % (health_host, port)

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            import urllib.request
            req = urllib.request.Request(health_url, headers={"User-Agent": "BraeinAI/launcher"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if 200 <= resp.status < 400:
                    print("[browser] Server is up - opening %s" % url)
                    try:
                        webbrowser.open_new_tab(url)
                    except Exception as exc:
                        print("[browser] webbrowser.open failed: %s" % exc)
                    return
        except Exception:
            time.sleep(0.5)

    print("[browser] Server did not become ready within %ss - please open "
          "%s manually." % (timeout_s, url))


# =========================================================================
# Start
# =========================================================================

import uvicorn                      # noqa: E402
import deployment_config as depcfg  # noqa: E402
from app import app                 # noqa: E402

if __name__ == "__main__":
    # Start the browser opener BEFORE uvicorn so it can begin polling while
    # the app imports and initialises (that can take a few seconds).
    try:
        _bt = threading.Thread(
            target=_open_browser_when_ready,
            args=(depcfg.HOST, depcfg.PORT),
            daemon=True,
        )
        _bt.start()
    except Exception as _exc:
        print("[browser] could not start browser opener: %s" % _exc)

    uvicorn.run(
        app,
        host=depcfg.HOST,
        port=depcfg.PORT,
        log_level="info",
        # Behind IIS the TCP peer is always the proxy. Honour X-Forwarded-For
        # and X-Forwarded-Proto, but only from the proxy's own address -
        # otherwise any caller can forge its source address in the login
        # throttle counter and the access log (deployment_config section 11).
        proxy_headers=True,
        forwarded_allow_ips=depcfg.FORWARDED_ALLOW_IPS,
        # IIS ARR 3.0 cannot route compressed WebSocket frames; with
        # permessage-deflate negotiated, /ws/training never connects and the
        # training panel stays empty with no error.
        ws_per_message_deflate=depcfg.WS_PER_MESSAGE_DEFLATE,
    )