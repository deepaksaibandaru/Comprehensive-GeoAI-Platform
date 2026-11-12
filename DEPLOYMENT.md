# Braein.ai — Windows Server / IIS deployment

BSDI GeoAI Studio, for IGA. Read `deployment_config.py` alongside this: every
setting named here lives there, and that file is the surface the NCSC review
is run from.

---

## 1. Topology

```
   browser  ==HTTPS==>  IIS  (TLS, logging, request filtering)
                         |   http://127.0.0.1:8000     loopback only
                         +=>  uvicorn / FastAPI  (run_server.py)
                                 |
                                 +=>  PyTorch, filesystem under the app dir
```

The Python process **never** binds a routable address. `HOST` stays
`127.0.0.1`. IIS is the only thing listening on the network, so TLS,
certificate lifecycle, IP restrictions and the W3C access log are all handled
by the platform the IGA operations team already runs.

---

## 2. Why ARR, not HttpPlatformHandler

Both keep the app on loopback. The deciding factors were the two things this
application cannot work without.

| | **ARR + URL Rewrite** (chosen) | HttpPlatformHandler |
|---|---|---|
| **WebSockets** (`/ws/training`, `/ws`) | Documented and supported with the IIS *WebSocket Protocol* feature installed. One caveat, handled below. | Not documented in the Microsoft configuration reference at all. Community reports are mixed. The training progress panel is the demo. |
| **`X-Forwarded-For`** | ARR appends it automatically. `auth.py` throttles failed logins per username + address, so this has to be real. | The reference documents only `X-IIS-WindowsAuthToken`. No forwarded-address guarantee, so every client would look like the proxy and lockouts would aggregate across users. |
| Process lifetime | Not managed — needs a scheduled task or service wrapper (step 6). | Managed by IIS, which is genuinely nicer. |
| Config surface | Server-level "Enable proxy" switch, which is a governance item (§8). | None. |
| Maintenance | URL Rewrite 2.1 / ARR 3.0, both current. | Last release 1.2 (2016). |

HttpPlatformHandler wins on process lifetime and loses on the two
requirements that are not negotiable. ARR was chosen and the process-lifetime
gap is closed in step 6. The HttpPlatformHandler configuration is in the
appendix if IGA's hosting standard mandates it — adopt it only after proving
a WebSocket upgrade survives it, and expect to lose per-client login
throttling.

---

## 3. Prerequisites on the server

Install, in this order:

1. **IIS** with: *Web Server → Application Development → **WebSocket Protocol***.
   Without this feature ARR cannot carry `/ws/training`, the training panel
   stays empty, and nothing in any log says why.
2. **URL Rewrite 2.1**.
3. **Application Request Routing 3.0**.
4. **Python 3.11+** with this app's `requirements.txt` installed, ideally in a
   venv beside the application.

Then, once, at **server** level:

- IIS Manager → server node → *Application Request Routing Cache* →
  *Server Proxy Settings* → tick **Enable proxy**.
- Set **Time-out (seconds)** to `600`. Training requests and long polls
  outlive the 120 s default.
- Set **Response buffer threshold** to `0` so streamed responses are not
  buffered.

---

## 4. Install

1. Put the application anywhere **outside** `C:\inetpub` — e.g.
   `D:\BSDI\geoai\`. IIS must not be able to serve it.
2. Create an **empty** folder for the IIS site, e.g. `C:\inetpub\braein-proxy`,
   and copy **`web.config`** from the application folder into it. That folder
   is the site's physical path and holds nothing else. A site rooted on the
   application folder is one mis-configured handler away from serving `.env`,
   `users.json` or a trained `.pth` over HTTP.
3. Create the IIS site, bind **https** with the IGA certificate, and remove
   the http binding (or add an http→https redirect rule).
4. Allow the two forwarded headers the rewrite rule sets, once per server:

   ```powershell
   Add-WebConfiguration -Filter "/system.webServer/rewrite/allowedServerVariables" `
     -Value @{name="HTTP_X_FORWARDED_PROTO"} -PSPath "IIS:\"
   Add-WebConfiguration -Filter "/system.webServer/rewrite/allowedServerVariables" `
     -Value @{name="HTTP_X_FORWARDED_HOST"} -PSPath "IIS:\"
   ```

   Without this IIS returns **500.19** on the first request. `X-Forwarded-For`
   is added by ARR itself and does **not** need whitelisting.

   If `web.config` is rejected with *"This configuration section cannot be
   used at this path"*, unlock the section:

   ```powershell
   & $env:windir\system32\inetsrv\appcmd.exe unlock config -section:system.webServer/webSocket
   ```

---

## 5. Configure the application

```powershell
Copy-Item .env.example .env
```

Then edit `.env`. Every variable is documented there with the reason it
matters. The minimum for a non-loopback deployment:

| Variable | Why |
|---|---|
| `SESSION_SECRET` | Without it, cookies are signed with a per-process key: every restart signs everyone out and two workers reject each other's cookies. |
| `SESSION_COOKIE_SECURE=true` | IIS terminates TLS; without this the session cookie is sent over plain HTTP. |
| `API_KEY` | A stable key for scripts. Unset means a new one every restart. |
| `CORS_ALLOW_ORIGINS` | The real published origin. |
| `IGA_IMAGE_SERVER` | **The 34:05 finding.** Unset, Bahrain imagery is fetched from a public basemap. |
| `WS_PER_MESSAGE_DEFLATE=false` | ARR 3.0 cannot route compressed WebSocket frames (§7). |
| `FORWARDED_ALLOW_IPS=127.0.0.1` | Trust `X-Forwarded-For` only from IIS. |

`HOST` stays `127.0.0.1`. **`run_server.py` refuses to start** — exit code 2,
not a warning — if `HOST` is not loopback while `SESSION_SECRET` is unset or
`SESSION_COOKIE_SECURE` is off.

`.env` holds secrets. Restrict it to the service account:

```powershell
icacls D:\BSDI\geoai\.env /inheritance:r /grant "IIS APPPOOL\BraeinAI:(R)" /grant "Administrators:(F)"
```

The same applies to `users.json`.

---

## 6. Keep the Python process running

ARR does not manage process lifetime. Zero extra dependencies, using the
built-in scheduler:

```powershell
$action  = New-ScheduledTaskAction -Execute "D:\BSDI\geoai\.venv\Scripts\python.exe" `
                                   -Argument "-X utf8 run_server.py" `
                                   -WorkingDirectory "D:\BSDI\geoai"
$trigger = New-ScheduledTaskTrigger -AtStartup
$set     = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
                                        -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName "BraeinAI" -Action $action -Trigger $trigger `
                       -Settings $set -User "DOMAIN\svc_braein" -RunLevel Limited
```

`-X utf8` matters: the startup banner contains emoji and a cp1252 console
raises `UnicodeEncodeError` before the server binds, which looks exactly like
the application failing to start.

A service wrapper (NSSM, WinSW) gives better restart semantics. Both are
third-party and would need to go on the supply-chain register, so they are
an operations decision, not a default here.

Redirect stdout to a file and keep it: the startup banner is the record of
the deployment's security posture at the moment it started.

---

## 7. WebSockets — the two things that break them

1. **The IIS WebSocket Protocol feature must be installed.** `web.config`
   contains `<webSocket enabled="true" />`, but the element is inert without
   the Windows feature.
2. **ARR 3.0 cannot route compressed WebSocket frames.** uvicorn negotiates
   `permessage-deflate` by default, so `/ws/training` fails to establish and
   the training panel simply stays empty. Set `WS_PER_MESSAGE_DEFLATE=false`
   in `.env`; `run_server.py` passes it to uvicorn.

The client derives `wss://` from `window.location.protocol`, so TLS
termination at IIS needs no frontend change.

---

## 8. What this costs, in governance terms

State these in the IGA architecture document:

- **"Enable proxy" is a server-level ARR switch.** It makes the IIS instance
  capable of proxying, so the site bindings and the single rewrite rule are
  what keep it from acting as an open forward proxy. Do not host unrelated
  sites on this instance without re-reviewing that.
- **`X-Forwarded-For` is trusted only from `127.0.0.1`** (`FORWARDED_ALLOW_IPS`).
  Widening it would let any caller forge its source address in the login
  throttle counter and the access log.
- **The session cookie is the authentication mechanism.** No outbound URL
  Rewrite rule may touch `Set-Cookie`. `web.config` ships with an empty
  `<outboundRules />` for that reason.
- **IIS serves no file from disk.** All access control, including on `/chips`,
  `/uploads` and `/outputs`, is the application's session middleware.
  Introducing an IIS static-file route would bypass it.
- **No new outbound host.** With `ALLOW_PRETRAINED_DOWNLOAD=false` and
  `IGA_IMAGE_SERVER` set, the running application calls nothing outside the
  IGA estate except the configured geocoder — set `GEOCODER_URL` to an IGA
  service or leave it empty to disable place search.

---

## 9. Pre-trained weights (air gap)

`pretrained=True` used to fetch backbone weights from `download.pytorch.org`
on first use. That host is in no allowlist and does not resolve on an
air-gapped network — and several code paths in `models/` caught the failure
and trained from **random initialisation** instead, reporting success.

Now: `ALLOW_PRETRAINED_DOWNLOAD=false` (default), weights load from
`TORCH_HOME`, and a missing weight stops training with an error naming the
file.

On a **connected staging machine**:

```powershell
python stage_torch_weights.py          # downloads, hashes, writes MANIFEST.txt
```

Copy `models\torch_home\` to the server, then **on the server**:

```powershell
python stage_torch_weights.py --verify   # offline, checks every SHA-256
python stage_torch_weights.py --list     # what is needed, what is present
```

9 files, ~1 GB, pinned to the torchvision version recorded in the manifest.
The startup banner reports how many are staged.

---

## 10. Verify the deployment

Run through this before handing over. Items 3–6 are the ones that silently
fail.

1. `python run_server.py` on the server prints the configuration banner, and
   the banner says `BASEMAP RESIDENCY: IGA-HOSTED`.
2. `https://<host>/api/health` returns 200 without a session.
3. `https://<host>/` redirects to `/login`; signing in lands on the app, and
   the header badge reads **BASEMAP: IGA**.
4. A chip image (`/chips/...`) loads **while signed in**, and returns 401 or a
   redirect in a private window. That is the control the cookie exists for.
5. Start a short training run and confirm the progress panel updates — that
   is the WebSocket working end to end through ARR.
6. In the browser devtools, the session cookie shows `Secure`, `HttpOnly`,
   `SameSite=Lax`.
7. Fail a login three times from a second machine and confirm the lockout
   applies to that machine, not to everyone — proves `X-Forwarded-For` arrives.
8. `python test_auth.py .` passes 46/46 on the server.

---

## Appendix — HttpPlatformHandler alternative

Only if IGA's hosting standard requires it. IIS then owns the process
lifetime (step 6 becomes unnecessary) and assigns the loopback port in
`HTTP_PLATFORM_PORT`, which `run_server.py` already honours.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="httpplatformhandler" path="*" verb="*"
           modules="httpPlatformHandler" resourceType="Unspecified" />
    </handlers>
    <httpPlatform processPath="D:\BSDI\geoai\.venv\Scripts\python.exe"
                  arguments="-X utf8 run_server.py"
                  processesPerApplication="1"
                  startupTimeLimit="120"
                  requestTimeout="00:20:00"
                  stdoutLogEnabled="true"
                  stdoutLogFile="D:\BSDI\geoai\logs\stdout.log">
      <environmentVariables>
        <environmentVariable name="HOST" value="127.0.0.1" />
        <environmentVariable name="SESSION_COOKIE_SECURE" value="true" />
      </environmentVariables>
    </httpPlatform>
    <httpErrors existingResponse="PassThrough" />
  </system.webServer>
</configuration>
```

Before adopting it, prove both of these on the target server:

- a WebSocket upgrade to `/ws/training` completes and streams;
- `X-Forwarded-For` reaches the app — otherwise every user shares one
  throttle bucket and one user's failed logins lock out the rest.

`processPath` must be the venv's `python.exe`, and `startupTimeLimit` must
allow for importing torch, which takes tens of seconds on first start.
