"""
stage_torch_weights.py - pre-stage the torchvision backbone weights
====================================================================
Closes the last open NCSC finding: `pretrained=True` reached
download.pytorch.org at runtime, from the training machine, on a host that
was in nobody's outbound allowlist - and failed outright in an air-gapped
deployment.

Same pattern as vendor_assets.py: PINNED, HASHED, MANIFESTED. An auditor
asking "where did these .pth files come from and what guarantees they are
the ones you tested with" gets MANIFEST.txt as the answer.

Where the files go
------------------
deployment_config.TORCH_CHECKPOINT_DIR, i.e. $TORCH_HOME/hub/checkpoints,
which is exactly where torch.hub looks. deployment_config exports
TORCH_HOME before torch is imported, so the application finds them with no
further configuration.

Usage
-----
    python stage_torch_weights.py            # fetch what is missing
    python stage_torch_weights.py --list     # what is needed, what is here
    python stage_torch_weights.py --verify   # check files against MANIFEST
    python stage_torch_weights.py --force    # re-fetch everything

GOVERNANCE
----------
This script is the ONLY thing in the deployment that talks to
download.pytorch.org, and it is run deliberately by an operator on a
connected staging machine - never by the application and never on the
production host. Copy $TORCH_HOME to the air-gapped server afterwards and
run --verify there. The application itself keeps
ALLOW_PRETRAINED_DOWNLOAD=false (deployment_config section 10).

Integrity
---------
torchvision embeds a SHA-256 prefix in every weight filename
(resnet50-11ad3fa6.pth). This script checks the full digest against that
prefix, so a corrupted or substituted download is rejected at staging
time, not discovered during training.
"""

import argparse
import hashlib
import os
import re
import ssl
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deployment_config as depcfg   # noqa: E402

DEST = depcfg.TORCH_CHECKPOINT_DIR
MANIFEST = depcfg.WEIGHTS_MANIFEST

# -------------------------------------------------------------------------
# The weights this application can ask for.
#
# Every entry corresponds to a `pretrained=True` path in models/. Keep this
# list in step with the model registry:
#
#   unet.py           resnet18/34/50/101, mobilenet_v2, efficientnet_b0, vgg
#   model_factory.py  faster r-cnn resnet50 fpn, mask r-cnn resnet50 fpn,
#                     mobilenet_v2 detector backbone
#   rcnn.py           resnet backbones
#
# ultralytics / YOLO weights are deliberately NOT staged: ultralytics is
# AGPL-3.0 and gated off (deployment_config section 2). Enabling it is a
# legal decision, and staging its weights here would quietly pre-empt one.
# -------------------------------------------------------------------------
WEIGHTS = [
    ("ResNet18_Weights", "DEFAULT", "U-Net / R-CNN encoder"),
    ("ResNet34_Weights", "DEFAULT", "U-Net / R-CNN encoder"),
    ("ResNet50_Weights", "IMAGENET1K_V2", "U-Net encoder (explicit in unet.py)"),
    ("ResNet50_Weights", "DEFAULT", "R-CNN / classifier encoder"),
    ("ResNet101_Weights", "DEFAULT", "U-Net / R-CNN encoder"),
    ("MobileNet_V2_Weights", "DEFAULT", "lightweight encoder + detector backbone"),
    ("EfficientNet_B0_Weights", "DEFAULT", "lightweight encoder"),
    ("VGG16_Weights", "DEFAULT", "classic CNN encoder"),
    ("detection.FasterRCNN_ResNet50_FPN_Weights", "DEFAULT", "detection (COCO)"),
    ("detection.MaskRCNN_ResNet50_FPN_Weights", "DEFAULT", "instance segmentation (COCO)"),
]

HASH_IN_NAME = re.compile(r"-([0-9a-f]{6,})\.pth$")


def resolve():
    """(url, filename, enum, member, resolved_member, purpose) per weight.

    DEFAULT is an alias that moves between torchvision releases, so the
    member it resolves to is recorded in the manifest rather than assumed.
    """
    import torchvision
    import torchvision.models as tvm

    rows = []
    seen = set()
    for enum_path, member, purpose in WEIGHTS:
        obj = tvm
        for part in enum_path.split("."):
            obj = getattr(obj, part)
        weights = getattr(obj, member)
        url = weights.url
        if url in seen:
            continue
        seen.add(url)
        rows.append({
            "url": url,
            "file": os.path.basename(url.split("?")[0]),
            "enum": enum_path,
            "member": member,
            "resolved": weights.name,
            "purpose": purpose,
        })
    return torchvision.__version__, rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "BraeinAI/1.0"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
        with open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def check_embedded_hash(path, digest):
    """torchvision puts a SHA-256 prefix in the filename. Honour it."""
    m = HASH_IN_NAME.search(os.path.basename(path))
    if not m:
        return None            # no pin available for this file
    return digest.startswith(m.group(1))


def write_manifest(tv_version, rows):
    os.makedirs(DEST, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("Braein.ai - pre-staged torchvision backbone weights\n")
        f.write("=" * 78 + "\n")
        f.write("Generated by stage_torch_weights.py.\n\n")
        f.write("These files exist so that the application never calls\n")
        f.write("%s at runtime (deployment_config section 10,\n"
                % depcfg.PRETRAINED_WEIGHTS_HOST)
        f.write("ALLOW_PRETRAINED_DOWNLOAD=false). Copy this directory to the\n")
        f.write("air-gapped host and run --verify there.\n\n")
        f.write("torchvision : %s\n" % tv_version)
        f.write("destination : %s\n" % DEST)
        f.write("licence     : torchvision weights - BSD-3-Clause (PyTorch),\n")
        f.write("              trained on ImageNet-1k / COCO 2017\n\n")
        f.write("Files\n-----\n")
        for r in rows:
            f.write("%s  %9d\n" % (r["sha256"], r["size"]))
            f.write("    file      %s\n" % r["file"])
            f.write("    weights   %s.%s -> %s\n"
                    % (r["enum"], r["member"], r["resolved"]))
            f.write("    used for  %s\n" % r["purpose"])
            f.write("    source    %s\n\n" % r["url"])
        f.write("%d files.\n" % len(rows))


def read_manifest():
    """filename -> sha256, from a previously written manifest."""
    out = {}
    if not os.path.exists(MANIFEST):
        return out
    digest = None
    for line in open(MANIFEST, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            digest = parts[0]
        elif len(parts) == 2 and parts[0] == "file" and digest:
            out[parts[1]] = digest
            digest = None
    return out


def cmd_list():
    tv_version, rows = resolve()
    print("torchvision %s   ->  %s\n" % (tv_version, DEST))
    missing = 0
    for r in rows:
        path = os.path.join(DEST, r["file"])
        here = os.path.exists(path)
        missing += 0 if here else 1
        print("  [%s] %-44s %s.%s"
              % ("ok" if here else "  ", r["file"], r["enum"], r["member"]))
    print("\n  %d of %d staged." % (len(rows) - missing, len(rows)))
    return 1 if missing else 0


def cmd_verify():
    expected = read_manifest()
    if not expected:
        print("No MANIFEST.txt in %s - run without --verify on a connected "
              "machine first." % DEST)
        return 1
    bad = missing = 0
    for name, digest in sorted(expected.items()):
        path = os.path.join(DEST, name)
        if not os.path.exists(path):
            print("  MISSING  %s" % name)
            missing += 1
        elif sha256(path) != digest:
            print("  CHANGED  %s" % name)
            bad += 1
        else:
            print("  ok       %s" % name)
    if bad or missing:
        print("\nFAILED: %d changed, %d missing of %d"
              % (bad, missing, len(expected)))
        return 1
    print("\n  all %d weight files match the manifest" % len(expected))
    return 0


def cmd_fetch(force):
    tv_version, rows = resolve()
    print("torchvision %s" % tv_version)
    print("staging %d weight files into %s\n" % (len(rows), DEST))
    failed = 0
    done = []
    for r in rows:
        path = os.path.join(DEST, r["file"])
        try:
            if os.path.exists(path) and not force:
                how, size = "cached", os.path.getsize(path)
            else:
                size, how = fetch(r["url"], path), "fetched"
            digest = sha256(path)
            pin = check_embedded_hash(path, digest)
            if pin is False:
                os.remove(path)
                print("  REJECTED %-44s digest does not match the hash pinned "
                      "in its filename" % r["file"])
                failed += 1
                continue
            r["sha256"] = digest
            r["size"] = size
            done.append(r)
            print("  %-8s %10d B  %-44s %s"
                  % (how, size, r["file"], "pin ok" if pin else "no pin"))
        except Exception as exc:
            print("  FAILED   %-44s %s\n           <- %s"
                  % (r["file"], exc, r["url"]))
            failed += 1

    if done:
        write_manifest(tv_version, done)
        print("\n  manifest: %s" % MANIFEST)
    print("  %d ok, %d failed" % (len(done), failed))
    if failed:
        return 1
    print("\n  Now copy %s\n  to the deployment host, then run:"
          "\n      python stage_torch_weights.py --verify" % depcfg.TORCH_HOME)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Pre-stage torchvision weights.")
    ap.add_argument("--verify", action="store_true",
                    help="check staged files against MANIFEST.txt (offline)")
    ap.add_argument("--list", action="store_true",
                    help="list required weights and whether they are staged")
    ap.add_argument("--force", action="store_true", help="re-fetch everything")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    if args.verify:
        return cmd_verify()
    return cmd_fetch(args.force)


if __name__ == "__main__":
    sys.exit(main())
