"""Reassemble split model parts (< 100MB chunks stored in Git) into complete .dlpk and .pth model files."""
import os
import glob

def reassemble_models():
    base_dirs = ["models/dlpk_imported", "models/saved", "models/imported"]
    for d in base_dirs:
        if not os.path.exists(d):
            continue
        part0_files = glob.glob(os.path.join(d, "*.part00")) + glob.glob(os.path.join(d, "*.part0"))
        for part0 in part0_files:
            # Determine output filename: e.g. 'models/dlpk_imported/package.dlpk' from '...package.dlpk.part00'
            out_file = part0.rsplit(".part", 1)[0]
            prefix = out_file + ".part"
            parts = sorted([p for p in os.listdir(d) if p.startswith(os.path.basename(prefix))])
            full_parts = [os.path.join(d, p) for p in parts]
            
            # Check if output file already exists and has the expected size
            expected_size = sum(os.path.getsize(p) for p in full_parts)
            if os.path.exists(out_file) and os.path.getsize(out_file) == expected_size:
                print(f"✅ Already reassembled: {out_file} ({expected_size / (1024*1024):.2f} MB)")
                continue

            print(f"📦 Reassembling {len(full_parts)} parts into {out_file} ...")
            with open(out_file, "wb") as out_f:
                for p in full_parts:
                    with open(p, "rb") as in_f:
                        out_f.write(in_f.read())
            print(f"🎉 Successfully restored {out_file} ({os.path.getsize(out_file) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    reassemble_models()
