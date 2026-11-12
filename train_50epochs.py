import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Set environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(os.cpu_count() or 8)

from app import (
    load_checkpoint,
    MaskRCNNChipDataset,
    detection_collate,
    _match_instance_masks,
    update_project,
    get_project
)
import dlpk

def main():
    print("=" * 60)
    print("  Braein.ai - Building Mask R-CNN Fine-Tuning (50 Epochs)")
    print("=" * 60)
    
    device = torch.device("cpu")
    print(f"Device: {device} (using {torch.get_num_threads()} CPU threads)")
    
    base_model_path = "models/saved/Building_MaskRCNN_base.pth"
    print(f"Loading base checkpoint: {base_model_path}...")
    model, meta = load_checkpoint(base_model_path, device=device, requested_num_classes=2)
    print(f"Base model loaded: {type(model).__name__} with {meta.get('num_classes', 2)} classes")
    
    train_dir = "labels/building/dataset/train"
    val_dir = "labels/building/dataset/val"
    tile_size = 256
    num_classes = 2
    class_names = ["background", "building"]
    
    train_ds = MaskRCNNChipDataset(train_dir, tile_size=tile_size, num_classes=num_classes)
    val_ds = MaskRCNNChipDataset(val_dir, tile_size=tile_size, num_classes=num_classes)
    print(f"Dataset: {len(train_ds)} train chips, {len(val_ds)} val chips")
    
    batch_size = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=detection_collate)
    
    epochs = 50
    lr = 0.0001
    weight_decay = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    models_dir = "labels/building/models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "building_best.pth")
    saved_pth = "models/saved/Building_Finetuned_50epochs.pth"
    
    best_metric = 0.0
    best_epoch = 0
    start_time = time.time()
    
    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        epoch_loss = 0.0
        nb = 0
        
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"Warning: Non-finite loss at epoch {epoch}: {losses.item()}")
                continue
                
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += float(losses.item())
            nb += 1
            
        avg_loss = epoch_loss / max(nb, 1)
        scheduler.step()
        
        # Validation
        model.eval()
        per_sample_iou = []
        with torch.no_grad():
            for images, targets in val_loader:
                predictions = model(images)
                pred, tgt = predictions[0], targets[0]
                scores, pm = pred.get("scores"), pred.get("masks")
                if pm is None:
                    pm_np = np.zeros((0, tile_size, tile_size), dtype=np.uint8)
                else:
                    pm_np = pm.detach().cpu().numpy()
                    pm_np = pm_np[:, 0] if pm_np.ndim == 4 else pm_np
                    score_np = scores.detach().cpu().numpy() if scores is not None else np.ones((len(pm_np),), dtype=np.float32)
                    pm_np = pm_np[score_np >= 0.5] > 0.5
                
                m = _match_instance_masks(pm_np, tgt["masks"].cpu().numpy() > 0, iou_threshold=0.5)
                per_sample_iou.append(float(m["iou"]))
                
        val_metric = float(np.mean(per_sample_iou)) if per_sample_iou else 0.0
        elapsed = time.time() - ep_start
        
        # Save best model
        saved_flag = ""
        if val_metric > best_metric or epoch == 1:
            best_metric = val_metric
            best_epoch = epoch
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_iou": best_metric,
                "best_epoch": best_epoch,
                "num_classes": num_classes,
                "class_names": class_names,
                "backbone": "unet",
                "model_name": "building_finetuned",
                "mode": "binary",
                "tile_size": tile_size,
                "target_size": tile_size,
                "model_type": "mask_rcnn",
                "architecture": "MaskRCNN",
                "model_config": {
                    "model_type": "mask_rcnn",
                    "architecture": "MaskRCNN",
                    "backbone": "unet",
                    "n_classes": num_classes,
                    "tile_size": tile_size,
                    "detection_min_size": (256,),
                    "detection_max_size": 512
                },
                "checkpoint_version": 2,
                "task": "instance_segmentation"
            }
            torch.save(checkpoint_payload, best_model_path)
            torch.save(checkpoint_payload, saved_pth)
            saved_flag = " ⭐ [BEST MODEL SAVED]"
            
        print(f"Epoch [{epoch:02d}/{epochs}] Loss: {avg_loss:.4f} | Mask-IoU: {val_metric:.4f} (Best: {best_metric:.4f} at ep {best_epoch}) | {elapsed:.1f}s{saved_flag}")
        sys.stdout.flush()

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Training Finished in {total_time/60:.1f} minutes!")
    print(f"  Best Model Checkpoint: {best_model_path} (IoU: {best_metric:.4f})")
    print("=" * 60)
    
    # Export to DLPK
    print("\nPackaging fine-tuned model into Esri Deep Learning Package (.dlpk)...")
    out_dlpk = "models/saved/Building_Finetuned_50epochs.dlpk"
    meta_dlpk = dlpk.read_checkpoint_meta(saved_pth)
    res = dlpk.export_dlpk(
        saved_pth,
        out_dlpk,
        meta=meta_dlpk,
        model_name="Building_Finetuned_50epochs",
        description="Building Footprint Mask R-CNN fine-tuned for 50 epochs on Bahrain vector annotations",
        include_torchscript=False
    )
    print(f"Export successful: {out_dlpk} ({os.path.getsize(out_dlpk) / (1024*1024):.1f} MB)")
    
    # Also overwrite Building_Finetuned.dlpk for convenience
    primary_dlpk = "models/saved/Building_Finetuned.dlpk"
    import shutil
    shutil.copy2(out_dlpk, primary_dlpk)
    print(f"Also updated primary: {primary_dlpk}")
    
    # Update project
    proj = get_project("proj_400d3103")
    if proj:
        update_project("proj_400d3103", {
            "has_model": True,
            "model_path": best_model_path,
            "evaluation_score": best_metric,
            "chips_count": len(train_ds)
        })
        print("Updated project Building_Model successfully.")
    
    print("\n" + "=" * 60)
    print("  ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()

