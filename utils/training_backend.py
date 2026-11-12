import os,json,asyncio,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T

class ClassificationDataset(Dataset):
    def __init__(self, split_dir):
        self.root=split_dir; self.img=os.path.join(split_dir,'images'); self.tgt=os.path.join(split_dir,'class_labels')
        self.files=sorted([f for f in os.listdir(self.img) if f.lower().endswith(('.png','.jpg','.jpeg')) and os.path.exists(os.path.join(self.tgt,os.path.splitext(f)[0]+'.json'))])
        self.tf=T.Compose([T.ToTensor()])
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        f=self.files[i]; x=self.tf(Image.open(os.path.join(self.img,f)).convert('RGB'))
        with open(os.path.join(self.tgt,os.path.splitext(f)[0]+'.json'),encoding='utf-8') as h: y=int(json.load(h).get('class_id',0))
        return x,y

async def classification_training_loop(epochs,batch_size,lr,label_path,safe_label,class_names,num_classes,model_name,model,optimizer,scheduler,device,grad_clip=1.0,early_stop=0,**kwargs):
    from app import training_state,add_log
    root=os.path.join(label_path,'dataset'); train=ClassificationDataset(os.path.join(root,'train')); val=ClassificationDataset(os.path.join(root,'val'))
    if len(train)==0: raise RuntimeError('Classic CNN dataset has no class_labels targets.')
    loss_fn=torch.nn.CrossEntropyLoss(); tr=DataLoader(train,batch_size=batch_size,shuffle=True); va=DataLoader(val,batch_size=batch_size or 1) if len(val) else None
    best=1e30; wait=0; training_state.is_training=True; training_state.total_epochs=epochs
    for ep in range(1,epochs+1):
        model.train(); total=0.0
        for x,y in tr:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(); z=model(x); loss=loss_fn(z,y); loss.backward();
            if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            optimizer.step(); total+=float(loss.item())*len(x)
        tl=total/max(1,len(train)); vl=tl
        if va:
            model.eval(); s=0.0; n=0
            with torch.no_grad():
                for x,y in va: s+=float(loss_fn(model(x.to(device)),y.to(device)).item())*len(x); n+=len(x)
            vl=s/max(1,n)
        if scheduler:
            try:
                scheduler.step(vl) if scheduler.__class__.__name__.startswith('ReduceLROnPlateau') else scheduler.step()
            except Exception: pass
        training_state.epoch=ep; training_state.train_losses.append(tl); training_state.val_losses.append(vl); training_state.progress_percent=100*ep/epochs
        if vl < best-1e-7: best=vl; wait=0; torch.save({'model_state_dict':model.state_dict(),'model_type':'classic_cnn','architecture':'ClassicCNNClassifier','num_classes':num_classes,'class_names':class_names,'epoch':ep,'best_val_loss':best,'model_name':model_name},os.path.join(label_path,'models',model_name+'.pth'))
        else: wait+=1
        if early_stop and wait>=early_stop: break
        await asyncio.sleep(0)
    training_state.is_training=False; training_state.training_completed=True; training_state.status='Training complete'; add_log('success',f'Classic CNN training complete: {training_state.epoch} epochs')

async def yolo_training_loop(epochs,batch_size,lr,label_path,safe_label,class_names,model_name,device,early_stop=0,**kwargs):
    from app import training_state,add_log
    # AGPL-3.0 gate - see deployment_config.require_yolo()
    try:
        import deployment_config as depcfg
        YOLO = depcfg.require_yolo()
    except Exception as e:
        raise RuntimeError("YOLO training is disabled in this deployment: " + str(e)) from e
    root=os.path.join(label_path,'dataset'); yaml=os.path.join(root,'yolo_dataset.yaml')
    names={i:n for i,n in enumerate([c for c in class_names if str(c).lower()!='background'])}
    with open(yaml,'w',encoding='utf-8') as f:
        f.write(f"path: {root.replace(chr(92),'/')}\ntrain: train/images\nval: val/images\ntest: test/images\nnames: {names}\n")
    # Ultralytics expects labels next to images. Mirror canonical labels.
    import shutil
    for split in ('train','val','test'):
        ld=os.path.join(root,split,'labels'); os.makedirs(ld,exist_ok=True); src=os.path.join(root,split,'yolo_labels')
        if os.path.isdir(src):
            for f in os.listdir(src): shutil.copy2(os.path.join(src,f),os.path.join(ld,f))
    model=YOLO('yolo11n.pt'); training_state.is_training=True; training_state.total_epochs=epochs
    await asyncio.to_thread(model.train,data=yaml,epochs=epochs,batch=batch_size,imgsz=640,lr0=lr,project=os.path.join(label_path,'models'),name=model_name,exist_ok=True,verbose=False)
    training_state.is_training=False; training_state.training_completed=True; training_state.status='Training complete'; add_log('success','YOLO training complete')
