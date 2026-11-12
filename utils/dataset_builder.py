"""GeoAI dataset builder: native-pixel image chips and model-specific targets.

The source image is never resized during chip creation. Image and annotation
pixels share one geographic-to-pixel transform; semantic, detection, instance,
classification and YOLO targets are written together for model compatibility.
"""
from __future__ import annotations
import os, json, random, base64
from io import BytesIO
from datetime import datetime
import cv2
import numpy as np
from PIL import Image

SEMANTIC_MODEL_TYPES = {"unet", "classic_cnn"}
DETECTION_MODEL_TYPES = {"rcnn", "fast_rcnn", "resnet_detector"}
YOLO_MODEL_TYPES = {"yolo"}
INSTANCE_MODEL_TYPES = {"mask_rcnn"}
ALL_MODEL_TYPES = SEMANTIC_MODEL_TYPES | DETECTION_MODEL_TYPES | INSTANCE_MODEL_TYPES | YOLO_MODEL_TYPES


def convert_to_serializable(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, dict): return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convert_to_serializable(v) for v in obj]
    return obj


def normalize_model_type(model_type):
    name = str(model_type or "unet").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"u_net":"unet", "cnn":"classic_cnn", "classic":"classic_cnn", "faster_rcnn":"rcnn",
               "faster_r_cnn":"rcnn", "fast_r_cnn":"fast_rcnn", "maskrcnn":"mask_rcnn", "mask_r_cnn":"mask_rcnn",
               "yolov8":"yolo", "yolov9":"yolo", "yolov10":"yolo", "yolov11":"yolo", "yolov12":"yolo"}
    name = aliases.get(name, name)
    return name if name in ALL_MODEL_TYPES else "unet"


def get_model_target_contract(model_type):
    m = normalize_model_type(model_type)
    if m == "classic_cnn":
        return {"model_type":m,"model_family":"classification","primary_target":"class_label","target_path":"<split>/class_labels/<chip>.json","fallback_target_path":"<split>/masks/<chip>.png","training_api":"model(images) -> class_logits","backend":"native ClassicCNNClassifier"}
    if m in SEMANTIC_MODEL_TYPES:
        return {"model_type":m,"model_family":"semantic_segmentation","primary_target":"semantic_mask","target_path":"<split>/masks/<chip>.png","training_api":"model(images) -> logits"}
    if m in DETECTION_MODEL_TYPES:
        return {"model_type":m,"model_family":"object_detection","primary_target":"boxes_labels","target_path":"<split>/detection_targets/<chip>.json","fallback_target_path":"<split>/instance_targets/<chip>.json","training_api":"model(images, targets) -> loss_dict","backend":"torchvision Faster R-CNN"}
    if m == "yolo":
        return {"model_type":m,"model_family":"object_detection","primary_target":"boxes_labels","target_path":"<split>/yolo_labels/<chip>.txt","training_api":"YOLO train(data=...) -> detector","backend":"Ultralytics YOLO"}
    return {"model_type":m,"model_family":"instance_segmentation","primary_target":"boxes_labels_masks","target_path":"<split>/instance_targets/<chip>.json","mask_path":"<split>/instance_masks/<chip_stem>/instance_XXXX.png","training_api":"model(images, targets) -> loss_dict","backend":"torchvision Mask R-CNN"}


def _key(v): return str(v or "").strip().lower().replace(" ", "_").replace("-", "_")


class DatasetBuilder:
    def __init__(self, source_image_path=None, annotation_path=None, geojson_path=None, features=None,
                 class_names=None, tile_size=512, overlap=.2, bounds=None, label="object",
                 shape_type="rectangle", model_type="unet", task_type=None, chip_prefix="chip_"):
        self.source_image_path = source_image_path
        self.chip_prefix = str(chip_prefix or "chip_")
        self.tile_size = int(tile_size)
        self.overlap = min(max(float(overlap), 0.0), .9)
        self.label, self.shape_type, self.task_type = label, shape_type, task_type or ""
        self.model_type = normalize_model_type(model_type)
        self.model_target_contract = get_model_target_contract(self.model_type)
        self.chip_images = []
        if not source_image_path or not os.path.exists(source_image_path):
            raise ValueError(f"Source image not found: {source_image_path}")
        self.image = Image.open(source_image_path).convert("RGB")
        self.img_width, self.img_height = self.image.size
        if features is not None: self.features = list(features)
        elif annotation_path and os.path.exists(annotation_path): self.features = self._load_features_from_file(annotation_path)
        elif geojson_path and os.path.exists(geojson_path): self.features = self._load_features_from_file(geojson_path)
        else: self.features = []
        self.annotation_path = annotation_path or geojson_path
        self.bounds = self._normalize_bounds(bounds) if bounds else self._get_bounds_from_features()
        self.is_geo = self._is_geo_bounds()
        self.class_names = list(class_names) if class_names else self._extract_class_names()
        deduped, seen = [], set()
        for name in self.class_names:
            if _key(name) not in seen: seen.add(_key(name)); deduped.append(str(name))
        self.class_names = deduped
        if not any(_key(x) == "background" for x in self.class_names): self.class_names.insert(0, "background")
        else:
            i = next(i for i,x in enumerate(self.class_names) if _key(x)=="background")
            self.class_names.insert(0, self.class_names.pop(i))
        if len(self.class_names) < 2: self.class_names = ["background", str(label)]
        self.num_classes = len(self.class_names)
        self.class_to_id = {}
        for i, name in enumerate(self.class_names):
            for alias in {str(name), str(name).lower(), str(name).lower().replace(" ", "_"), str(name).lower().replace("_", " "), _key(name)}:
                self.class_to_id[alias] = i
        self._resolved_class_distribution = {}
        for f in self.features:
            cid = self._feature_class_id_safe(f, f.get("properties") or {})
            if cid is not None: self._resolved_class_distribution[cid] = self._resolved_class_distribution.get(cid,0)+1
        print(f"📷 Source image: {self.img_width}x{self.img_height}; features={len(self.features)}; model={self.model_type}; bounds={self.bounds}")

    def _load_features_from_file(self, fp):
        try:
            data = json.loads(open(fp, "r", encoding="utf-8").read())
            return data.get("features", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        except Exception as e:
            print(f"⚠️ Annotation load error: {e}"); return []

    def _extract_class_names(self):
        names, seen = ["background"], {"background"}
        for f in self.features:
            p = f.get("properties") or {}
            raw = next((p.get(k) for k in ("class_name","label","category","class") if p.get(k)), None)
            if raw is None and p.get("class_id") is not None: raw = f"class_{p['class_id']}"
            if raw and _key(raw) not in seen: names.append(str(raw)); seen.add(_key(raw))
        return names

    def _normalize_bounds(self, b):
        if isinstance(b, dict):
            return {"west":float(b.get("west",0)),"south":float(b.get("south",0)),"east":float(b.get("east",self.img_width)),"north":float(b.get("north",self.img_height)),"crs":str(b.get("crs","EPSG:4326"))}
        if isinstance(b,(list,tuple)) and len(b)==4: return {"west":float(b[0]),"south":float(b[1]),"east":float(b[2]),"north":float(b[3]),"crs":"EPSG:4326"}
        return {"west":0.,"south":0.,"east":float(self.img_width),"north":float(self.img_height),"crs":"EPSG:4326"}

    def _get_bounds_from_features(self):
        coords=[]
        def walk(v):
            if isinstance(v,(list,tuple)) and len(v)>=2 and all(isinstance(x,(int,float)) for x in v[:2]): coords.append(v[:2])
            elif isinstance(v,(list,tuple)):
                for x in v: walk(x)
        for f in self.features:
            g=f.get("geometry") or {}
            if g.get("type") in ("Polygon","MultiPolygon","LineString","MultiLineString","Point","MultiPoint","GeometryCollection"):
                walk(g.get("coordinates", []))
        if not coords: return {"west":0.,"south":0.,"east":float(self.img_width),"north":float(self.img_height),"crs":"EPSG:4326"}
        geo=abs(float(coords[0][0]))<=180 and abs(float(coords[0][1]))<=90
        return {"west":min(float(c[0]) for c in coords),"south":min(float(c[1]) for c in coords),"east":max(float(c[0]) for c in coords),"north":max(float(c[1]) for c in coords),"crs":"EPSG:4326" if geo else "PIXEL"}

    def _is_geo_bounds(self):
        b=self.bounds
        return b.get("crs") != "PIXEL" and abs(float(b["west"]))<=180 and abs(float(b["east"]))<=180 and abs(float(b["south"]))<=90 and abs(float(b["north"]))<=90 and b["east"]>b["west"] and b["north"]>b["south"]

    def _geo_to_pixel(self, lon, lat):
        b=self.bounds
        if self.is_geo:
            x=(float(lon)-b["west"])/(b["east"]-b["west"])*self.img_width
            y=(b["north"]-float(lat))/(b["north"]-b["south"])*self.img_height
        else: x,y=float(lon),float(lat)
        # Do not clamp: off-chip/out-of-image geometry must not be dragged to an edge.
        return int(round(x)), int(round(y))

    def _pixel_to_geo(self,x,y):
        b=self.bounds
        if self.is_geo:
            return (b["west"]+float(x)/self.img_width*(b["east"]-b["west"]), b["north"]-float(y)/self.img_height*(b["north"]-b["south"]))
        return float(x),float(y)

    def _feature_class_id_safe(self, feature, props=None):
        p=props if props is not None else feature.get("properties") or {}
        raw=p.get("class_id")
        if isinstance(raw,(int,float)) and 0<=int(raw)<self.num_classes: return int(raw)
        for k in ("class_name","label","category","class"):
            val=p.get(k)
            if val:
                for alias in {str(val),str(val).lower(),str(val).lower().replace(" ","_"),str(val).lower().replace("_"," "),_key(val)}:
                    if alias in self.class_to_id: return self.class_to_id[alias]
        return None

    def _feature_class_id(self, feature, strict=True):
        cid=self._feature_class_id_safe(feature)
        if cid is None: cid=1
        return max(1,min(self.num_classes-1,int(cid)))

    def _geometry_pixels(self, geometry, chip_x=0, chip_y=0):
        """Transform GeoJSON coordinates to image pixels, then local chip pixels."""
        def ring(coords): return np.asarray([[self._geo_to_pixel(p[0],p[1])[0]-chip_x,self._geo_to_pixel(p[0],p[1])[1]-chip_y] for p in coords],dtype=np.int32)
        return ring

    def _feature_to_chip_mask(self, feature, chip_x, chip_y, chip_size):
        mask=np.zeros((chip_size,chip_size),dtype=np.uint8); g=feature.get("geometry") or {}; typ=g.get("type"); c=g.get("coordinates",[]); cid=self._feature_class_id(feature)
        ring=self._geometry_pixels(g,chip_x,chip_y)
        def draw_poly(poly):
            if not poly:return
            pts=ring(poly[0])
            if len(pts)>=3: cv2.fillPoly(mask,[pts.reshape((-1,1,2))],cid)
            for hole in poly[1:]:
                hp=ring(hole)
                if len(hp)>=3: cv2.fillPoly(mask,[hp.reshape((-1,1,2))],0)
        try:
            if typ=="Polygon": draw_poly(c)
            elif typ=="MultiPolygon":
                for poly in c: draw_poly(poly)
            elif typ in ("LineString","MultiLineString"):
                lines=[c] if typ=="LineString" else c
                for line in lines:
                    pts=ring(line)
                    if len(pts)>=2: cv2.polylines(mask,[pts.reshape((-1,1,2))],False,cid,thickness=3)
            elif typ=="Point":
                px,py=self._geo_to_pixel(*c); cv2.circle(mask,(px-chip_x,py-chip_y),5,cid,-1)
            elif typ=="MultiPoint":
                for p in c:
                    px,py=self._geo_to_pixel(*p); cv2.circle(mask,(px-chip_x,py-chip_y),5,cid,-1)
        except Exception as e: print(f"⚠️ Feature rasterization failed: {e}")
        return mask

    def _create_full_mask(self):
        mask=np.zeros((self.img_height,self.img_width),dtype=np.uint8)
        for f in self.features:
            cm=self._feature_to_chip_mask(f,0,0,max(self.img_width,self.img_height))
            # Feature helper's mask is square; clip safely into full image.
            h=min(self.img_height,cm.shape[0]); w=min(self.img_width,cm.shape[1])
            if h and w: mask[:h,:w]=np.where(cm[:h,:w]>0,cm[:h,:w],mask[:h,:w])
        return mask

    def _create_chip_instances(self, chip_x, chip_y, chip_size):
        instances=[]
        for i,f in enumerate(self.features):
            m=self._feature_to_chip_mask(f,chip_x,chip_y,chip_size); ys,xs=np.where(m>0)
            if len(xs)<2: continue
            x1,y1,x2,y2=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
            cid=self._feature_class_id(f); props=f.get("properties") or {}
            instances.append({"instance_id":i,"class_id":cid,"class_name":self.class_names[cid],"bbox":[x1,y1,x2,y2],"area":int((m>0).sum()),"bbox_area":(x2-x1)*(y2-y1),"iscrowd":0,"mask":m,"properties":props})
        return instances

    def _get_chip_geo_bounds(self,x,y,width,height):
        a=self._pixel_to_geo(x,y); b=self._pixel_to_geo(x+width,y+height)
        return {"west":float(min(a[0],b[0])),"south":float(min(a[1],b[1])),"east":float(max(a[0],b[0])),"north":float(max(a[1],b[1]))}

    def _mask_to_color_visualization(self,mask):
        rgb=np.zeros((*mask.shape,3),dtype=np.uint8); rgb[mask>0]=(255,255,255); return rgb
    def _save_mask_raw(self,mask,path): Image.fromarray(np.where(mask>0,255,0).astype(np.uint8),mode="L").save(path)
    def _save_mask_visual(self,mask,path): Image.fromarray(self._mask_to_color_visualization(mask),mode="RGB").save(path)
    def _save_mask_overlay(self,image,mask,path,alpha=.55):
        overlay=image.astype(np.float32).copy(); fg=mask>0; overlay[fg]=overlay[fg]*(1-alpha)+np.array([255,40,40])*alpha; Image.fromarray(np.clip(overlay,0,255).astype(np.uint8),mode="RGB").save(path)
    def _image_to_base64(self,array):
        b=BytesIO(); Image.fromarray(array.astype(np.uint8)).save(b,format="PNG"); return base64.b64encode(b.getvalue()).decode()

    def _generate_chips(self):
        tile=self.tile_size; step=max(1,int(tile*(1-self.overlap))); image=np.asarray(self.image); chips=[]
        # Generate only complete native-pixel chips; no resizing. Pad only when source is smaller than a chip.
        ph=max(0,tile-self.img_height); pw=max(0,tile-self.img_width)
        if ph or pw:
            image=np.pad(image,((0,ph),(0,pw),(0,0)),mode="reflect" if min(self.img_width,self.img_height)>1 else "edge")
        full_mask=np.zeros(image.shape[:2],dtype=np.uint8)
        for y in range(0,max(1,image.shape[0]-tile+1),step):
            for x in range(0,max(1,image.shape[1]-tile+1),step):
                crop=image[y:y+tile,x:x+tile]
                if crop.shape[:2]!=(tile,tile): continue
                m=np.zeros((tile,tile),dtype=np.uint8)
                instances=self._create_chip_instances(x,y,tile)
                # Combine feature masks into the semantic mask using the same local transform.
                for inst in instances: m=np.where(inst["mask"]>0,inst["mask"],m).astype(np.uint8)
                has=bool(np.any(m>0))
                if not has and random.random()>.45: continue
                chips.append({"name":f"{self.chip_prefix}{len(chips):06d}.png","image":crop,"mask":m,"x":x,"y":y,"has_foreground":has,"instances":instances,"geo_bounds":self._get_chip_geo_bounds(x,y,tile,tile)})
        # Ensure final edge coverage if step doesn't land on the last valid window.
        if image.shape[1]>=tile and image.shape[0]>=tile:
            xs=max(0,image.shape[1]-tile); ys=max(0,image.shape[0]-tile)
            for y,x in ((0,xs),(ys,0),(ys,xs)):
                if any(c["x"]==x and c["y"]==y for c in chips): continue
                crop=image[y:y+tile,x:x+tile]; instances=self._create_chip_instances(x,y,tile); m=np.zeros((tile,tile),dtype=np.uint8)
                for inst in instances: m=np.where(inst["mask"]>0,inst["mask"],m).astype(np.uint8)
                has=bool(np.any(m>0))
                if has or random.random()<=.45: chips.append({"name":f"{self.chip_prefix}{len(chips):06d}.png","image":crop,"mask":m,"x":x,"y":y,"has_foreground":has,"instances":instances,"geo_bounds":self._get_chip_geo_bounds(x,y,tile,tile)})
        self.chip_images=[]
        previews=[c for c in chips if c["has_foreground"]][:6]+[c for c in chips if not c["has_foreground"]][:2]
        for c in previews:
            self.chip_images.append({"image":self._image_to_base64(c["image"]),"mask":self._image_to_base64(self._mask_to_color_visualization(c["mask"])),"name":c["name"].replace(".png",""),"has_foreground":c["has_foreground"],"instance_count":len(c["instances"])})
        print(f"📊 Generated {len(chips)} chips ({sum(c['has_foreground'] for c in chips)} positive)")
        return chips

    def build_dataset(self, output_dir=None, split_ratios=None, append=False):
        split_ratios=split_ratios or {"train":.70,"val":.15,"test":.15}
        output_dir=output_dir or os.path.dirname(self.source_image_path) or "."
        dataset_dir=os.path.join(output_dir,"dataset"); chips_dir=os.path.join(output_dir,"chips")
        for split in ("train","val","test"):
            for sub in ("images","masks","masks_vis","masks_overlay","class_labels","yolo_labels","instance_masks","instance_targets","detection_targets"):
                os.makedirs(os.path.join(dataset_dir,split,sub),exist_ok=True)
        for sub in ("images","masks"): os.makedirs(os.path.join(chips_dir,sub),exist_ok=True)
        chips=self._generate_chips()
        if not chips: raise RuntimeError("Dataset builder generated zero chips.")
        random.seed(42); random.shuffle(chips); n=len(chips)
        train_n=max(1,min(n,int(n*split_ratios.get("train",.7))))
        val_n=max(1,int(n*split_ratios.get("val",.15))) if n>3 else 0
        if train_n+val_n>n: val_n=max(0,n-train_n)
        split_chips={"train":chips[:train_n],"val":chips[train_n:train_n+val_n],"test":chips[train_n+val_n:]}
        
        # If appending to existing dataset, load existing bounds_map and stats
        bounds_map = {}
        if append and os.path.exists(os.path.join(dataset_dir, "chip_bounds.json")):
            try:
                with open(os.path.join(dataset_dir, "chip_bounds.json"), "r", encoding="utf-8") as f:
                    bounds_map = json.load(f)
            except Exception: bounds_map = {}
            
        stats={}
        for split, data in split_chips.items():
            stats[split]={"total":len(data),"positive":sum(c["has_foreground"] for c in data),"negative":sum(not c["has_foreground"] for c in data),"instances":sum(len(c["instances"]) for c in data)}
            for chip in data:
                name=chip["name"]; stem=os.path.splitext(name)[0]; base=os.path.join(dataset_dir,split)
                gb=chip["geo_bounds"]; bounds_map[name]={**gb,"x":chip["x"],"y":chip["y"],"px":chip["x"],"py":chip["y"],"width":self.tile_size,"height":self.tile_size,"split":split,"has_foreground":chip["has_foreground"],"instance_count":len(chip["instances"])}
                Image.fromarray(chip["image"]).save(os.path.join(base,"images",name))
                self._save_mask_raw(chip["mask"],os.path.join(base,"masks",name)); self._save_mask_visual(chip["mask"],os.path.join(base,"masks_vis",stem+"_vis.png")); self._save_mask_overlay(chip["image"],chip["mask"],os.path.join(base,"masks_overlay",stem+"_ov.png"))
                fg,counts=np.unique(chip["mask"][chip["mask"]>0],return_counts=True); cid=int(fg[np.argmax(counts)]) if len(fg) else 0
                cl={"image":name,"image_id":stem,"primary_class_id":cid,"primary_class_name":self.class_names[cid],"class_ids":[0] if cid==0 else [0,cid],"class_names":[self.class_names[i] for i in ([0] if cid==0 else [0,cid])],"multi_label":False,"target_type":"class_label"}
                json.dump(cl,open(os.path.join(base,"class_labels",stem+".json"),"w",encoding="utf-8"),indent=2)
                inst_targets=[]
                inst_dir=os.path.join(base,"instance_masks",stem); os.makedirs(inst_dir,exist_ok=True)
                for ii,inst in enumerate(chip["instances"]):
                    fn=f"instance_{ii:04d}.png"; Image.fromarray((inst["mask"]>0).astype(np.uint8)*255,mode="L").save(os.path.join(inst_dir,fn))
                    inst_targets.append({"instance_index":ii,"source_feature_id":int(inst["instance_id"]),"class_id":int(inst["class_id"]),"class_name":inst["class_name"],"bbox":inst["bbox"],"area":inst["area"],"iscrowd":0,"mask_file":os.path.join("instance_masks",stem,fn)})
                target={"image":name,"image_id":stem,"width":self.tile_size,"height":self.tile_size,"num_classes":self.num_classes,"class_names":self.class_names,"instances":inst_targets}
                json.dump(convert_to_serializable(target),open(os.path.join(base,"instance_targets",stem+".json"),"w",encoding="utf-8"),indent=2)
                det={"image":name,"image_id":stem,"width":self.tile_size,"height":self.tile_size,"num_classes":self.num_classes,"class_names":self.class_names,"boxes":[i["bbox"] for i in chip["instances"]],"labels":[i["class_id"] for i in chip["instances"]],"instances":[{"instance_index":j,"source_feature_id":i["instance_id"],"class_id":i["class_id"],"class_name":i["class_name"],"bbox":i["bbox"],"area":i["area"],"bbox_area":i["bbox_area"],"iscrowd":0} for j,i in enumerate(chip["instances"])],"target_type":"boxes_labels","source_instance_target":os.path.join("instance_targets",stem+".json")}
                json.dump(convert_to_serializable(det),open(os.path.join(base,"detection_targets",stem+".json"),"w",encoding="utf-8"),indent=2)
                lines=[]
                for i in chip["instances"]:
                    x1,y1,x2,y2=map(float,i["bbox"]); bw,bh=x2-x1,y2-y1
                    if bw<=0 or bh<=0: continue
                    lines.append(f"{max(0,int(i['class_id'])-1)} {(x1+x2)/2/self.tile_size:.8f} {(y1+y2)/2/self.tile_size:.8f} {bw/self.tile_size:.8f} {bh/self.tile_size:.8f}")
                open(os.path.join(base,"yolo_labels",stem+".txt"),"w",encoding="utf-8").write("\n".join(lines))
                Image.fromarray(chip["image"]).save(os.path.join(chips_dir,"images",name)); self._save_mask_raw(chip["mask"],os.path.join(chips_dir,"masks",name))
        json.dump(bounds_map,open(os.path.join(dataset_dir,"chip_bounds.json"),"w",encoding="utf-8"),indent=2)
        class_id_map={str(i):str(v) for i,v in enumerate(self.class_names)}
        contract=self.model_target_contract

        # Load and accumulate previous metadata if appending
        prev_metadata = {}
        if append and os.path.exists(os.path.join(dataset_dir, "metadata.json")):
            try:
                with open(os.path.join(dataset_dir, "metadata.json"), "r", encoding="utf-8") as f:
                    prev_metadata = json.load(f)
            except Exception: prev_metadata = {}
            
        combined_stats = dict(stats)
        if prev_metadata.get("split_stats"):
            prev_s = prev_metadata["split_stats"]
            for s in ("train", "val", "test"):
                if s in prev_s and s in combined_stats:
                    for k in ("total", "positive", "negative", "instances"):
                        combined_stats[s][k] = combined_stats[s].get(k, 0) + prev_s[s].get(k, 0)

        total_chips_count = n + prev_metadata.get("total_chips", 0) if append else n
        all_previews = (prev_metadata.get("chip_previews", []) + self.chip_images)[:10] if append else self.chip_images

        metadata={"source_image":str(self.source_image_path),"annotation_file":self.annotation_path,"class_names":self.class_names,"num_classes":self.num_classes,"class_id_map":class_id_map,"resolved_class_distribution":self._resolved_class_distribution,"tile_size":self.tile_size,"overlap":self.overlap,"image_size":[self.img_width,self.img_height],"bounds":self.bounds,"is_geo":self.is_geo,"created":datetime.now().isoformat(),"total_chips":total_chips_count,"num_features":len(self.features),"split_stats":combined_stats,"selected_model_type":self.model_type,"model_family":contract["model_family"],"primary_target":contract["primary_target"],"model_target_contract":contract,"supported_model_types":sorted(ALL_MODEL_TYPES),"target_assets":{"semantic":"<split>/masks/<chip>.png","detection":"<split>/detection_targets/<chip>.json","instance":"<split>/instance_targets/<chip>.json","instance_masks":"<split>/instance_masks/<chip_stem>/instance_XXXX.png"},"chip_bounds_file":"chip_bounds.json","instance_target_format":"torchvision_mask_rcnn","instance_target_version":2,"instance_masks_shape":[self.tile_size,self.tile_size],"semantic_mask_format":"binary_0_255","pixel_mapping":"img_width/img_height; native pixels; no resize during chip creation","no_synthetic_fallback":True,"target_manifest_file":"target_manifest.json","chip_previews":all_previews}
        manifest={"dataset_version":4,"selected_model_type":self.model_type,"model_family":contract["model_family"],"primary_target":contract["primary_target"],"contract":contract,"num_classes":self.num_classes,"class_names":self.class_names,"class_id_map":class_id_map,"splits":{}}
        
        # Load previous manifest if appending
        prev_manifest = {}
        if append and os.path.exists(os.path.join(dataset_dir, "target_manifest.json")):
            try:
                with open(os.path.join(dataset_dir, "target_manifest.json"), "r", encoding="utf-8") as f:
                    prev_manifest = json.load(f)
            except Exception: prev_manifest = {}

        for split,data in split_chips.items():
            prev_sp = prev_manifest.get("splits", {}).get(split, {}) if append else {}
            manifest["splits"][split]={
                "image_count": len(data) + prev_sp.get("image_count", 0),
                "positive_count": sum(c["has_foreground"] for c in data) + prev_sp.get("positive_count", 0),
                "instance_count": sum(len(c["instances"]) for c in data) + prev_sp.get("instance_count", 0),
                "class_id_counts": {str(cid): sum(1 for c in data for i in c["instances"] if i["class_id"]==cid) + prev_sp.get("class_id_counts", {}).get(str(cid), 0) for cid in range(self.num_classes)},
                "image_dir": f"{split}/images",
                "semantic_dir": f"{split}/masks",
                "detection_dir": f"{split}/detection_targets",
                "instance_target_dir": f"{split}/instance_targets",
                "instance_mask_dir": f"{split}/instance_masks"
            }
        json.dump(convert_to_serializable(manifest),open(os.path.join(dataset_dir,"target_manifest.json"),"w",encoding="utf-8"),indent=2)
        json.dump(convert_to_serializable(metadata),open(os.path.join(dataset_dir,"metadata.json"),"w",encoding="utf-8"),indent=2)
        json.dump(convert_to_serializable(metadata),open(os.path.join(chips_dir,"metadata.json"),"w",encoding="utf-8"),indent=2)
        print(f"✅ Dataset built: {total_chips_count} chips ({n} in current batch); split stats={combined_stats}; targets={contract['primary_target']}")
        return dataset_dir, metadata, combined_stats

    def get_chip_previews(self): return self.chip_images
