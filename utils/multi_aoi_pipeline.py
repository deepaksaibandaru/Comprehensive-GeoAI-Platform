"""Multi-AOI imagery capture and dataset assembly for BrainAI.

Clusters GeoJSON road annotations in a local metric CRS, buffers each cluster,
fetches map tiles at a fixed target zoom, creates paired chips/masks using the
existing DatasetBuilder, and assigns entire AOIs to train/val/test splits.
"""
from __future__ import annotations
import math, os, shutil, tempfile, uuid
from pathlib import Path
from typing import Any, Dict, List
import requests
from PIL import Image
from io import BytesIO
from shapely.geometry import shape, box, mapping
from shapely.ops import transform as shp_transform, unary_union
from pyproj import CRS, Transformer


def _cluster(features: List[dict], cluster_distance_m: float, buffer_m: float) -> List[dict]:
    geoms=[]
    for f in features:
        try:
            g=shape(f.get('geometry'))
            if not g.is_empty:
                if not g.is_valid: g=g.buffer(0)
                if not g.is_empty: geoms.append((g, f))
        except Exception:
            continue
    if not geoms: raise ValueError('No valid annotation geometries were provided.')
    bounds=[g.bounds for g,_ in geoms]
    lon0=sum((b[0]+b[2])/2 for b in bounds)/len(bounds)
    lat0=sum((b[1]+b[3])/2 for b in bounds)/len(bounds)
    metric=CRS.from_proj4(f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs')
    fwd=Transformer.from_crs('EPSG:4326',metric,always_xy=True).transform
    inv=Transformer.from_crs(metric,'EPSG:4326',always_xy=True).transform
    projected=[shp_transform(fwd,g) for g,_ in geoms]
    parent=list(range(len(projected)))
    def find(i):
        while parent[i] != i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b: parent[b]=a
    for i,a in enumerate(projected):
        ab=a.bounds
        for j in range(i+1,len(projected)):
            b=projected[j]; bb=b.bounds
            if ab[2]+cluster_distance_m < bb[0] or bb[2]+cluster_distance_m < ab[0] or ab[3]+cluster_distance_m < bb[1] or bb[3]+cluster_distance_m < ab[1]: continue
            if a.distance(b)<=cluster_distance_m: union(i,j)
    groups={}
    for i,g in enumerate(projected): groups.setdefault(find(i),[]).append(i)
    aois=[]
    for idxs in groups.values():
        combined=unary_union([projected[i] for i in idxs])
        if buffer_m: combined=combined.buffer(buffer_m)
        w,s,e,n=shp_transform(inv,combined).bounds
        # Filter source features to this cluster; preserve each original geometry.
        aois.append({'bounds':{'west':w,'south':s,'east':e,'north':n},
                     'features':[geoms[i][1] for i in idxs], 'annotation_count':len(idxs)})
    aois.sort(key=lambda a:(-a['bounds']['north'],a['bounds']['west']))
    return aois


def _zoom_for_target_gsd(lat: float, target_gsd_m: float, max_zoom: int, min_zoom: int = 1) -> dict:
    """Estimate the Web-Mercator tile zoom needed for target metres/pixel.

    This is map-pixel scale, not proof of native satellite sensor resolution.
    Imagery providers may resample tiles or have lower source detail.
    """
    if target_gsd_m <= 0:
        raise ValueError("target_gsd_m must be greater than zero")
    latitude_factor = max(0.01, math.cos(math.radians(max(-85.0, min(85.0, lat)))))
    zoom_float = math.log2(156543.03392804097 * latitude_factor / target_gsd_m)
    requested_zoom = int(math.ceil(zoom_float))
    chosen_zoom = max(min_zoom, min(max_zoom, requested_zoom))
    estimated_mpp = 156543.03392804097 * latitude_factor / (2 ** chosen_zoom)
    return {
        "zoom": chosen_zoom,
        "requested_zoom": requested_zoom,
        "estimated_web_mercator_m_per_pixel": estimated_mpp,
        "target_gsd_m": target_gsd_m,
        "tile_scale_meets_target": estimated_mpp <= target_gsd_m * 1.05,
        "zoom_capped": chosen_zoom < requested_zoom,
    }


def _tile_xy(lon, lat, z):
    n=2**z
    x=(lon+180.0)/360.0*n
    lat=max(-85.05112878,min(85.05112878,lat))
    r=math.radians(lat)
    y=(1.0-math.asinh(math.tan(r))/math.pi)/2.0*n
    return x,y

def _tile_lon(x,z): return x/(2**z)*360.0-180.0
def _tile_lat(y,z): return math.degrees(math.atan(math.sinh(math.pi*(1-2*y/(2**z)))))



def _tile_chunks(bounds, zoom, max_tiles):
    """Yield tile-aligned sub-bounds so large AOIs are captured in manageable blocks."""
    x0f,y0f=_tile_xy(bounds['west'],bounds['north'],zoom)
    x1f,y1f=_tile_xy(bounds['east'],bounds['south'],zoom)
    x0,x1=math.floor(x0f),math.floor(x1f); y0,y1=math.floor(y0f),math.floor(y1f)
    side=max(1,int(math.sqrt(max_tiles)))
    for yy in range(y0,y1+1,side):
        for xx in range(x0,x1+1,side):
            xe=min(x1+1,xx+side); ye=min(y1+1,yy+side)
            yield {'west':_tile_lon(xx,zoom),'east':_tile_lon(xe,zoom),
                   'north':_tile_lat(yy,zoom),'south':_tile_lat(ye,zoom),'crs':'EPSG:4326'}

def _capture(bounds: Dict[str,float], zoom: int, tile_url: str, dest: Path, max_tiles: int):
    x0f,y0f=_tile_xy(bounds['west'],bounds['north'],zoom)
    x1f,y1f=_tile_xy(bounds['east'],bounds['south'],zoom)
    x0,x1=math.floor(x0f),math.floor(x1f); y0,y1=math.floor(y0f),math.floor(y1f)
    count=(x1-x0+1)*(y1-y0+1)
    if count>max_tiles:
        raise ValueError(f'AOI requires {count} map tiles at zoom {zoom}, above safety limit {max_tiles}. Reduce AOI buffer/cluster size or raise max tiles per AOI.')
    tiles={}
    for x in range(x0,x1+1):
        for y in range(y0,y1+1):
            url=tile_url.replace('{s}','a').replace('{z}',str(zoom)).replace('{x}',str(x)).replace('{y}',str(y))
            try:
                r=requests.get(url,timeout=20,headers={'User-Agent':'BrainAI GeoAI dataset builder/1.0'})
                r.raise_for_status()
                with Image.open(BytesIO(r.content)) as im: tiles[(x,y)]=im.convert('RGB').copy()
            except Exception as e:
                raise RuntimeError(f'Failed to fetch imagery tile z={zoom}, x={x}, y={y}: {e}') from e
    width=(x1-x0+1)*256; height=(y1-y0+1)*256
    stitched=Image.new('RGB',(width,height))
    for (x,y),im in tiles.items(): stitched.paste(im,((x-x0)*256,(y-y0)*256))
    dest.parent.mkdir(parents=True,exist_ok=True); stitched.save(dest)
    return str(dest), {'west':_tile_lon(x0,zoom),'east':_tile_lon(x1+1,zoom),
                       'north':_tile_lat(y0,zoom),'south':_tile_lat(y1+1,zoom),
                       'crs':'EPSG:4326','zoom':zoom}, width,height,count


def _resample_image_to_target_gsd(image_path: str, latitude: float, zoom: int, target_gsd_m: float):
    """Resample fetched Web-Mercator tiles to the requested ground pixel scale.

    This estimates ground scale at the AOI-part centre. It makes the output grid
    target-sized; it cannot create native sensor detail absent from the provider.
    """
    if target_gsd_m <= 0:
        raise ValueError("target_gsd_m must be greater than zero")
    source_ground_mpp = 156543.03392804097 * max(0.01, math.cos(math.radians(max(-85.0, min(85.0, latitude))))) / (2 ** int(zoom))
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        old_w, old_h = source.size
        new_w = max(1, int(round(old_w * source_ground_mpp / target_gsd_m)))
        new_h = max(1, int(round(old_h * source_ground_mpp / target_gsd_m)))
        if (new_w, new_h) != (old_w, old_h):
            source = source.resize((new_w, new_h), Image.Resampling.LANCZOS)
            source.save(image_path)
    return {"estimated_source_ground_m_per_pixel": source_ground_mpp,
            "target_gsd_m": float(target_gsd_m),
            "resampled_width": new_w, "resampled_height": new_h,
            "resampled_to_target_grid": True}


def _features_for_bounds(features, bounds):
    clip=box(bounds['west'],bounds['south'],bounds['east'],bounds['north'])
    out=[]
    for f in features:
        try:
            g=shape(f.get('geometry'))
            if not g.is_valid: g=g.buffer(0)
            if not g.intersects(clip): continue
            cut=g.intersection(clip)
            if cut.is_empty: continue
            nf=dict(f); nf['geometry']=mapping(cut); out.append(nf)
        except Exception: continue
    return out


def build_multi_aoi_dataset(features: List[dict], tile_url: str, output_dir: str,
                            tile_size: int=256, overlap: float=.2, zoom: int=18,
                            cluster_distance_m: float=250, aoi_buffer_m: float=150,
                            class_names=None, label='Road', model_type='unet',
                            max_tiles_per_aoi: int=64, progress_callback=None,
                            target_gsd_m: float=0.30, max_zoom: int=19) -> dict:
    """Build all AOI chips into one dataset; each AOI belongs to one split only."""
    if not 0 <= overlap < 1: raise ValueError('overlap must be between 0 and 1.')
    aois=_cluster(features,cluster_distance_m,aoi_buffer_m)
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        for sub in ('images','masks','masks_vis','masks_overlay','class_labels','detection_targets','yolo_labels','instance_targets','instance_masks'):
            (root/split/sub).mkdir(parents=True,exist_ok=True)
    # Stable cluster-level splits: a geographic AOI never leaks across splits.
    n=len(aois); order=list(range(n))
    if n>=3:
        ntrain=max(1,int(n*.70)); nval=max(1,int(n*.15));
        while ntrain+nval>=n: 
            if ntrain>1: ntrain-=1
            else: nval-=1
        split_by_idx={i:('train' if rank<ntrain else 'val' if rank<ntrain+nval else 'test') for rank,i in enumerate(order)}
    elif n==2: split_by_idx={0:'train',1:'test'}
    else: split_by_idx={0:'train'}
    from utils.dataset_builder import DatasetBuilder
    classes=class_names or ['background',label]
    total=0; aoi_summaries=[]; errors=[]; split_counts={'train':0,'val':0,'test':0}; run_id=uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix='brainai_multi_aoi_') as td:
        for i,aoi in enumerate(aois):
            center_lat = (aoi['bounds']['south'] + aoi['bounds']['north']) / 2.0
            zoom_info = _zoom_for_target_gsd(center_lat, float(target_gsd_m), int(max_zoom))
            aoi_zoom = zoom_info['zoom']
            target_split=split_by_idx[i]
            aoi_chip_total=0; aoi_tile_total=0; aoi_width=0; aoi_height=0; part_count=0
            try:
                for part_idx, part_bounds in enumerate(_tile_chunks(aoi['bounds'],aoi_zoom,max_tiles_per_aoi)):
                    image_path, snapped, w,h,tile_count=_capture(part_bounds,aoi_zoom,tile_url,Path(td)/f'aoi_{i:04d}_part_{part_idx:04d}.png',max_tiles_per_aoi)
                    part_center_lat = (snapped['south'] + snapped['north']) / 2.0
                    resample_info = _resample_image_to_target_gsd(image_path, part_center_lat, aoi_zoom, float(target_gsd_m))
                    with Image.open(image_path) as resized_image:
                        w, h = resized_image.size
                    local_features=_features_for_bounds(aoi['features'],snapped)
                    if not local_features: continue
                    temp_root=Path(td)/f'build_{i:04d}_{part_idx:04d}'; temp_root.mkdir()
                    builder=DatasetBuilder(source_image_path=image_path,features=local_features,class_names=classes,
                        tile_size=tile_size,overlap=overlap,bounds=snapped,label=label,
                        model_type=model_type,task_type='semantic_segmentation' if model_type=='unet' else None)
                    dataset_dir, metadata, chip_stats=builder.build_dataset(output_dir=str(temp_root))
                    part_chip_count=0
                    for src_split in ('train','val','test'):
                        src_base=Path(dataset_dir)/src_split
                        if not src_base.exists(): continue
                        for subdir in src_base.iterdir():
                            if not subdir.is_dir(): continue
                            target_dir=root/target_split/subdir.name; target_dir.mkdir(parents=True,exist_ok=True)
                            for file in subdir.iterdir():
                                if not file.is_file(): continue
                                target_name=f'run{run_id}_aoi{i:04d}_part{part_idx:04d}_{file.name}'
                                shutil.copy2(file,target_dir/target_name)
                                if subdir.name=='images': part_chip_count+=1
                    # Only count image/mask pairs as usable chips.
                    pref=f'run{run_id}_aoi{i:04d}_part{part_idx:04d}_'
                    imgs=len(list((root/target_split/'images').glob(pref+'*')))
                    masks=len(list((root/target_split/'masks').glob(pref+'*')))
                    aoi_chip_total += min(imgs,masks)
                    aoi_tile_total += tile_count; aoi_width=max(aoi_width,w); aoi_height=max(aoi_height,h); part_count+=1
                if aoi_chip_total == 0:
                    raise ValueError('No aligned image/mask chips were generated for this AOI. Check imagery coverage, label geometry, and chip size.')
                total+=aoi_chip_total; split_counts[target_split]+=aoi_chip_total
                aoi_summaries.append({'bounds':aoi['bounds'],'annotation_count':aoi['annotation_count'],
                    'zoom':aoi_zoom, 'requested_zoom':zoom_info['requested_zoom'],
                    'target_gsd_m':float(target_gsd_m),
                    'estimated_web_mercator_m_per_pixel':zoom_info['estimated_web_mercator_m_per_pixel'],
                    'tile_scale_meets_target':zoom_info['tile_scale_meets_target'],
                    'zoom_capped':zoom_info['zoom_capped'],
                    'estimated_source_ground_m_per_pixel':resample_info['estimated_source_ground_m_per_pixel'],
                    'resampled_to_target_grid':resample_info['resampled_to_target_grid'],
                    'tile_count':aoi_tile_total,'capture_parts':part_count,'split':target_split,
                    'chips':aoi_chip_total,'max_part_width':aoi_width,'max_part_height':aoi_height})
            except Exception as e:
                errors.append({'aoi_bounds':aoi['bounds'],'error':str(e)})
    if total==0:
        raise RuntimeError('No chips were generated from any AOI. ' + '; '.join(e['error'] for e in errors[:3]))
    import json
    meta={'aoi_count':len(aois),'processed_aoi_count':len(aoi_summaries),'failed_aoi_count':len(errors),
          'total_chips':total,'chips_per_split':split_counts,'target_zoom_requested':zoom,
          'target_gsd_m':float(target_gsd_m),'service_max_zoom':int(max_zoom), 'tile_size':tile_size,
          'overlap':overlap,'cluster_distance_m':cluster_distance_m,'aoi_buffer_m':aoi_buffer_m,
          'split_strategy':'geographic AOI-level split','aois':aoi_summaries,'errors':errors}
    (root/'multi_aoi_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta
