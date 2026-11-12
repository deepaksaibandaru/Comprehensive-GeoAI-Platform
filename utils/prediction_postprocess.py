"""Geometry-safe prediction post-processing helpers. Never replaces an instance mask with its bbox."""
import numpy as np, cv2

def native_instance_mask_to_polygons(mask, min_area=3):
    a=(np.asarray(mask)>0).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(a,8)
    out=[]
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA] < min_area: continue
        yy,xx=np.where(lab==i); x1,y1,x2,y2=xx.min(),yy.min(),xx.max(),yy.max()
        contours,_=cv2.findContours((lab==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c=max(contours,key=cv2.contourArea); pts=c[:,0,:].astype(float).tolist()
            if len(pts)>=3: out.append({'polygon_pixels':pts,'bbox':[int(x1),int(y1),int(x2+1),int(y2+1)],'area':int(stats[i,cv2.CC_STAT_AREA])})
    return out

def boxes_only_to_features(boxes, scores=None, class_ids=None):
    feats=[]
    for i,b in enumerate(np.asarray(boxes)): 
        x1,y1,x2,y2=map(float,b); feats.append({'geometry_type':'box','bbox':[x1,y1,x2,y2],'score':float(scores[i]) if scores is not None and i<len(scores) else 1.0,'class_id':int(class_ids[i]) if class_ids is not None and i<len(class_ids) else 1})
    return feats
