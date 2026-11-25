import cv2, numpy as np
F='/home/claude/paper/figures/'
def runs(mask):
    # return [start,end) segments where mask False (content)
    segs=[];s=None
    for i,m in enumerate(mask):
        if not m and s is None: s=i
        if m and s is not None: segs.append((s,i)); s=None
    if s is not None: segs.append((s,len(mask)))
    return [x for x in segs if x[1]-x[0]>50]
def panels(path):
    im=cv2.imread(path); w=(im.min(axis=2)>235)
    rows=runs(w.mean(axis=1)>0.97); cols=runs(w.mean(axis=0)>0.97)
    return im,rows,cols
def cartoon(img):
    s=img.copy()
    for _ in range(2): s=cv2.bilateralFilter(s,9,60,60)
    Z=s.reshape(-1,3).astype(np.float32)
    _,lab,cen=cv2.kmeans(Z,10,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,1.0),3,cv2.KMEANS_PP_CENTERS)
    q=cen.astype(np.uint8)[lab.flatten()].reshape(s.shape)
    g=cv2.medianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),7)
    e=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,5)
    e=cv2.morphologyEx(e,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    return cv2.bitwise_and(q,q,mask=e)
for f in ['deocclusion.jpg','movies_deocclusion.jpg']:
    im,r,c=panels(F+f); print(f,r,c)
sel=[('movies_deocclusion.jpg',0),('movies_deocclusion.jpg',1),('movies_deocclusion.jpg',2),('deocclusion.jpg',0),('deocclusion.jpg',1)]
G=6; W=600; out=[]
for f,ri in sel:
    im,r,c=panels(F+f); y0,y1=r[ri]
    orig=im[y0:y1,c[0][0]:c[0][1]]; comp=im[y0:y1,c[3][0]:c[3][1]]
    tiles=[orig,comp,cartoon(comp)]
    h=int(round(orig.shape[0]*W/orig.shape[1]))
    tiles=[cv2.resize(t,(W,h),interpolation=cv2.INTER_AREA) for t in tiles]
    sep=np.full((h,G,3),255,np.uint8)
    out.append(np.hstack([tiles[0],sep,tiles[1],sep,tiles[2]]))
rows=[]
for i,o in enumerate(out):
    rows.append(o)
    if i<len(out)-1: rows.append(np.full((G,o.shape[1],3),255,np.uint8))
fig=np.vstack(rows)
cv2.imwrite(F+'cartoonization.jpg',fig,[cv2.IMWRITE_JPEG_QUALITY,90]); print(fig.shape)
