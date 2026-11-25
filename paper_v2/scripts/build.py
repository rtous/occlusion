exec(open('cartoon2.py').read())
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
cv2.imwrite(F+'cartoonization.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,90])
