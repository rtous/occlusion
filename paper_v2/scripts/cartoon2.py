import cv2, numpy as np
exec(open('cartoon.py').read().split('def cartoon')[0])
def cartoon(img,k=24):
    s=img.copy()
    for _ in range(3): s=cv2.bilateralFilter(s,7,45,5)
    lab=cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
    Z=lab.reshape(-1,3).astype(np.float32)
    _,l,cen=cv2.kmeans(Z,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.5),3,cv2.KMEANS_PP_CENTERS)
    q=cv2.cvtColor(cen.astype(np.uint8)[l.flatten()].reshape(s.shape),cv2.COLOR_LAB2BGR)
    q=cv2.medianBlur(q,3)
    g=cv2.cvtColor(s,cv2.COLOR_BGR2GRAY)
    e=cv2.Canny(cv2.GaussianBlur(g,(3,3),0),50,130)
    n,lbl,st,_=cv2.connectedComponentsWithStats(e,8)
    keep=np.zeros_like(e)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]>=20: keep[lbl==i]=255
        q[keep>0]=(25,25,25)
    return q
