import cv2, numpy as np, json
# Intra-oral synthetic: gum background, 6 teeth with realistic apparent widths (RED 0.7-ish but imperfect), tilted 6 deg
W,H=1200,800
img=np.full((H,W,3),(120,110,200),np.uint8)  # BGR pinkish gum
cv2.rectangle(img,(0,560),(W,H),(60,40,90),-1)  # dark mouth below
mid=600; c=118; widths=[c*0.52,c*0.74,c,c*0.97,c*0.70,c*0.50]  # 13 12 11 21 22 23 (asymmetric on purpose)
lens=[140,122,150,148,124,138]; zen=[300,312,300,302,313,301]
xs=[]; x=mid
for w in widths[2::-1]: xs.insert(0,(x-w,x)); x-=w
x=mid
for w in widths[3:]: xs.append((x,x+w)); x+=w
truth={}
for (x0,x1),L,z,t in zip(xs,lens,zen,['13','12','11','21','22','23']):
    g=4
    k=['13','12','11','21','22','23'].index(t)+1
    cv2.rectangle(img,(int(x0+g),int(z)),(int(x1-g),int(z+L)),(196+9*k,235,240),-1)
    truth[t]=[x0+g,x1-g,z,z+L]
M=cv2.getRotationMatrix2D((W/2,H/2),-6,1.0)  # rotate content 6 deg clockwise
img=cv2.warpAffine(img,M,(W,H),borderValue=(60,40,90))
img=cv2.GaussianBlur(img,(3,3),0)
cv2.imwrite('fixtures/intra.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,95])
json.dump(truth,open('fixtures/intra_truth.json','w'))
# Face synthetic
F=np.full((1100,800,3),(235,235,235),np.uint8)
cv2.ellipse(F,(400,520),(260,360),0,0,360,(150,180,215),-1)
for ex in (310,490): cv2.circle(F,(ex,430),22,(255,255,255),-1); cv2.circle(F,(ex,430),10,(40,30,20),-1)
cv2.ellipse(F,(400,700),(110,40),0,0,180,(80,60,160),-1)
cv2.rectangle(F,(320,690),(480,720),(235,240,245),-1)
cv2.imwrite('fixtures/face.jpg',F)
# missing-12 variant
im=np.full((H,W,3),(120,110,200),np.uint8); cv2.rectangle(im,(0,560),(W,H),(60,40,90),-1)
for (x0,x1),L,z,t in zip(xs,lens,zen,['13','12','11','21','22','23']):
    k=['13','12','11','21','22','23'].index(t)+1
    if t=='12': cv2.rectangle(im,(int(x0+4),int(z+30)),(int(x1-4),int(z+L)),(50,30,70),-1); continue
    cv2.rectangle(im,(int(x0+4),int(z)),(int(x1-4),int(z+L)),(196+9*k,235,240),-1)
cv2.imwrite('fixtures/intra_missing12.jpg',cv2.GaussianBlur(im,(3,3),0),[cv2.IMWRITE_JPEG_QUALITY,95])
print('ok')
