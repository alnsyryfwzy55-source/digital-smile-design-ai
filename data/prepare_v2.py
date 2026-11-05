import json,os,cv2,numpy as np,csv,collections,shutil
SIX=['13','12','11','21','22','23']; CID={n:i+1 for i,n in enumerate(SIX)}
SRC={'fdi':('src/Teeth-de',['train','valid','test']),'front':('src/tooth_fr',['train','valid','test']),'lowres':('src/teeth_v1',['train'])}
ROUTE={'train':'train','valid':'val','test':'test'}
OUT='smile_teeth_v2'; shutil.rmtree(OUT,ignore_errors=True)
for sp in ['train','val','test']:
    os.makedirs(f'{OUT}/{sp}/images');os.makedirs(f'{OUT}/{sp}/masks')
def dhash(img,n=16):
    g=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(n+1,n),interpolation=cv2.INTER_AREA)
    return (g[:,1:]>g[:,:-1]).ravel()
def clean(m):
    out=np.zeros_like(m)
    for c in range(1,7):
        b=(m==c).astype(np.uint8)
        if b.sum()==0: continue
        n,lab,st,_=cv2.connectedComponentsWithStats(b,8)
        if n>2: b=(lab==1+np.argmax(st[1:,cv2.CC_STAT_AREA])).astype(np.uint8)
        # fill holes
        cnt,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        f=np.zeros_like(b); cv2.drawContours(f,cnt,-1,1,-1)
        out[(f>0)&(out==0)]=c
    return out
cands=[]
stats=collections.Counter()
for name,(root,splits) in SRC.items():
    for split in splits:
        p=f'{root}/{split}/_annotations.coco.json'
        d=json.load(open(p)); cats={c['id']:c['name'] for c in d['categories']}
        by=collections.defaultdict(list)
        for a in d['annotations']: by[a['image_id']].append(a)
        for k,im in enumerate(d['images']):
            img=cv2.imread(f"{root}/{split}/{im['file_name']}")
            if img is None: continue
            h,w=img.shape[:2]; m=np.zeros((h,w),np.uint8)
            for a in by[im['id']]:
                n=cats[a['category_id']]
                if n not in CID: continue
                for seg in a.get('segmentation') or []:
                    if isinstance(seg,list) and len(seg)>=6:
                        cv2.fillPoly(m,[np.array(seg,np.float32).reshape(-1,2).round().astype(np.int32)],CID[n])
            m=clean(m); present=sum((m==c).sum()>=40*(h/640)**2 for c in range(1,7))
            need=6 if split!='train' else 5
            if present<need: stats[f'drop_incomplete_{name}_{split}']+=1; continue
            cands.append(dict(src=name,split=ROUTE[split],k=k,img=img,m=m,present=present,h=dhash(img)))
# leakage: a val/test image too similar to ANY train image (or to another val/test) is dropped
tr=[c for c in cands if c['split']=='train']
H=np.array([c['h'] for c in tr])
keep=[]
seen=[]
for c in cands:
    if c['split']!='train':
        dist=(H!=c['h']).sum(1).min()
        dd=min([(c['h']!=s).sum() for s in seen],default=999)
        if dist<=40 or dd<=40: stats[f'drop_leak_{c["src"]}_{c["split"]}']+=1; continue
        seen.append(c['h'])
    keep.append(c)
# also drop exact-ish duplicates inside train
tseen=[];final=[]
for c in keep:
    if c['split']=='train':
        if any((c['h']!=s).sum()<=8 for s in tseen): stats['drop_dup_train']+=1; continue
        tseen.append(c['h'])
    final.append(c)
rows=[]
for c in final:
    fid=f"{c['src']}_{c['split']}_{c['k']:03d}"
    cv2.imwrite(f"{OUT}/{c['split']}/images/{fid}.jpg",c['img'],[cv2.IMWRITE_JPEG_QUALITY,95])
    cv2.imwrite(f"{OUT}/{c['split']}/masks/{fid}.png",c['m'])
    rows.append((c['split'],fid,c['src'],c['present'],c['img'].shape[1]))
with open(f'{OUT}/manifest.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['split','id','source','teeth_present','size']);w.writerows(rows)
print(collections.Counter((r[0],r[2]) for r in rows)); print(dict(stats))
