"""Frozen VL01 development evaluation; never train or tune thresholds."""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import ImageFile
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.roi_data import ROIDataset, roi_window
from src.config import CFG
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

class Views(ROIDataset):
    def __init__(self,*a,shift=False,**kw):
        self.shift=shift
        super().__init__(*a,mode='fixed',train=False,classes=['A7','ABNORMAL'],**kw)
    def prepare_image(self,image,row):
        boxes=json.loads(row.boxes)
        x,y,r,b=roi_window(*image.size,boxes)
        if self.shift:
            # One predeclared diagonal; shift is relative to crop size, boundary-clamped.
            w,h=r-x,b-y
            x=min(image.width-w,max(0,x+round(.2*w)))
            y=min(image.height-h,max(0,y+round(.2*h)))
            r,b=x+w,y+h
        return image.crop((x,y,r,b))

def scores(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(macro_f1=float(f1_score(y,p>=.5,average='macro')),
                auroc=float(roc_auc_score(y,p)),recall=float(tp/(tp+fn)),
                specificity=float(tn/(tn+fp)),fp=int(fp),fn=int(fn))

def run(args):
    import timm
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    root=args.data;out=args.out;out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(root/'eval_manifest.parquet')
    if args.sample:
        ids=json.loads((root/'sample_ids.json').read_text())
        df=df.set_index('sha256').loc[ids].reset_index()
    if args.smoke: df=df.groupby('label',group_keys=False).head(1).reset_index(drop=True)
    df['original_label']=df.label
    df['label']=df.label.where(df.label=='A7','ABNORMAL')
    if args.portable:
        df['image_path']=df.sha256.map(lambda v:str(root/'images'/f'{v}.jpg'))
        df['zip_path']=None
    cfg=CFG(img_size=384,batch_size=args.batch_size,num_workers=0)
    config=dict(dataset_sha=hashlib.sha256((root/'eval_manifest.parquet').read_bytes()).hexdigest(),
        ids_sha=hashlib.sha256('\n'.join(df.sha256).encode()).hexdigest(),
        code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        weights={m:hashlib.sha256((root/f'{m}.pt').read_bytes()).hexdigest() for m in ['fixed','photo']},
        sample=args.sample,smoke=args.smoke,threshold=.5,epoch=5,
        conditions=['clean','blur2 kernel13 reflect','shift +20% width and height clamped'],
        n=len(df),device=device,torch=str(torch.__version__),timm=timm.__version__)
    protocol=out/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=config:
        raise ValueError('Evaluation protocol changed; use new output directory')
    protocol.write_text(json.dumps(config,indent=2))
    models={}
    for mode in ['fixed','photo']:
        m=timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k',pretrained=False,num_classes=2)
        m.load_state_dict(torch.load(root/f'{mode}.pt',map_location='cpu',weights_only=True))
        models[mode]=m.eval().to(device)
    y=(df.original_label!='A7').to_numpy()
    summary=[]
    started=time.monotonic()
    for view in ['clean','blur2','shift']:
        done=all((out/f'{m}_{view}.npz').exists() for m in models)
        if not done:
            ds=Views(df,cfg,shift=view=='shift')
            loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
            probs={m:[] for m in models}
            with torch.inference_mode():
                for i,(x,_) in enumerate(loader):
                    x=x.to(device)
                    if view=='blur2':x=gaussian_blur(x,[13,13],[2.,2.])
                    for mode,m in models.items():
                        with torch.autocast(device_type=device,enabled=device=='cuda'):
                            p=m(x).float().softmax(1)[:,1]
                        probs[mode].extend(p.cpu().tolist())
                    if (i+1)%25==0: print(view,(i+1)*args.batch_size,'/',len(df),flush=True)
            ds.close()
            for mode,p in probs.items():
                np.savez_compressed(out/f'{mode}_{view}.npz',sha256=df.sha256.to_numpy(dtype=str),
                     labels=df.original_label.to_numpy(dtype=str),probability=np.asarray(p))
        for mode in models:
            a=np.load(out/f'{mode}_{view}.npz',allow_pickle=False)
            assert np.array_equal(a['sha256'],df.sha256.to_numpy())
            record=dict(mode=mode,view=view,**scores(y,a['probability']))
            summary.append(record);print(json.dumps(record),flush=True)
        pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
    (out/'session.json').write_text(json.dumps(dict(completed=True,elapsed_sec=time.monotonic()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sample',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--portable',action='store_true')
    p.add_argument('--batch-size',type=int,default=8);run(p.parse_args())
