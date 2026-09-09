import sys, tempfile, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
root=Path(__file__).resolve().parent / 'photo_pilot_bundle'
sys.path.insert(0,str(root))
from src.config import CFG
from src.photo_data import PhotoDataset
from src.roi_data import ROIDataset
from tools.kaggle_safe_crop import load_data
with tempfile.TemporaryDirectory() as temp:
 d=Path(temp)/'data';d.mkdir();rows=[]
 for label in range(1,8):
  for split,n in [('train',2),('val',1)]:
   for i in range(n):
    name=f'{label}_{split}_{i}.jpg';path=d/name
    Image.fromarray(np.random.default_rng(label*100+i+(50 if split=='val' else 0)).integers(0,256,(48,80,3),dtype=np.uint8)).save(path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append(dict(sha256=sha,label=f'A{label}',group=f'{label}_{split}',animal_id=f'{label}_{split}',img_w=80,img_h=48,boxes='[[20,10,40,30]]',pilot_split=split,image_path=name,zip_path=None,zip_member=None))
 pd.DataFrame(rows).to_parquet(d/'pilot_manifest.parquet',index=False)
 (d/'pilot_package.json').write_text(json.dumps(dict(pilot_manifest_sha256=hashlib.sha256((d/'pilot_manifest.parquet').read_bytes()).hexdigest(),code_sha256={},train_rows=14,val_rows=7)))
 tr,va,_=load_data(d);cfg=CFG(img_size=32);cfg.photo_epoch=0
 a=PhotoDataset(va,cfg,train=False,mode='fixed',classes=['A7','ABNORMAL'])
 b=PhotoDataset(va,cfg,train=False,mode='photo',classes=['A7','ABNORMAL'])
 c=ROIDataset(va,cfg,train=False,mode='fixed',classes=['A7','ABNORMAL'])
 assert all(torch.equal(a[i][0],b[i][0]) and torch.equal(a[i][0],c[i][0]) for i in range(len(va)))
 for mode in ['fixed','photo']:
  ds=PhotoDataset(tr,cfg,train=True,mode=mode,classes=['A7','ABNORMAL'])
  assert torch.isfinite(ds[0][0]).all()
 command=[sys.executable,str(root/'tools/kaggle_safe_crop.py'),'--profile','photo','--data',str(d),'--out',str(Path(temp)/'out'),'--smoke','--epochs','1','--workers','0','--batch-size','4']
 subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
 out=Path(temp)/'out';summary=json.loads((out/'comparison.json').read_text())
 assert summary['completed_epochs']=={'fixed':1,'photo':1}
 assert 'blur2_macro_f1' in summary['comparison']
 t=(out/'fixed_last.pt').stat().st_mtime_ns
 subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
 assert t==(out/'fixed_last.pt').stat().st_mtime_ns
 assert (Path(temp)/'photo_crop_pilot_resume.zip').exists()
 print('PASS: identical fixed validation; actual photometric; paired training; blur metrics; resume; separate ZIP')
