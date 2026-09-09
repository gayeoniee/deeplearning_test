"""Pack frozen VL01 development evaluation, with original JPEGs and inference weights."""
from pathlib import Path
import hashlib,json,zipfile,io
import pandas as pd

def main():
    root=Path('data/work/vl01_photo_eval');dest=Path('data/packages/vl01_photo_eval.zip')
    if dest.exists(): raise FileExistsError(dest)
    df=pd.read_parquet(root/'eval_manifest.parquet')
    portable=df.copy();portable['zip_path']=None;portable['zip_member']=None
    portable['image_path']='images/'+portable.sha256+'.jpg'
    temp=dest.with_suffix('.zip.part')
    hashes={}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as out:
        for archive,rows in df.groupby('zip_path'):
            with zipfile.ZipFile(archive) as source:
                for i,row in enumerate(rows.itertuples()):
                    data=source.read(row.zip_member)
                    assert hashlib.sha256(data).hexdigest()==row.sha256
                    out.writestr(f'images/{row.sha256}.jpg',data)
        b=io.BytesIO();portable.to_parquet(b,index=False);out.writestr('eval_manifest.parquet',b.getvalue())
        hashes['eval_manifest.parquet']=hashlib.sha256(b.getvalue()).hexdigest()
        for name in ['fixed.pt','photo.pt','audit.json','sample_ids.json']:
            data=(root/name).read_bytes();out.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        for name in ['tools/eval_vl01_photo.py','src/__init__.py','src/config.py','src/roi_data.py','src/original_data.py','src/safe_crop.py']:
            data=Path(name).read_bytes();out.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
        out.writestr('package.json',json.dumps(dict(files_sha256=hashes,images='original JPEG SHA256 equals filename',rows=len(df),note='development evaluation, not independent external test'),indent=2))
    temp.replace(dest)
    print(dest,round(dest.stat().st_size/1024**3,2),'GiB',flush=True)
if __name__=='__main__': main()
