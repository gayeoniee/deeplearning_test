"""Audit saved validation predictions without training or opening holdout."""
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.roi_data import roi_window
from src.original_data import letterbox


def main():
    out = Path('data/work/roi_error_audit')
    out.mkdir(parents=True, exist_ok=True)
    saved = Path('data/work/roi_crop_pilot_result')
    with zipfile.ZipFile('data/packages/dogskin_safe_crop_pilot.zip') as z:
        manifest = pd.read_parquet(io.BytesIO(z.read('pilot_manifest.parquet')))
        val = manifest[manifest.pilot_split == 'val'].copy().set_index('sha256')
        for mode in ['fixed', 'safe']:
            pred = np.load(saved / f'{mode}_epoch3_val.npz', allow_pickle=False)
            assert len(pred['sha256']) == len(set(pred['sha256'])) == len(val)
            assert set(pred['sha256']) == set(val.index)
            assert np.array_equal(val.loc[pred['sha256'], 'label'], pred['original_label'])
            val[f'{mode}_p'] = pd.Series(pred['probability'], index=pred['sha256'])
        val['abnormal'] = val.label != 'A7'
        for mode in ['fixed', 'safe']:
            val[f'{mode}_error'] = (val[f'{mode}_p'] >= .5) != val.abnormal
        val['error_kind'] = np.where(~val.fixed_error, 'correct',
                                      np.where(val.abnormal, 'FN', 'FP'))
        geometry = []
        for row in val.itertuples():
            boxes = json.loads(row.boxes)
            x,y,r,b = roi_window(row.img_w, row.img_h, boxes)
            geometry.append(dict(sha256=row.Index, box_count=len(boxes),
                                 roi_w=r-x, roi_h=b-y,
                                 roi_image_fraction=(r-x)*(b-y)/(row.img_w*row.img_h),
                                 min_box_side=min((min(a[2]-a[0],a[3]-a[1]) for a in boxes), default=0)))
        val = val.join(pd.DataFrame(geometry).set_index('sha256'))
        # Same 384px validation input. Recheck old texture association on this model.
        texture = []
        for i, (sha, row) in enumerate(val.iterrows()):
            with Image.open(io.BytesIO(z.read(row.image_path))) as im:
                im = im.convert('RGB')
            roi = letterbox(im.crop(roi_window(*im.size,json.loads(row.boxes))),384)
            a = np.asarray(roi,dtype=np.float32)/255
            g = a @ np.array([.299,.587,.114],dtype=np.float32)
            lap = g[:-2,1:-1]+g[2:,1:-1]+g[1:-1,:-2]+g[1:-1,2:]-4*g[1:-1,1:-1]
            texture.append(dict(sha256=sha, sharpness=float(lap.var()),
                hair=float(np.abs(np.diff(g,axis=0)).mean()+np.abs(np.diff(g,axis=1)).mean())))
            if (i+1)%1000 == 0:
                print(f'Texture audit {i+1}/4000',flush=True)
        val = val.join(pd.DataFrame(texture).set_index('sha256'))
        val['roi_band'] = pd.cut(val[['roi_w','roi_h']].max(axis=1),
                                 [0,320,640,1280,float('inf')], labels=['<=320','321-640','641-1280','>1280'])
        summary = {'epoch':3, 'threshold':.5, 'val_rows':len(val),
                   'errors':val.error_kind.value_counts().to_dict(),
                   'both_wrong':int((val.fixed_error & val.safe_error).sum()),
                   'fixed_only_wrong':int((val.fixed_error & ~val.safe_error).sum()),
                   'safe_only_wrong':int((~val.fixed_error & val.safe_error).sum())}
        summary['texture_associations'] = []
        for label in ['A7','A2']:
            subset = val[val.label == label]
            for feature in ['sharpness','hair']:
                summary['texture_associations'].append(dict(label=label,feature=feature,
                    error_median=float(subset.loc[subset.fixed_error,feature].median()),
                    correct_median=float(subset.loc[~subset.fixed_error,feature].median()),
                    auc_predicting_error=float(roc_auc_score(subset.fixed_error,subset[feature]))))
        for field in ['label','source_chunk','roi_band']:
            table = val.groupby(field, observed=True).agg(n=('fixed_error','size'),
                   fixed_errors=('fixed_error','sum'),safe_errors=('safe_error','sum'))
            table['fixed_error_rate'] = table.fixed_errors/table.n
            table['safe_error_rate'] = table.safe_errors/table.n
            table.to_csv(out / f'by_{field}.csv')
            summary[field] = table.reset_index().to_dict('records')
        val.reset_index().to_csv(out/'validation_errors.csv',index=False)
        # Deterministic random sample, plus separate high-confidence errors.
        selected = []
        for kind in ['FP','FN']:
            errors = val[val.error_kind == kind]
            samples = {'random':errors.sample(min(12,len(errors)),random_state=42),
                       'confident':errors.sort_values('fixed_p',ascending=kind=='FN').head(12)}
            for sampling, rows in samples.items():
                sheet = Image.new('RGB',(1200,4*205),'white')
                draw = ImageDraw.Draw(sheet)
                for i, (sha,row) in enumerate(rows.iterrows()):
                    with Image.open(io.BytesIO(z.read(row.image_path))) as im:
                        im = im.convert('RGB')
                    window = roi_window(*im.size,json.loads(row.boxes))
                    roi = im.crop(window)
                    full = im.copy()
                    ImageDraw.Draw(full).rectangle(window,outline='lime',width=8)
                    x,y=(i%3)*400,(i//3)*205
                    for dx, picture in [(0,full),(200,roi)]:
                        tile=ImageOps.contain(picture,(196,170))
                        sheet.paste(tile,(x+dx,y+22))
                    draw.text((x,y),f'{i+1} {row.label} p={row.fixed_p:.3f} {sha[:8]}',fill='black')
                    selected.append({'sheet':f'{kind}_{sampling}.jpg','cell':i+1,'sha256':sha,
                                     'label':row.label,'p':row.fixed_p,'window':window})
                sheet.save(out/f'{kind}_{sampling}.jpg',quality=90)
        (out/'review_samples.json').write_text(json.dumps(selected,indent=2))
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
