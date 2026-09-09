"""Validate saved paired predictions and bootstrap groups, not individual images."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from tools.eval_vl01_photo import scores

def main():
    root=Path('data/work/vl01_photo_eval'); out=root/'local_sample'
    assert json.loads((out/'session.json').read_text())['completed']
    ids=json.loads((root/'sample_ids.json').read_text())
    df=pd.read_parquet(root/'eval_manifest.parquet').set_index('sha256').loc[ids]
    codes,groups=pd.factorize(df.group)
    rng=np.random.default_rng(20260909)
    weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
    y=(df.label!='A7').to_numpy(); summary=[]
    for view in ['clean','blur2','shift']:
        bootstrap={};records={}
        for mode in ['fixed','photo']:
            a=np.load(out/f'{mode}_{view}.npz',allow_pickle=False)
            assert a['sha256'].tolist()==ids
            assert np.array_equal(a['labels'],df.label.to_numpy())
            prob=a['probability'];pred=prob>=.5;records[mode]=scores(y,prob)
            cells=[]
            for mask in [~y & ~pred,~y & pred,y & ~pred,y & pred]:
                cells.append(np.bincount(codes,weights=mask.astype(float),minlength=len(groups)))
            tn,fp,fn,tp=(weights@np.array(cells).T).T
            bootstrap[mode]=tp/np.maximum(2*tp+fp+fn,1)+tn/np.maximum(2*tn+fp+fn,1)
        lo,hi=np.quantile(bootstrap['photo']-bootstrap['fixed'],[.025,.975])
        summary.append(dict(view=view,metrics=records,
                            delta_f1_ci95_group_bootstrap=[float(lo),float(hi)]))
    result=dict(n=len(df),groups=len(groups),bootstrap_repeats=2000,results=summary,
                note='Exploratory group bootstrap, one training seed; no independence from earlier model selection claimed.')
    (out/'verified_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__': main()
