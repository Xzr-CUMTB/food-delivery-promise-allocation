"""Fixed-prediction paired, pointwise rider bootstrap. Ties use score blocks."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

def scores(y,p,w):
    order=np.argsort(-p,kind='stable');ys=y[order];ws=w[order]
    keep=np.r_[p[order][:-1]!=p[order][1:],True]
    tp=np.cumsum(ws*ys)[keep];fp=np.cumsum(ws*(1-ys))[keep]
    den=tp+fp
    auc=np.trapezoid(np.r_[0,tp/tp[-1]],np.r_[0,fp/fp[-1]]) if tp[-1]*fp[-1]>0 else np.nan
    ap=np.sum(np.diff(np.r_[0,tp])*np.divide(tp,den,out=np.zeros_like(tp,dtype=float),where=den>0))/tp[-1] if tp[-1]>0 else np.nan
    return np.array([auc,ap,np.average((p-y)**2,weights=w),np.average(-y*np.log(np.clip(p,1e-15,1))-(1-y)*np.log(np.clip(1-p,1e-15,1)),weights=w)])
def run(file,out):
    pred=pd.read_parquet(file)
    pivot=pred.pivot(index='state_id',columns='model',values='probability')
    obs=pred.drop_duplicates('state_id').set_index('state_id').loc[pivot.index]
    train_ids=None;rows=[]
    scopes={'All test':np.ones(len(obs),bool),'23 October':obs.date.eq(20221023),'24 October':obs.date.eq(20221024)}
    for scope,mask in scopes.items():
        y=obs.late_any[mask].to_numpy(int);ids,group=np.unique(obs.courier_id[mask],return_inverse=True)
        rng=np.random.default_rng(20261111);n=len(ids); boot=[]
        pairs=[('linear','linear_G'),('spline','spline_G'),('tree','tree_G'),('linear','linear_LB'),('linear','linear_UB'),('linear','linear_no_release_G')]
        for i in range(1000):
            weights=np.bincount(rng.integers(n,size=n),minlength=n)[group]
            result={model:scores(y,pivot.loc[mask,model].to_numpy(),weights) for model in pivot}
            boot.append([result[b]-result[a] for a,b in pairs])
        boot=np.array(boot)
        for j,(a,b) in enumerate(pairs):
            for k,metric in enumerate(['AUC','AP','Brier','log loss']):
                v=boot[:,j,k];good=np.isfinite(v)
                rows.append(dict(scope=scope,baseline=a,new=b,metric=metric,low=np.nanquantile(v,.025),high=np.nanquantile(v,.975),attempted=1000,valid=good.sum(),undefined=(~good).sum()))
    pd.DataFrame(rows).to_csv(out/'prediction_intervals.csv',index=False)
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
