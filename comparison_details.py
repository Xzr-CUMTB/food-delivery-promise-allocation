"""Additional paired intervals, decision-rule timing, and exact-set extension distributions."""
import itertools,sys
from pathlib import Path
import numpy as np
import pandas as pd
from policy_experiments import prepare,allocation,route_check
from exact import enumerate_objectives
from solver import evaluate_route

def run(root,work,out):
    src=root/'experiments/processed';a=dict(np.load(src/'states.npz'))
    states=pd.read_parquet(src/'states.parquet').set_index('state_id')
    z=pd.read_parquet(work/'scenario_detail.parquet').query("nodes<=14 and scenario=='Reference'")
    u=z[z.rule=='Uniform'].sort_values('state_id').set_index('state_id')
    ids,groups=np.unique(states.loc[u.index,'courier_id'],return_inverse=True)
    draws=np.random.default_rng(20261111).integers(len(ids),size=(1000,len(ids)))
    sizes=np.bincount(groups)[draws].sum(axis=1);rows=[]
    for rule,g in z.groupby('rule'):
        b=g.set_index('state_id').loc[u.index]
        for threshold in [.02,1,2,5]:
            diff=(b.lo>threshold).astype(int)-(u.lo>threshold).astype(int)
            values=100*np.bincount(groups,weights=diff)[draws].sum(1)/sizes
            rows.append([rule,threshold,100*diff.mean(),*np.quantile(values,[.025,.975]),1000,0])
    pd.DataFrame(rows,columns=['rule','threshold','paired_pp','lower95','upper95','valid','undefined']).to_csv(out/'all_rule_intervals.csv',index=False)
    z.iloc[5:].groupby('rule').ms.agg(n='count',p50='median',p95=lambda x:x.quantile(.95),p99=lambda x:x.quantile(.99)).to_csv(out/'rule_timing.csv')
    pool=states.loc[u.index].query('2<=n_orders<=4 and n_nodes<=8').copy()
    def inputs(row):
        i,n=row.array_id,row.n_nodes
        return prepare(a['xy'][i,:n+1],a['deadline'][i,:n],a['release'][i,:n],a['pred'][i,:n],np.full(n,-1))
    positive=[]
    for row in pool.itertuples():
        ip=inputs(row);l,_,_,_=route_check(ip,np.zeros(row.n_nodes),14,False);positive.append(l>.02)
    pool['positive']=positive;pool=pool.sort_index().groupby(['date','n_orders','positive']).head(1)
    result=[];perms={}
    for idx,row in enumerate(pool.itertuples(),1):
        ip=inputs(row);dist,d,r,p,s,rest=ip;n=row.n_nodes;ix=np.flatnonzero(d<1e8)
        if n not in perms:perms[n]=np.array(list(itertools.permutations(range(n))),np.int32)
        optimum,_,_,route,_=enumerate_objectives(perms[n],dist,d,r,p,6.,.02)
        if optimum>3*len(ix)+1e-8:continue
        _,optimal_comp=evaluate_route(perms[n][route],dist,d,r,p,s,rest)
        b=allocation(ip,'Capped');lo,hi,comp,_=route_check(ip,b,14)
        for j in ix:result.append([idx,j,b[j],max(comp[j]-d[j]-.02,0),max(optimal_comp[j]-d[j]-.02,0)])
    df=pd.DataFrame(result,columns=['instance','node','full_capped','required_capped_route','exact_optimum_route'])
    df.to_csv(out/'feasible_exact_orders.csv',index=False)
if __name__=='__main__':run(*map(Path,sys.argv[1:4]))
