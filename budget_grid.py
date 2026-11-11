"""Budget–cap grid, simulated service and tolerance-matched numerical checks."""
from pathlib import Path
import itertools,sys
import numpy as np
import pandas as pd
from policy_experiments import individual,prepare,route_check
from allocation import allocate
from solver import solve
from exact import enumerate_objectives

def run(root,out):
    src=root/'experiments/processed';a=dict(np.load(src/'states.npz'))
    m=pd.read_parquet(src/'states.parquet')
    m=m[m.n_nodes.le(14)&m.epoch.eq(m.groupby('date').epoch.transform('min'))].copy()
    ref=pd.read_parquet(root/'experiments/results/reference_oracle.parquet').set_index('state_id')
    small=m[m.n_nodes.le(8)].sort_values('state_id').copy()
    small['positive']=small.state_id.map(ref.lower)>.02
    checks=set(small.groupby('positive').head(50).state_id)
    identity=[];records=[];zero=[]
    for row in m.itertuples():
        i=row.array_id;n=row.n_nodes
        ip=prepare(a['xy'][i,:n+1],a['deadline'][i,:n],a['release'][i,:n],a['pred'][i,:n],np.full(n,-1))
        dist,d,rel,p,svc,rest=ip;ix,gap=individual(*ip);nn=len(ix)
        if row.state_id in checks:
            lo,hi,*_=solve(dist,d,rel,p,svc,.005,14,rest)
            for shift in (0.,1.,3.,5.):
                dd=d.copy();dd[ix]+=shift
                ll,uu,*_=solve(dist,dd,rel,p,svc,.005,14,rest)
                overlaps=ll<=max(hi-shift,0)+1e-7 and max(lo-shift,0)<=uu+1e-7
                assert overlaps
                identity.append([len(identity)+1,shift,float(ll),float(uu),overlaps])
        if n<=8 and 2<=nn<=4:
            lo,_,*_=solve(dist,d,rel,p,svc,.02,14,rest)
            zero.append([row.state_id,i,row.date,nn,n,lo>.02])
        for mean in (1.,3.,5.):
            for cap in (3.,6.,np.inf):
                if mean>cap:continue
                for rule in ('Uniform','Individual'):
                    b=np.zeros(n);w=np.ones(nn) if rule=='Uniform' else .1+np.maximum(gap,0)
                    b[ix],ok=allocate(w,mean*nn,cap);assert ok
                    lo,hi,c,_=route_check(ip,b,14)
                    records.append([mean,cap,rule,nn,b.sum(),(c[ix]>d[ix]+.02).sum(),(c[ix]>d[ix]+b[ix]+.02).sum(),lo>.02,lo<=.02<hi,c.max(),b.max()])
    q=pd.DataFrame(records,columns=['budget','cap','rule','orders','minutes','late_recorded','late_revised','positive','unknown','end','max_extension'])
    tab=q.groupby(['budget','cap','rule'],as_index=False).agg(cohorts=('end','size'),orders=('orders','sum'),minutes=('minutes','sum'),late_recorded=('late_recorded','sum'),late_revised=('late_revised','sum'),positive=('positive','sum'),unknown=('unknown','sum'),mean_end=('end','mean'),max_extension=('max_extension','max'))
    tab.to_csv(out/'budget_grid.csv',index=False)
    pd.DataFrame(identity,columns=['check','shift','lower','upper','intersects_expected']).to_csv(out/'translation_checks.csv',index=False)
    small=pd.DataFrame(zero,columns=['state_id','i','date','orders','n','positive']).sort_values('state_id').groupby(['date','orders','positive']).head(1)
    exact=[];perms={}
    for j,row in enumerate(small.itertuples(),1):
        i,n=row.i,row.n
        ip=prepare(a['xy'][i,:n+1],a['deadline'][i,:n],a['release'][i,:n],a['pred'][i,:n],np.full(n,-1))
        dist,d,rel,p,_,_=ip
        if n not in perms:perms[n]=np.array(list(itertools.permutations(range(n))),np.int32)
        for cap in [3.,6.,np.inf]:
            for tol in [0.,.02]:
                total,end,deficit,route,num=enumerate_objectives(perms[n],dist,d,rel,p,cap,tol)
                exact.append([j,cap,tol,total if route>=0 else np.nan,route>=0,total<=3*row.orders+1e-8,deficit,num])
    pd.DataFrame(exact,columns=['instance','cap','tolerance','minimum','cap_feasible','budget_feasible','exact_G','legal_routes']).to_csv(out/'exact_grid.csv',index=False)
if __name__=='__main__':run(*map(Path,sys.argv[1:3]))
