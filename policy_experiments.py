"""Scenario-adapted allocations for reconstructed Meituan cohorts (minutes)."""
from pathlib import Path
import sys,time
import numpy as np
import pandas as pd
from solver import solve, bounds, feasible_dp, evaluate_route, distance_matrix
from allocation import allocate

# Each setting changes one input relative to the reference.
SCENARIOS = {
    "Reference": {}, "Longitude at 30 deg": {"latitude":30}, "Longitude at 50 deg":{"latitude":50},
    "Speed 12 km/h":{"speed":12}, "Speed 18 km/h":{"speed":18},
    "Service 0.25 min":{"service":.25}, "Service 1 min":{"service":1.},
    "No readiness wait":{"no_release":True}, "Shared pickup stop":{"shared":True},
    "Initial rest 0 min":{"rest":0.}, "Initial rest 5 min":{"rest":5.}
}

def prepare(xy,due,rel,pred,shops,latitude=40,speed=15,service=.5,rest=3.,no_release=False,shared=False):
    points=xy.copy()
    points[:,0]*=np.cos(np.deg2rad(latitude))/np.cos(np.deg2rad(40))
    dist=distance_matrix(points,speed)
    rel=rel.copy()
    if no_release: rel[:]=0
    if shared:
        # For consecutive pickups at the same merchant this makes the next
        # completion max(previous completion, next release + service).
        for j in np.flatnonzero(shops>=0):
            same=np.flatnonzero(shops==shops[j])
            dist[j,same]=-service
    return dist,due,rel,pred,service,rest

def individual(dist,due,rel,pred,service,rest):
    early=np.maximum(rest+dist[-1,:-1],rel)+service
    delivery=np.flatnonzero(due<1e8)
    for j in delivery:
        if pred[j]>=0:
            k=pred[j]
            early[j]=max(early[k]+dist[k,j],rel[j])+service
    return delivery,early[delivery]-due[delivery]

def route_check(inputs,b,node_limit,secondary=True):
    dist,due,rel,pred,service,rest=inputs
    revised=due+b
    lo,hi,route,_,done=solve(dist,revised,rel,pred,service,.02,node_limit,rest)
    if secondary and len(due)<=node_limit:
        yes,route=feasible_dp(dist,revised,rel,pred,service,hi+1e-8,rest)
        assert yes
    loss,completion=evaluate_route(route,dist,revised,rel,pred,service,rest)
    assert loss<=hi+1e-6
    return lo,hi,completion,done

def allocation(inputs,rule,limit=14):
    dist,due,rel,pred,s,rest=inputs
    ix,excess=individual(*inputs)
    b=np.zeros(len(due)); budget=len(ix)*3.
    if rule=="Uniform": b[ix]=3.
    elif rule=="Slack rank":
        remaining=budget
        for v in np.unique(excess)[::-1]:
            tied=ix[excess==v]; amount=min(6.,remaining/len(tied))
            b[tied]=amount;remaining-=len(tied)*amount
            if remaining<=1e-10:break
    else:
        if rule=="Route lateness":
            _,_,comp,_=route_check(inputs,b,limit)
            excess=comp[ix]-due[ix]
        b[ix],ok=allocate(.1+np.maximum(excess,0),budget,np.inf if rule=="Uncapped" else 6.)
        assert ok
    assert abs(b.sum()-budget)<1e-7
    return b

def run(root,out):
    processed=root/'experiments/processed'
    a=dict(np.load(processed/'states.npz'))
    meta=pd.read_parquet(processed/'states.parquet')
    meta=meta[meta.epoch.eq(meta.groupby('date').epoch.transform('min'))]
    links=pd.read_parquet(processed/'state_order_links.parquet')
    shops=np.full(a['pred'].shape,-1,dtype=np.int64)
    missing=set()
    idmap=meta.set_index('state_id').array_id.to_dict()
    for state,g in links[links.state_id.isin(idmap)].groupby('state_id'):
        i=idmap[state];pick=g.fetch_time>g.dispatch_time_snapshot
        indices=np.cumsum(1+pick.to_numpy(int))-2
        shops[i,indices[pick]]=g.loc[pick,'poi_id'].to_numpy()
        if (pick & g.estimate_meal_prepare_time.eq(0)).any():missing.add(state)
    store=[]
    for label,kw in SCENARIOS.items():
        t0=time.perf_counter()
        for row in meta.itertuples():
            n=row.n_nodes;i=row.array_id
            if n>14 and label!='Reference':continue
            limit=18 if n>14 else 14
            inputs=prepare(a['xy'][i,:n+1],a['deadline'][i,:n],a['release'][i,:n],a['pred'][i,:n],shops[i,:n],**kw)
            use=['Uniform','Capped','Uncapped']
            if label=='Reference' and n<=14:use+=['Slack rank','Route lateness']
            for rule in use:
                start=time.perf_counter()
                b=allocation(inputs,rule,limit)
                lo,hi,c,done=route_check(inputs,b,limit,secondary=label=='Reference' and n<=14)
                runtime=1000*(time.perf_counter()-start)
                ix,gap=individual(*inputs)
                store.append((label,row.state_id,n,rule,lo,hi,done,runtime,np.max(b),np.all(np.maximum(gap,0)==np.maximum(gap,0)[0]),row.state_id in missing))
        print(label,round(time.perf_counter()-t0,1),flush=True)
        pd.DataFrame(store,columns=['scenario','state_id','nodes','rule','lo','hi','done','ms','max_extension','equal_priority','raw_missing']).to_parquet(out/'scenario_detail.parquet',index=False)
if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))