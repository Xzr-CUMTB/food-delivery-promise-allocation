"""Paired cohort comparisons, route-specific extensions and small exact benchmarks."""
from pathlib import Path
import itertools,sys,time,platform
import numpy as np
import pandas as pd
from policy_experiments import prepare,individual,allocation,route_check
from exact import enumerate_objectives

def run(root,work,out):
    save=lambda name,rows: pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    data=root/'experiments/processed'
    a=dict(np.load(data/'states.npz'));m=pd.read_parquet(data/'states.parquet')
    detail=pd.read_parquet(work/'scenario_detail.parquet')
    assert detail.scenario.nunique()==11
    base=detail.query("scenario=='Reference' and nodes<=14")
    scenarios=[]
    def compare(label,g):
        pairs={r:z.set_index('state_id').sort_index() for r,z in g.groupby('rule')}
        u=pairs['Uniform']
        for r,z in pairs.items():
            assert z.index.equals(u.index)
            for delta in ([.02,1.,2.,5.] if label=='Reference' else [.02]):
                pos=z.lo>delta;ud=(~pos)&(z.hi>delta); ref=u.lo>delta
                dif=ref.astype(int)-pos.astype(int)
                uncap=(pairs['Uncapped'].lo>delta).sum()
                scenarios.append(dict(scenario=label,rule=r,threshold=delta,n=len(z),positive=pos.sum(),unknown=ud.sum(),pct=100*pos.mean(),upper_pct=100*(pos|ud).mean(),net=dif.sum(),pp=dif.mean()*100,relative_pct=100*dif.sum()/ref.sum() if ref.any() else np.nan,retained_pct=100*dif.sum()/(ref.sum()-uncap) if ref.sum()>uncap else np.nan,improved=(z.hi<u.lo-.02).sum(),worse=(z.lo>u.hi+.02).sum(),mean_lower=z.lo.mean(),mean_upper=z.hi.mean()))
    for name,g in detail[detail.nodes<=14].groupby('scenario',sort=False):compare(name,g)
    compare('Recorded readiness available',base[~base.raw_missing])
    compare('Over 14 nodes',detail[detail.nodes>14])
    save('scenario_comparisons',scenarios)
    rows=[];times=[];orders=[];perm={};exact=[];cold=time.perf_counter()
    for ix,row in enumerate(m[m.state_id.isin(base.state_id)].sort_values('state_id').itertuples()):
        i=row.array_id;n=row.n_nodes
        state=time.perf_counter()
        inputs=prepare(a['xy'][i,:n+1],a['deadline'][i,:n],a['release'][i,:n],a['pred'][i,:n],np.full(n,-1))
        b=allocation(inputs,'Capped')
        lo,hi,c,done=route_check(inputs,b,14)
        revised=inputs[1]+b
        elapsed=(time.perf_counter()-state)*1000
        if ix==0: cold_ms=elapsed
        else:times.append((n,elapsed,done))
        ds,gap=individual(*inputs)
        b0=np.zeros(n);l0,u0,c0,_=route_check(inputs,b0,14)
        trimmed=np.maximum(c[ds]-inputs[1][ds]-hi,0)
        assert np.all(trimmed<=b[ds]+1e-7)
        if u0<=.02:trimmed[:]=0
        assert np.max((c0 if u0<=.02 else c)[ds]-inputs[1][ds]-trimmed)<=max(hi,u0 if u0<=.02 else 0)+1e-7
        group='Single order' if row.n_orders==1 else ('Equal priorities' if np.ptp(np.maximum(gap,0))==0 else 'Mixed priorities')
        rows.append((row.state_id,row.courier_id,row.date,row.n_orders,n,group,l0,u0,lo,hi,trimmed.sum(),sum(trimmed==0),int(np.all(gap<=0))))
        erules={r:allocation(inputs,r)[ds] for r in ['Uniform','Capped','Uncapped','Slack rank','Route lateness']}
        for j,node in enumerate(ds):orders.append((row.state_id,node,c[node],inputs[1][node],*[v[j] for v in erules.values()]))
    states=pd.DataFrame(rows,columns=['state_id','rider','date','orders','nodes','group','l0','u0','lo','hi','trim','zero_orders','floor'])
    alloc=pd.DataFrame(orders,columns=['state_id','node','completion','deadline',*erules.keys()])
    u=base[base.rule=='Uniform'].set_index('state_id').loc[states.state_id]
    h=base[base.rule=='Capped'].set_index('state_id').loc[states.state_id]
    assert np.allclose(h.lo,states.lo) and np.allclose(h.hi,states.hi)
    states['initial']=states.l0>.02
    states['uniform']=u.lo.to_numpy()>.02;states['capped']=states.lo>.02
    states['better']=states.hi<u.lo.to_numpy()-.02;states['worse']=states.lo>u.hi.to_numpy()+.02
    states['both_within']=(states.hi<=.02)&(u.hi.to_numpy()<=.02)
    states['net']=states.uniform.astype(int)-states.capped.astype(int)
    states['category']=np.select([states.group!='Mixed priorities',states.both_within,states.better,states.worse],['identical','both within threshold','improved','worse'],default='overlap')
    save('interval_categories',states.category.value_counts().rename_axis('category').reset_index(name='n'))
    save('priority_groups',states.groupby('group').agg(n=('state_id','size'),initial=('initial','sum'),uniform=('uniform','sum'),capped=('capped','sum'),improved=('better','sum'),worse=('worse','sum'),net=('net','sum'),all_floor=('floor','sum')).reset_index())
    save('budget_use',[dict(full=len(alloc)*3,trimmed=states.trim.sum(),unused=len(alloc)*3-states.trim.sum(),zero_orders=states.zero_orders.sum(),zero_cohorts=(states.trim==0).sum(),original_within=(states.u0<=.02).sum(),retained=len(states))])
    times=pd.DataFrame(times,columns=['nodes','ms','done'])
    trows=[]
    for group,g in [('All warm',times),*times.groupby('nodes')]:
        trows.append(dict(group=group,n=len(g),p50=g.ms.median(),p90=g.ms.quantile(.9),p95=g.ms.quantile(.95),p99=g.ms.quantile(.99),maximum=g.ms.max(),unresolved=(~g.done).sum(),first_call_ms=cold_ms))
    save('timing',trows)
    save('extensions',[dict(rule=r,p50=alloc[r].median(),p95=alloc[r].quantile(.95),p99=alloc[r].quantile(.99),maximum=alloc[r].max()) for r in erules])
    sub=states[states.orders.between(2,4)&states.nodes.le(8)].sort_values('state_id').groupby(['date','orders','initial']).head(1)
    for row in sub.itertuples():
        i=int(m.set_index('state_id').at[row.state_id,'array_id']);n=row.nodes
        dist,d,r,p,_,_=prepare(a['xy'][i,:n+1],a['deadline'][i,:n],a['release'][i,:n],a['pred'][i,:n],np.full(n,-1))
        if n not in perm:perm[n]=np.array(list(itertools.permutations(range(n))),np.int32)
        optimum,_,_,_,_=enumerate_objectives(perm[n],dist,d,r,p,6.,.02)
        aa=alloc[alloc.state_id==row.state_id];need=np.maximum(aa.completion-aa.deadline-.02,0.)
        valid=row.hi<=.02 and need.max()<=6+1e-7
        exact.append(dict(instance=len(exact)+1,orders=row.orders,budget=3*row.orders,optimum=optimum if optimum<1e19 else np.nan,exact_feasible=optimum<=3*row.orders+1e-8,capped_feasible=valid,required=need.sum() if valid else np.nan,excess=need.sum()-optimum if valid else np.nan,relative_excess=(need.sum()-optimum)/optimum if valid and optimum>1e-8 else np.nan,lo=row.lo,hi=row.hi))
    save('exact_comparison',exact)
    rng=np.random.default_rng(20261111);ci=[];riders=np.unique(states.rider)
    draws=rng.integers(0,len(riders),(1000,len(riders)))
    for delta in [.02,1.,2.,5.]:
        s=pd.DataFrame(dict(rider=states.rider,delta=(states.lo>delta).astype(int)-(u.lo.to_numpy()>delta).astype(int)))
        t=s.groupby('rider').delta.agg(['sum','size']);boot=100*t['sum'].to_numpy()[draws].sum(1)/t['size'].to_numpy()[draws].sum(1)
        ci.append(dict(delta=delta,pp=100*s.delta.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975),attempted=1000,valid=1000,undefined=0))
    save('paired_intervals',ci)
if __name__=='__main__':run(*map(Path,sys.argv[1:4]))
