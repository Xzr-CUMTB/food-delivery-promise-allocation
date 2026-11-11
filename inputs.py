"""Counts for the sample flow, release-time checks, and large-cohort coverage."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

def run(root,out):
    src=root/'experiments/processed'
    m=pd.read_parquet(src/'states.parquet')
    links=pd.read_parquet(src/'state_order_links.parquet')
    invalid=pd.read_parquet(src/'inconsistent_references.parquet').groupby('state_id')[['missing_order','courier_mismatch','date_mismatch','not_yet_accepted','already_completed','invalid_chronology']].any()
    active=invalid.copy();flow=[]
    for col in active:
        flow.append((col,int(active[col].sum())))
        active=active[~active[col]]
    pd.DataFrame(flow,columns=['first_failed_check','states']).to_csv(out/'exclusions.csv',index=False)
    m['first']=m.epoch.eq(m.groupby('date').epoch.transform('min'))
    m['sample']=np.select([~m['first'],m.n_nodes.gt(14)],['Later snapshots','Over 14 nodes'],default='Policy sample')
    ref=pd.read_parquet(root/'experiments/results/reference_oracle.parquet').set_index('state_id')
    for k in ['lower','upper']:m[k]=m.state_id.map(ref[k])
    m.groupby('sample').agg(n=('state_id','size'),mean_orders=('n_orders','mean'),mean_pickups=('n_pickups','mean'),mean_nodes=('n_nodes','mean'),min_nodes=('n_nodes','min'),max_nodes=('n_nodes','max'),median_slack=('min_slack','median'),mean_lower=('lower','mean'),mean_upper=('upper','mean')).to_csv(out/'coverage.csv')
    times=m[['date','epoch']].drop_duplicates().sort_values(['date','epoch'])
    times['local_time']=pd.to_datetime(times.epoch,unit='s',utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%H:%M:%S')
    times[['date','local_time']].to_csv(out/'snapshots.csv',index=False)
    p=links[links.fetch_time>links.dispatch_time_snapshot]
    f=p.estimate_meal_prepare_time
    ref=ref[ref.min_slack.gt(0)&ref.resolved.eq(1)].sort_values(['courier_id','date','epoch','state_id']).drop_duplicates(['courier_id','date'])
    discord=ref[ref.lower>ref.realized_max_lateness+.02]
    pd.DataFrame([dict(pending=len(p),absent=int((f==0).sum()),known_by_snapshot=int(((f>0)&(f<=p.dispatch_time_snapshot)).sum()),after_snapshot=int((f>p.dispatch_time_snapshot).sum()),after_pickup=int((f>p.fetch_time).sum()),malformed=int((f.isna()|(f<0)).sum()),discordant=len(discord),overlap=int(discord.index.isin(m[m['sample']=='Policy sample'].state_id).sum()))]).to_csv(out/'input_checks.csv',index=False)
if __name__=='__main__':run(*map(Path,sys.argv[1:3]))
