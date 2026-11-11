"""Reconstruct audited decision states; all generated joins are traceable."""
import ast,json,time,zipfile
import numpy as np,pandas as pd
from pathlib import Path
import sys
SRC,DATA=map(Path,sys.argv[1:3])
CONFIG={"coordinate_scale":1000000,"reference_latitude_degrees":40}
start=time.perf_counter()
with zipfile.ZipFile(SRC/"all_waybill_info_meituan_0322.csv.zip") as z:
    orders=pd.read_csv(z.open("all_waybill_info_meituan_0322.csv"))
accepted=orders[orders.is_courier_grabbed.eq(1)].copy()
assert not accepted.order_id.duplicated().any()
accepted.to_parquet(DATA/"accepted_orders.parquet",index=False)
snap=pd.read_csv(SRC/"dispatch_rider_meituan.csv")
snap["state_id"]=snap.index
snap["order_refs"]=snap.courier_waybills.map(ast.literal_eval)
long=snap.explode("order_refs").dropna(subset=["order_refs"]).copy()
long["order_refs"]=long.order_refs.astype(int)
joined=long.merge(accepted,left_on="order_refs",right_on="order_id",suffixes=("_snapshot","_order"),how="left",validate="many_to_one")
flags={
"missing_order":joined.order_id.isna(),
"courier_mismatch":joined.courier_id_snapshot.ne(joined.courier_id_order),
"date_mismatch":joined.dt_snapshot.ne(joined.dt_order),
"not_yet_accepted":joined.grab_time.gt(joined.dispatch_time_snapshot),
"already_completed":joined.arrive_time.le(joined.dispatch_time_snapshot),
"invalid_chronology":joined.fetch_time.lt(joined.grab_time)|joined.arrive_time.lt(joined.fetch_time)
}
bad=np.zeros(len(joined),bool)
for name,flag in flags.items():joined[name]=flag;bad|=flag.to_numpy()
joined["valid_reference"]=~bad
good_state=joined.groupby("state_id").valid_reference.all()
valid_ids=set(good_state[good_state].index)
active=snap[snap.state_id.isin(valid_ids)].copy()
joined[joined.state_id.isin(valid_ids)].to_parquet(DATA/"state_order_links.parquet",index=False)
joined[~joined.valid_reference].to_parquet(DATA/"inconsistent_references.parquet",index=False)
records={int(r.order_id):r for r in accepted.itertuples(index=False)}
K=26;N=len(active)
xy=np.zeros((N,K+1,2),np.float64)
due=np.full((N,K),1e9,np.float64)
release=np.zeros((N,K),np.float64)
pred=np.full((N,K),-1,np.int32)
nodes=np.zeros(N,np.int32)
features=np.zeros((N,K,8),np.float32)
meta=[]
scale=CONFIG["coordinate_scale"];xf=111.32*np.cos(np.deg2rad(CONFIG["reference_latitude_degrees"]))
for i,s in enumerate(active.itertuples(index=False)):
    t=s.dispatch_time;cursor=0;rs=[records[int(k)] for k in s.order_refs]
    for r in rs:
        deadline=(r.estimate_arrived_time-t)/60
        remaining_pickup=r.fetch_time>t
        age=(t-r.grab_time)/60
        known_wait=(min(r.fetch_time,t)-r.grab_time)/60
        pickup=-1
        if remaining_pickup:
            pickup=cursor
            xy[i,cursor]=[(r.sender_lng-s.rider_lng)/scale*xf,(r.sender_lat-s.rider_lat)/scale*110.57]
            release[i,cursor]=max(0,(r.estimate_meal_prepare_time-t)/60)
            features[i,cursor]=[xy[i,cursor,0]/5,xy[i,cursor,1]/5,deadline/30,1,release[i,cursor]/30,age/30,known_wait/30,r.is_prebook]
            cursor+=1
        xy[i,cursor]=[(r.recipient_lng-s.rider_lng)/scale*xf,(r.recipient_lat-s.rider_lat)/scale*110.57]
        due[i,cursor]=deadline;pred[i,cursor]=pickup
        features[i,cursor]=[xy[i,cursor,0]/5,xy[i,cursor,1]/5,deadline/30,0,0,age/30,known_wait/30,r.is_prebook]
        cursor+=1
    nodes[i]=cursor # start is xy[i,cursor], initialized to zero
    areas=[int(r.da_id) for r in rs]
    meta.append({"array_id":i,"state_id":int(s.state_id),"date":int(s.dt),"epoch":int(t),"courier_id":int(s.courier_id),
                 "area":max(set(areas),key=areas.count),"n_orders":len(rs),"n_nodes":cursor,"n_pickups":sum(r.fetch_time>t for r in rs),
                 "min_slack":min((r.estimate_arrived_time-t)/60 for r in rs),
                 "mean_slack":np.mean([(r.estimate_arrived_time-t)/60 for r in rs]),
                 "known_wait_mean":np.mean([(min(r.fetch_time,t)-r.grab_time)/60 for r in rs]),
                 "realized_late_share":np.mean([r.arrive_time>r.estimate_arrived_time for r in rs]),
                 "realized_max_lateness":max(0,max((r.arrive_time-r.estimate_arrived_time)/60 for r in rs)),
                 "max_radius_km":float(np.sqrt((xy[i,:cursor]**2).sum(-1)).max()),
                 "prebook_share":np.mean([r.is_prebook for r in rs])})
np.savez_compressed(DATA/"states.npz",xy=xy,deadline=due,release=release,pred=pred,n=nodes,features=features)
pd.DataFrame(meta).to_parquet(DATA/"states.parquet",index=False)
audit={"source_snapshots":len(snap),"empty_snapshots":int(snap.order_refs.map(len).eq(0).sum()),"references":len(joined),
       "valid_active_states":N,"excluded_active_states":int((~good_state).sum()),
       "flags":{k:int(v.sum()) for k,v in flags.items()},"node_distribution":pd.Series(nodes).value_counts().sort_index().to_dict(),
       "join_key":"courier_waybills items -> accepted order_id, not waybill_id",
       "coordinate_assumption":"microdegrees with uniform translation; 2D conversion uses 40-degree reference latitude",
       "elapsed_seconds":time.perf_counter()-start}
(DATA/"reconstruction_counts.json").write_text(json.dumps(audit,indent=2),encoding="utf8")
print(json.dumps(audit,ensure_ascii=False))

