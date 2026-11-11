"""Run from two official source files into a new output directory."""
import argparse,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from policy_experiments import run as policies
from summarize import run as summarize
from inputs import run as checks
from solver import distance_matrix,solve,bounds
from prediction_intervals import run as intervals
from response_figures import run as figures
from models import main as predict

def reference(root):
    src=root/'experiments/processed'
    a=dict(np.load(src/'states.npz'));m=pd.read_parquet(src/'states.parquet')
    records=[]
    for x in m.itertuples():
        i=x.array_id;n=x.n_nodes;dist=distance_matrix(a['xy'][i,:n+1])
        d,r,p=[a[k][i,:n] for k in ['deadline','release','pred']]
        l,h,_,_,done=solve(dist,d,r,p,.5)
        lb,ub,_=bounds(dist,d,r,p,.5)
        records.append((l,h,done,lb,ub))
    m[['lower','upper','resolved','analytic_lower','heuristic_upper']]=pd.DataFrame(records,index=m.index)
    results=root/'experiments/results';results.mkdir(exist_ok=True)
    m.to_parquet(results/'reference_oracle.parquet',index=False)
    eligible=m[m.min_slack.gt(0)&m.resolved].sort_values(['courier_id','date','epoch','state_id']).drop_duplicates(['courier_id','date']).copy()
    eligible['positive_g']=eligible.lower>.02;eligible['late_any']=eligible.realized_late_share>0
    eligible['log_g']=np.log1p((eligible.lower+eligible.upper)/2)
    hour=pd.to_datetime(eligible.epoch,unit='s',utc=True).dt.tz_convert('Asia/Shanghai').dt.hour
    eligible['sin_hour']=np.sin(2*np.pi*hour/24);eligible['cos_hour']=np.cos(2*np.pi*hour/24)
    nr=[]
    for x in eligible.itertuples():
        i=x.array_id;n=x.n_nodes
        lo,hi,*_=solve(distance_matrix(a['xy'][i,:n+1]),a['deadline'][i,:n],np.zeros(n),a['pred'][i,:n],.5)
        nr.append(np.log1p((lo+hi)/2))
    eligible['no_release_log_g']=nr
    return eligible

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output
    for folder in ['experiments/processed','tables','figures','work']:(root/folder).mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('reconstruct.py')),str(args.source),str(root/'experiments/processed')],check=True)
    obs=reference(root);predict(obs,root/'work');checks(root,root/'tables')
    policies(root,root/'work');summarize(root,root/'work',root/'tables')
    intervals(root/'work/test_predictions.parquet',root/'tables')
    figures(root/'tables',root/'figures')
if __name__=='__main__':main()
