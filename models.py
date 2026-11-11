"""Registered finite comparison; validation chooses once, test predictions are frozen."""
from pathlib import Path
import json,time,warnings
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];P=R.parent;SEED=20261007
BASE=['n_orders','n_pickups','min_slack','mean_slack','known_wait_mean','max_radius_km','prebook_share','sin_hour','cos_hour']
def metrics(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]))
def main(d,out):
 d['log_lb']=np.log1p(d.analytic_lower);d['log_ub']=np.log1p(d.heuristic_upper)
 split={'train':d.date.le(20221021).to_numpy(),'validation':d.date.eq(20221022).to_numpy(),'test':d.date.ge(20221023).to_numpy()};y=d.late_any.to_numpy(int)
 matrices={};knots=[]
 for extra in ['', 'G']:
  cols=BASE+(['log_g'] if extra else []);X=d[cols].to_numpy(float);blocks=[X]
  for col in BASE[:6]+(['log_g'] if extra else []):
   x=d[[col]].to_numpy(float);k=np.unique(np.quantile(x[split['train'],0],np.linspace(0,1,4)))
   knots.append(dict(version=extra or 'baseline',feature=col,unique_train_quantile_knots=k.tolist()))
   if len(k)>1:
    st=SplineTransformer(degree=3,knots=k[:,None],include_bias=False,extrapolation='linear');st.fit(x[split['train']]);blocks.append(st.transform(x))
  for aa,bb in [('n_orders','min_slack'),('n_pickups','known_wait_mean'),('n_orders','max_radius_km')]:blocks.append((d[aa]*d[bb]).to_numpy()[:,None])
  matrices['spline_'+extra]=np.column_stack(blocks)
 (out/'spline_knots.json').write_text(json.dumps(knots,indent=2),encoding='utf-8')
 configs=[]
 for family in ['linear','spline','tree']:
  for extra in [False,True]:configs.append((family+('_G' if extra else ''),family,extra))
 configs += [('linear_LB','struct','log_lb'),('linear_UB','struct','log_ub'),('linear_no_release_G','struct','no_release_log_g')]
 scores=[];trials=[];predictions=[];choices=[]
 for name,family,extra in configs:
  X=matrices['spline_'+('G' if extra else '')] if family=='spline' else d[BASE+([extra] if family=='struct' else ['log_g'] if extra else [])].to_numpy(float)
  grid=[.1,1,10] if family=='spline' else [7,15,31] if family=='tree' else [1]
  candidates=[]
  for param in grid:
   m=HistGradientBoostingClassifier(max_leaf_nodes=int(param),learning_rate=.05,max_iter=200,l2_regularization=1,min_samples_leaf=30,early_stopping=False,random_state=SEED) if family=='tree' else make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=5000 if family=='spline' else 3000,random_state=SEED))
   t=time.perf_counter()
   with warnings.catch_warnings(record=True) as ww:
    warnings.simplefilter('always');m.fit(X[split['train']],y[split['train']])
   loss=log_loss(y[split['validation']],m.predict_proba(X[split['validation']])[:,1]);candidates.append((loss,param,m))
   trials.append(dict(model=name,parameter=param,validation_log_loss=loss,fit_seconds=time.perf_counter()-t,warnings=' | '.join(str(w.message) for w in ww)))
  loss,param,m=min(candidates,key=lambda z:(z[0],z[1]));choices.append(dict(model=name,chosen_parameter=param,validation_log_loss=loss))
  for label,mask in split.items():
   p=m.predict_proba(X[mask])[:,1];scores.append(dict(model=name,split=label,n=int(mask.sum()),riders=d.loc[mask,'courier_id'].nunique(),parameter=param,**metrics(y[mask],p)))
   if label=='test':
    q=d.loc[mask,['state_id','courier_id','date','n_orders','positive_g','late_any']].copy();q['model']=name;q['probability']=p;predictions.append(q)
  print(name,'selected',param,flush=True)
 pd.DataFrame(trials).to_csv(out/'validation_candidates.csv',index=False);pd.DataFrame(choices).to_csv(out/'selected_configs.csv',index=False)
 pd.DataFrame(scores).to_csv(out/'predictive_benchmarks.csv',index=False);pred=pd.concat(predictions,ignore_index=True);pred.to_parquet(out/'test_predictions.parquet',index=False)
