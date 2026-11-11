from pathlib import Path
import itertools,json,time
import numpy as np,pandas as pd
from numba import njit
from solver import solve,bounds,evaluate_route,distance_matrix
R=Path(__file__).resolve().parents[1];P=R.parent
@njit(cache=True)
def enumerate_objectives(routes,dist,due,rel,pred,cap,tolerance=0.):
 bestsum=1e20;bestend=1e20;bestmax=1e20;bestroute=-1;valid=0
 for z in range(len(routes)):
  loss,comp=evaluate_route(routes[z],dist,due,rel,pred,.5,3.)
  if loss>=1e19:continue
  valid+=1;bestmax=min(bestmax,loss);s=0.;ok=True
  for j in range(len(due)):
   if due[j]<1e8:
    late=max(0.,comp[j]-due[j]-tolerance);s+=late
    if late>cap+1e-8:ok=False
  end=comp.max()
  if ok and (s<bestsum-1e-9 or (abs(s-bestsum)<1e-9 and end<bestend-1e-9)):
   bestsum=s;bestend=end;bestroute=z
 return bestsum,bestend,bestmax,bestroute,valid
