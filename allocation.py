import numpy as np
from numba import njit

@njit(cache=True)
def allocate(weights,total,cap):
 out=np.zeros(len(weights));active=weights>0;remaining=total
 if np.sum(active)*cap+1e-8<total:return out,False
 for _ in range(len(weights)+1):
  s=weights[active].sum()
  if s<=0:break
  changed=False
  for j in range(len(weights)):
   if active[j] and remaining*weights[j]/s>cap+1e-10:
    out[j]=cap;active[j]=False;changed=True
  if changed:remaining=total-out.sum()
  else:
   for j in range(len(weights)):
    if active[j]:out[j]=remaining*weights[j]/s
   break
 return out,True

