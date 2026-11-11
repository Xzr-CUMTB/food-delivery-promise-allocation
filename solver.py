"""Precedence-constrained minimax tardiness with independent route validation.
All times are minutes relative to the decision epoch.
For fixed extension g, subset DP retains the earliest completion for each (visited,last).
This is sufficient under fixed travel times, release times, and constant node service.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def evaluate_route(route,dist,deadline,release,pred,service,initial_rest=0.0):
    n=len(deadline);seen=np.zeros(n,np.int8);t=initial_rest;last=n;loss=0.0
    completions=np.zeros(n)
    for q in range(n):
        j=route[q]
        if j<0 or j>=n or seen[j] or (pred[j]>=0 and not seen[pred[j]]):
            return 1e20,completions
        t=max(t+dist[last,j],release[j])+service
        completions[j]=t
        if deadline[j]<1e8:loss=max(loss,t-deadline[j])
        seen[j]=1;last=j
    return max(0.0,loss),completions

@njit(cache=True)
def greedy(dist,deadline,release,pred,service,mode=0,initial_rest=0.0):
    n=len(deadline);seen=np.zeros(n,np.int8);route=np.empty(n,np.int32)
    t=initial_rest;last=n
    for q in range(n):
        best=-1;bestscore=1e30
        for j in range(n):
            if seen[j] or (pred[j]>=0 and not seen[pred[j]]):continue
            ready=max(t+dist[last,j],release[j])+service
            due=deadline[j]
            if due>1e8:
                for k in range(n):
                    if pred[k]==j:due=min(due,deadline[k]-dist[j,k]-service)
            if mode==0:score=due+0.05*ready
            elif mode==1:score=ready
            else:score=due-ready+0.15*dist[last,j]
            if score<bestscore:bestscore=score;best=j
        route[q]=best;seen[best]=1;t=max(t+dist[last,best],release[best])+service;last=best
    loss,comp=evaluate_route(route,dist,deadline,release,pred,service,initial_rest)
    return loss,route,comp

@njit(cache=True)
def bounds(dist,deadline,release,pred,service,initial_rest=0.0):
    n=len(deadline);lb=0.0
    for j in range(n):
        if deadline[j]>1e8:continue
        if pred[j]>=0:
            k=pred[j];early=max(max(initial_rest+dist[n,k],release[k])+service+dist[k,j],release[j])+service
        else:early=max(initial_rest+dist[n,j],release[j])+service
        lb=max(lb,early-deadline[j])
    ub,route,comp=greedy(dist,deadline,release,pred,service,0,initial_rest)
    for mode in range(1,3):
        x,r,c=greedy(dist,deadline,release,pred,service,mode,initial_rest)
        if x<ub:ub=x;route=r;comp=c
    return max(0.,lb),ub,route

@njit(cache=True)
def feasible_dp(dist,deadline,release,pred,service,extension,initial_rest=0.0):
    n=len(deadline);M=1<<n
    table=np.full((M,n),1e20)
    parent=np.full((M,n),-1,np.int16)
    for j in range(n):
        if pred[j]>=0:continue
        t=max(initial_rest+dist[n,j],release[j])+service
        if t<=deadline[j]+extension+1e-8:table[1<<j,j]=t
    for mask in range(1,M):
        for last in range(n):
            t=table[mask,last]
            if t>=1e19:continue
            for j in range(n):
                if mask&(1<<j):continue
                if pred[j]>=0 and not(mask&(1<<pred[j])):continue
                nt=max(t+dist[last,j],release[j])+service
                nm=mask|(1<<j)
                if nt<=deadline[j]+extension+1e-8 and nt<table[nm,j]:
                    table[nm,j]=nt;parent[nm,j]=last
    end=-1;best=1e20
    for j in range(n):
        if table[M-1,j]<best:best=table[M-1,j];end=j
    route=np.full(n,-1,np.int32)
    if end<0:return False,route
    mask=M-1;cur=end
    for q in range(n-1,-1,-1):
        route[q]=cur;old=cur;cur=parent[mask,old];mask^=1<<old
    return True,route

@njit(cache=True)
def solve(dist,deadline,release,pred,service,tol=.02,max_nodes=14,initial_rest=0.0,warm=-1.0):
    lb,ub,route=bounds(dist,deadline,release,pred,service,initial_rest)
    calls=0
    if ub<=1e-8:return 0.,0.,route,0,True
    if len(deadline)>max_nodes:return lb,ub,route,0,False
    if ub-lb<=tol:return lb,ub,route,0,True
    # Test the analytical bound first: often it is attainable.
    feasible,r=feasible_dp(dist,deadline,release,pred,service,lb,initial_rest);calls+=1
    if feasible:return lb,lb,r,calls,True
    if warm>lb+tol and warm<ub-tol:
        yes,r=feasible_dp(dist,deadline,release,pred,service,warm,initial_rest);calls+=1
        if yes:ub=warm;route=r
        else:lb=warm
    while ub-lb>tol:
        mid=(lb+ub)*.5
        yes,r=feasible_dp(dist,deadline,release,pred,service,mid,initial_rest);calls+=1
        if yes:ub=mid;route=r
        else:lb=mid
    return lb,ub,route,calls,True

def distance_matrix(xy,speed=15.,multiplier=1.,latitude_only=False):
    delta=xy[:,None,:]-xy[None,:,:]
    if latitude_only:km=np.abs(delta[:,:,1])*(110./110.57)
    else:km=np.sqrt((delta**2).sum(-1))*multiplier
    return np.ascontiguousarray(km*60/speed,dtype=np.float64)

