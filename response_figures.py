"""Draw review figures from aggregate CSVs, with native vector and raster exports."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from PIL import Image
import zlib
import tifffile
import numpy as np

def lossless_tiff(source,target):
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(source) as im:
        w,h=im.size
        def strips():
            for y in range(0,h,32):
                yield zlib.compress(im.crop((0,y,w,min(h,y+32))).convert("RGB").tobytes(),3)
        tifffile.imwrite(target,strips(),shape=(h,w,3),dtype=np.uint8,photometric="rgb",compression="deflate",rowsperstrip=32,resolution=(1200,1200),resolutionunit="INCH")
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':10,'axes.linewidth':1.15,'xtick.direction':'in','ytick.direction':'in','xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
sand,teal,clay,gray='#C6A16B','#8AABA5','#C98262','#C0BDB8'
def run(data,out):
    get=lambda f:pd.read_csv(data/(f+'.csv'))
    s=get('scenario_comparisons');c=get('coverage').set_index('sample');p=get('priority_groups').set_index('group');e=get('exact_comparison')
    def panel():
        fig,axs=plt.subplots(2,2,figsize=(7.0,5.5),layout='constrained')
        return fig,axs.ravel()
    def bars(ax,labels,values,xlabel,fmt='.0f',colors=None):
        ids=np.arange(len(labels));ax.barh(ids,values,color=colors or sand,edgecolor='black',lw=.7,height=.62)
        ax.set_yticks(ids,labels);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_xlim(0,max(values)*1.37 or 1)
        for i,v in enumerate(values):ax.annotate(format(v,fmt),(v,i),xytext=(3,0),textcoords='offset points',va='center',fontsize=8)
    def finish(fig,axes,n,titles):
        for i,(ax,title) in enumerate(zip(axes,titles)):ax.set_title(chr(97+i)+'   '+title,loc='left',fontsize=11,fontweight='bold',pad=6)
        for ext in ['pdf','svg','png']:fig.savefig(out/f'RF{n}.{ext}',dpi=1200 if ext=='png' else 300)
        lossless_tiff(out/f'RF{n}.png',out/f'RF{n}.tiff')
        plt.close(fig)
    f,a=panel()
    bars(a[0],['Snapshot rows','Nonempty','Consistent','First snapshot','Policy sample'],[62044,43377,42018,13819,13772],'Rows or states',',.0f')
    labels=['Policy','Over 14','Later'];c=c.loc[['Policy sample','Over 14 nodes','Later snapshots']]
    x=np.arange(3)
    a[1].bar(x-.18,c.mean_orders,.36,color=sand,label='Orders');a[1].bar(x+.18,c.mean_nodes,.36,color=teal,label='Nodes');a[1].set_xticks(x,labels);a[1].set_ylabel('Mean per state');a[1].legend(frameon=False)
    bars(a[2],labels,c.median_slack,'Median minimum time to promise (min)','.1f')
    z=s.query("scenario=='Over 14 nodes'").set_index('rule').loc[['Uniform','Capped','Uncapped']]
    a[3].bar(x,z.positive,color=sand,label='Positive');a[3].bar(x,z.unknown,bottom=z.positive,color=teal,label='Undetermined');a[3].set_xticks(x,z.index);a[3].set_ylabel('Cohorts out of 47');a[3].legend(frameon=False)
    finish(f,a,1,['Sample selection','Workload','Time available','Beyond the node limit'])
    f,a=panel();p=p.loc[['Single order','Equal priorities','Mixed priorities']]
    bars(a[0],['Single','Equal, multi','Mixed, multi'],p.n,'Policy cohorts',',.0f')
    bars(a[1],['Initially positive','Improved','Worsened','Net fewer positive'],[p.initial.iloc[2],p.improved.iloc[2],p.worse.iloc[2],p.net.iloc[2]],'Mixed-priority cohorts',',.0f',[gray,teal,clay,sand])
    t=get('budget_use').iloc[0]
    bars(a[2],['Full allocation','Route-based use'],[t.full,t.trimmed],'Order-minutes',',.0f')
    bars(a[3],['Orders','Cohorts'],[100*t.zero_orders/38626,100*t.zero_cohorts/13772],'Receiving zero extension (%)','.1f')
    finish(f,a,2,['Where priorities differ','Effect within mixed cohorts','Budget use','Zero-extension decisions'])
    f,a=panel();base=s[s.scenario=='Reference'];u=base[base.rule=='Capped'];x=np.arange(4)
    for k,col,mk in [('Uniform',gray,'o'),('Capped',clay,'s'),('Uncapped',teal,'^')]:
        q=base[base.rule==k];a[0].plot(x,q.pct,marker=mk,color=col,label=k)
    a[0].set_xticks(x,['0.02','1','2','5']);a[0].set_xlabel('Decision threshold (min)');a[0].set_ylabel('Certified positive (%)');a[0].legend(frameon=False)
    ci=get('paired_intervals');a[1].errorbar(x,-ci.pp,yerr=np.vstack([ci.high-ci.pp,ci.pp-ci.low]),fmt='o',color='black',capsize=4);a[1].set_xticks(x,['0.02','1','2','5']);a[1].set_xlabel('Decision threshold (min)');a[1].set_ylabel('Uniform minus capped (pp)');a[1].set_ylim(bottom=0)
    st=s[(s.rule=='Capped') & (s.threshold==.02) & ~s.scenario.isin(['Recorded readiness available','Over 14 nodes'])]
    labels=['Reference','Longitude 30°','Longitude 50°','12 km/h','18 km/h','Service 0.25','Service 1','No ready wait','Shared stop','Rest 0','Rest 5']
    for ax,col,xlabel in [(a[2],'pp','Uniform minus capped (pp)'),(a[3],'retained_pct','Uncapped gain retained (%)')]:
        ax.plot(st[col],np.arange(len(st)),'o',color=clay if col=='pp' else teal);ax.set_yticks(np.arange(len(st)),labels,fontsize=8);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='y',lw=.35,color='#ddd')
    finish(f,a,3,['Decision thresholds','Paired 95% intervals','Operating inputs','Effect of the cap'])
    f,a=panel();q=base[base.threshold==.02].set_index('rule').loc[['Uniform','Capped','Slack rank','Route lateness','Uncapped']];labels=['Uniform','Individual','Slack rank','Route lateness','Uncapped']
    bars(a[0],labels,q.positive,'Positive cohorts',',.0f');o=get('extensions').set_index('rule').loc[q.index]
    x=np.arange(5);a[1].plot(o.p95,x,'s',color=clay,label='95th percentile');a[1].plot(o.maximum,x,'|',ms=12,color='black',label='Maximum');a[1].set_yticks(x,labels);a[1].invert_yaxis();a[1].set_xlabel('Order extension (min)');a[1].legend(frameon=False,fontsize=8,loc='center right')
    v=np.sort(e.excess.dropna());a[2].step(np.arange(1,len(v)+1),v,where='mid',color='black');a[2].set_xlabel('Rank among 42 feasible cohorts');a[2].set_ylabel('Excess over exact minimum\n(order-min)')
    t=get('timing').iloc[1:];a[3].plot(t.group.astype(int),t.p50,marker='.',color=teal,label='Median');a[3].plot(t.group.astype(int),t.p95,marker='.',color=clay,label='95th percentile');a[3].set_yscale('log');a[3].set_xlabel('Remaining nodes');a[3].set_ylabel('End-to-end time (ms)');a[3].legend(frameon=False)
    finish(f,a,4,['Allocation rules','Customer extensions','Exact resource comparison','Decision time'])
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
