"""Paired errors and explicitly retrospective examples for the sum follow-up."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from week01 import style
from .experiment import ROOT
from .plots import save_figure, COLORS, LABELS


def main():
    out=ROOT.parent/'results'
    main_rows=pd.read_csv(out/'runs/baseline/per_scene.csv')
    more=pd.read_csv(out/'runs/sum_pool/per_scene.csv')
    rows=pd.concat([main_rows,more[more.variant=='sum_pool']],ignore_index=True)
    means=pd.concat([pd.read_csv(out/'runs/baseline/metrics.csv'),pd.read_csv(out/'runs/sum_pool/metrics.csv').query("variant=='sum_pool'")])
    sums=means.groupby(['split','variant'])[['ADE','FDE']].agg(['mean','std']).fillna(0)
    sums.columns=['_'.join(c) for c in sums.columns]
    test=rows[(rows.split=='test')&(rows.seed>=0)]
    base=test[test.variant=='social'].set_index(['scene_id','seed'])
    variants=['no_neighbours','small_grid','large_grid','sum_pool']
    colors={**COLORS,'sum_pool':'#AA7A00'}
    labels={**LABELS,'sum_pool':'Sum / 4 m'}
    paired=[]
    for variant in variants:
        other=test[test.variant==variant].set_index(['scene_id','seed'])
        assert other.index.equals(base.index)
        for (sid,seed),r in other.iterrows():
            paired.append({'variant':variant,'scene_id':sid,'seed':seed,'primary_id':int(r.primary_id),
                           'ADE_delta':r.ADE-base.loc[(sid,seed)].ADE,
                           'FDE_delta':r.FDE-base.loc[(sid,seed)].FDE})
    paired=pd.DataFrame(paired)
    paired.to_csv(out/'tables/paired_differences.csv',index=False)
    scene_ids=sorted(test.scene_id.unique())
    style.apply()
    fig,axes=plt.subplots(1,3,figsize=(14,6.8),gridspec_kw={'width_ratios':[1,1,1.15]})
    fig.subplots_adjust(left=.085,right=.97,bottom=.21,top=.79,wspace=.42)
    fig.suptitle('Where do neighbour changes help or hurt?',x=.07,ha='left',y=.98,fontsize=20,fontweight='bold',color='#152A3A')
    fig.text(.07,.917,'Paired test errors on the same 12 scenes | negative change is better | new sum pooling is an exploratory follow-up',fontsize=10,color='#53636B')
    for ax,metric in zip(axes[:2],['ADE','FDE']):
        matrix=paired.groupby(['scene_id','variant'])[metric+'_delta'].mean().unstack().reindex(index=scene_ids,columns=variants)
        limit=max(abs(matrix.to_numpy()).max(),.05)
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        for i in range(len(scene_ids)):
            for j in range(4):
                v=matrix.iloc[i,j]
                ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=7.5,color='white' if abs(v)>limit*.62 else '#152A3A')
        ids=[int(test[test.scene_id==sid].iloc[0].primary_id) for sid in scene_ids]
        ax.set(xticks=range(4),xticklabels=['Zero','2 m','8 m','Sum'],yticks=range(12),
               yticklabels=[f'{sid} / {ped}' for sid,ped in zip(scene_ids,ids)],title=f'Change in {metric} vs original (m)',xlabel='Neighbour modification',ylabel='Scene / primary person')
        for y in [3.5,7.5]:ax.axhline(y,c='white',lw=2)
        fig.colorbar(im,ax=ax,pad=.025,fraction=.05)
    ax=axes[2]
    order=['social','no_neighbours','small_grid','large_grid','sum_pool']
    for seed,marker in zip([42,43,44],['o','s','^']):
        vals=means[(means.split=='test')&(means.seed==seed)].set_index('variant').reindex(order).ADE
        ax.plot(range(5),vals,marker=marker,label=f'Seed {seed}',alpha=.8)
    cv=float(means[(means.split=='test')&(means.variant=='constant_velocity')].ADE.iloc[0])
    ax.axhline(cv,c='#555555',ls='--',label='Constant velocity')
    ax.set(xticks=range(5),xticklabels=['Original','Zero','2 m','8 m','Sum'],ylabel='Test ADE (m)',title='Seed variation remains important',ylim=(0,None))
    ax.legend(fontsize=8);ax.grid(alpha=.16)
    for i,ax in enumerate(axes):style.panel_label(ax,chr(97+i),x=-.2,y=1.04)
    fig.text(.07,.097,'Heatmap cells average the 3 training seeds. White separators identify the three overlapping test windows; rows are not independent experiments.',fontsize=9,color='#637078')
    fig.text(.07,.052,'The original four models remain frozen. Sum pooling was proposed after viewing those test results; these extra comparisons are descriptive, not confirmatory.',fontsize=9,color='#637078')
    save_figure(fig,out/'figures/paired_scene_errors')
    old=np.load(out/'runs/baseline/predictions.npz'); new=np.load(out/'runs/sum_pool/predictions.npz')
    seed_rows=paired[(paired.variant=='sum_pool')&(paired.seed==42)].set_index('scene_id')
    selected=[int(old['test_scene_ids'][0]),int(seed_rows.ADE_delta.idxmin()),int(seed_rows.ADE_delta.idxmax())]
    titles=['First scene (fixed)','Lowest ADE change (retrospective)','Highest ADE change (retrospective)']
    fig,axes=plt.subplots(1,3,figsize=(14,5.6))
    fig.subplots_adjust(left=.07,right=.975,top=.79,bottom=.23,wspace=.28)
    fig.suptitle('Sum pooling: trajectory examples and remaining failures',x=.07,ha='left',y=.98,fontsize=18,fontweight='bold',color='#152A3A')
    fig.text(.07,.91,'Seed 42 | 8 observed + 12 predicted | the two extreme examples are selected after evaluation, not used to tune the model',fontsize=10,color='#53636B')
    cases=[]
    for i,(ax,sid,title) in enumerate(zip(axes,selected,titles)):
        index=int(np.flatnonzero(old['test_scene_ids']==sid)[0]); h=old['test_history'][index,:,0]; t=old['test_truth'][index,:,0]
        ax.plot(h[:,0],h[:,1],'o-',color='#152A3A',ms=3,label='Observed history')
        truth=np.vstack([h[-1],t]); ax.plot(truth[:,0],truth[:,1],'.-',c='black',lw=2,label='True future')
        for name,array in [('social',old['test_social_seed42']),('sum_pool',new['test_sum_pool_seed42']),('constant_velocity',old['test_constant_velocity'])]:
            p=np.vstack([h[-1],array[index,:,0]])
            ax.plot(p[:,0],p[:,1],ls='--' if name=='constant_velocity' else '-',color=colors[name],label=labels[name],lw=1.6)
            ax.plot(p[-1,0],p[-1,1],'x',color=colors[name],ms=5)
        delta=float(seed_rows.loc[sid,'ADE_delta'])
        ax.set(title=f'{title}\nScene {sid} | ADE change {delta:+.3f} m',xlabel='World x (m)',ylabel='World y (m)')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.17);style.panel_label(ax,chr(97+i),x=-.13,y=1.10)
        cases.append({'scene_id':sid,'seed':42,'selection':title,'ADE_delta':delta})
    handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=5,bbox_to_anchor=(.52,.1),fontsize=9)
    fig.text(.07,.045,'Equal x/y scale within each panel; panel extents differ. A lower error in a selected example does not establish a general improvement.',fontsize=9,color='#637078')
    save_figure(fig,out/'figures/sum_pool_cases')
    notes={'status':'exploratory follow-up','cases':cases,'sum_seed_ADE_improved':int((means.query("split=='test' and variant=='sum_pool'").set_index('seed').ADE-means.query("split=='test' and variant=='social'").set_index('seed').ADE<0).sum())}
    (out/'audit/extension_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    print(sums.reset_index().to_string(index=False));print(json.dumps(notes))


if __name__=='__main__':main()
