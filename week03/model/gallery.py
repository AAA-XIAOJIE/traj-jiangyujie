"""Complete fixed-seed primary gallery and all-agent descriptive comparisons."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from week01 import style
from .experiment import ROOT


def main():
    out=ROOT.parent/'results'; gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    base=np.load(out/'predictions.npz');summed=np.load(out/'sum_pool/predictions.npz');direction=np.load(out/'directional_sum/predictions.npz')
    arrays={v:base[f'test_{v}_seed42'] for v in ['social','no_neighbours','small_grid','large_grid']}
    arrays['sum_pool']=summed['test_sum_pool_seed42'];arrays['directional_sum']=direction['test_directional_sum_seed42']
    names={'social':'Original Social / 4 m','no_neighbours':'Zero neighbours','small_grid':'Social / 2 m','large_grid':'Social / 8 m','sum_pool':'Equal-weight sum / 4 m','directional_sum':'Heading-weighted sum / 4 m'}
    truth=base['test_truth'];history=base['test_history'];scene_ids=base['test_scene_ids']
    style.apply();plt.rcParams.update({'font.size':9})
    primary_ids=[1,17,33,49]*3
    bounds=[]
    for i in range(12):
        points=np.concatenate([history[i,:,0],truth[i,:,0]]+[a[i,:,0] for a in arrays.values()])
        lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;span=max((hi-lo).max()*1.18,.5)
        bounds.append((center,span))
    for variant,pred in arrays.items():
        fig,axes=plt.subplots(3,4,figsize=(13.6,10.8))
        fig.subplots_adjust(left=.06,right=.98,top=.89,bottom=.10,wspace=.27,hspace=.42)
        fig.suptitle(names[variant]+' — all 12 test scenes',x=.06,ha='left',y=.977,fontsize=20,fontweight='bold',color='#152A3A')
        fig.text(.06,.932,'Seed 42 | 8 historical points + 12 future points | identical per-scene axis limits across all six model sheets',fontsize=11,color='#53636B')
        for i,ax in enumerate(axes.flat):
            h=history[i,:,0];t=np.vstack([h[-1],truth[i,:,0]]);p=np.vstack([h[-1],pred[i,:,0]])
            ax.plot(h[:,0],h[:,1],'o-',c='#7C8790',ms=2.6,lw=1,label='Observed history')
            ax.plot(t[:,0],t[:,1],'o-',c='#152A3A',ms=2.6,lw=1.8,label='True future')
            ax.plot(p[:,0],p[:,1],'.-',c='#D55E00',ms=3,lw=1.5,label='Predicted future')
            ax.scatter(*h[-1],s=25,c='white',edgecolors='#152A3A',zorder=5)
            ax.plot(p[-1,0],p[-1,1],'x',c='#D55E00',ms=6)
            err=np.linalg.norm(pred[i,:,0]-truth[i,:,0],axis=-1)
            c,span=bounds[i]
            ax.set(xlim=(c[0]-span/2,c[0]+span/2),ylim=(c[1]-span/2,c[1]+span/2),
                   title=f'Scene {scene_ids[i]} / person {primary_ids[i]}\nADE {err.mean():.3f} m | FDE {err[-1]:.3f} m')
            ax.set_aspect('equal');ax.grid(alpha=.17)
            if i>=8:ax.set_xlabel('World x (m)')
            if i%4==0:ax.set_ylabel('World y (m)')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.52,.04))
        fig.text(.06,.022,'Rows are the three test windows; columns are the four fixed primary pedestrians. No future positions enter the predictor.',fontsize=9,color='#637078')
        fig.savefig(gallery/f'{variant}_all_scenes.png',dpi=250);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,12),sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.97,top=.90,bottom=.09,wspace=.12,hspace=.22)
    fig.suptitle('All 64 agents: measured and predicted future paths',x=.07,ha='left',y=.977,fontsize=18,fontweight='bold',color='#152A3A')
    fig.text(.07,.938,'One scene per time window (primary ID 1), seed 42 | neighbours are predicted jointly; official ADE/FDE still score only the primary',fontsize=9,color='#53636B')
    for row,index in enumerate([0,4,8]):
        for col,variant in enumerate(['social','sum_pool','directional_sum']):
            ax=axes[row,col]
            for j in range(64):
                h=history[index,:,j];t=np.vstack([h[-1],truth[index,:,j]]);p=np.vstack([h[-1],arrays[variant][index,:,j]])
                ax.plot(h[:,0],h[:,1],c='#AAB2B8',lw=.5,alpha=.5)
                ax.plot(t[:,0],t[:,1],c='#009E73',lw=.8,alpha=.8)
                ax.plot(p[:,0],p[:,1],c='#D55E00',lw=.65,alpha=.65)
            ax.set(title=f'{names[variant]}\nScene {scene_ids[index]}',xlim=(-12,12),ylim=(-12,12),aspect='equal')
            ax.grid(alpha=.15)
            if row==2:ax.set_xlabel('World x (m)')
            if col==0:ax.set_ylabel('World y (m)')
    fig.legend([Line2D([0],[0],c=c) for c in ['#AAB2B8','#009E73','#D55E00']],['Observed history','True future','Predicted future'],loc='lower center',ncol=3,bbox_to_anchor=(.52,.035))
    fig.savefig(gallery/'all_64_agents.png',dpi=250);plt.close(fig)
    metrics=pd.concat([pd.read_csv(out/'metrics.csv'),pd.read_csv(out/'sum_pool/metrics.csv').query("variant=='sum_pool'"),pd.read_csv(out/'directional_sum/metrics.csv').query("variant=='directional_sum'")])
    summary=metrics.groupby(['split','variant'])[['ADE','FDE']].agg(['mean','std']).fillna(0);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'extended_summary.csv',index=False)
    pairs=[]
    a=metrics.query("split=='test' and variant=='sum_pool'").set_index('seed')
    b=metrics.query("split=='test' and variant=='directional_sum'").set_index('seed')
    for seed in [42,43,44]:pairs.append({'seed':seed,'ADE_direction_minus_sum':float(b.loc[seed].ADE-a.loc[seed].ADE),'FDE_direction_minus_sum':float(b.loc[seed].FDE-a.loc[seed].FDE)})
    (out/'direction_effects.json').write_text(json.dumps(pairs,indent=2),encoding='utf8')
    text='# 全部测试轨迹图\n\n每张图展示种子 42 的全部 12 个测试场景，灰色为 8 点历史、深色为真实未来、橙色为预测未来。相同场景在六张图中坐标范围一致；三个种子的指标均完整保留。\n\n'
    for v,name in names.items():text+=f'## {name}\n\n![{name}]({v}_all_scenes.png)\n\n'
    text+='## 全部 64 人\n\n以下是三个测试时间窗口的全场辅助预测，不将邻居混入主要行人的 ADE/FDE。\n\n![全部 64 人](all_64_agents.png)\n'
    (gallery/'README.md').write_text(text,encoding='utf8')
    print(summary.reset_index().to_string(index=False));print('Seven complete trajectory sheets generated.');print(json.dumps(pairs))


if __name__=='__main__':main()
