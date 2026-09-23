"""Scientific figures from saved predictions; never re-fit or select on test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from week01 import style
from .experiment import ROOT, load_scenes

COLORS = {'social':'#0072B2','no_neighbours':'#D55E00','small_grid':'#009E73',
          'large_grid':'#CC79A7','constant_velocity':'#555555'}
LABELS = {'social':'Social / 4 m','no_neighbours':'Zero neighbours',
          'small_grid':'Social / 2 m','large_grid':'Social / 8 m',
          'constant_velocity':'Constant velocity'}
SHORT = ['Social\n4 m','Zero\nneighbours','Social\n2 m','Social\n8 m','Constant\nvelocity']


def save_figure(fig, base):
    """Week 01 visual style with Python 3.9-compatible vector export."""
    fig.savefig(base.with_suffix('.png'), dpi=300)
    fig.savefig(base.with_suffix('.pdf'), metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(base.with_suffix('.svg'), metadata={'Date':None})
    plt.close(fig)


def neighbour_audit():
    rows = []
    examples = []
    for split in ['train','val','test']:
        for sid, paths, xy in load_scenes(split):
            for t in range(8):
                relative = xy[t, 1:] - xy[t, 0]
                for side in [2.,4.,8.]:
                    valid = ((relative >= -side/2) & (relative < side/2)).all(axis=1)
                    cells = np.floor((relative[valid] + side/2) / (side/4)).astype(int)
                    counts = np.bincount(cells[:,0]*4+cells[:,1], minlength=16)
                    rows.append({'split':split,'scene_id':sid,'observation_index':t,'side_m':side,
                                 'visible_neighbours':int(valid.sum()),
                                 'occupied_cells':int((counts>0).sum()),
                                 'multi_neighbour_cells':int((counts>1).sum())})
            if split == 'train':
                r = xy[7,1:] - xy[7,0]
                examples.append((int(((r>=-2)&(r<2)).all(axis=1).sum()), sid, xy[7]-xy[7,0]))
    return pd.DataFrame(rows), max(examples, key=lambda item:(item[0], -item[1]))


def draw_path(ax, saved, scene_index, variants, title):
    history = saved['test_history'][scene_index, :, 0]
    truth = saved['test_truth'][scene_index, :, 0]
    ax.plot(history[:,0], history[:,1], 'o-', color='#152A3A', ms=3, label='Observed (8)')
    actual = np.vstack([history[-1], truth])
    ax.plot(actual[:,0], actual[:,1], 'o-', color='black', ms=2, lw=2, label='True future (12)')
    for variant in variants:
        key = 'test_constant_velocity' if variant == 'constant_velocity' else f'test_{variant}_seed42'
        pred = np.vstack([history[-1], saved[key][scene_index,:,0]])
        ax.plot(pred[:,0], pred[:,1], '--' if variant=='constant_velocity' else '-', color=COLORS[variant],
                lw=1.5, label=LABELS[variant])
        ax.plot(pred[-1,0], pred[-1,1], 'x', color=COLORS[variant], ms=5)
    ax.scatter(*history[-1], s=32, facecolors='white', edgecolors='#152A3A', zorder=5)
    ax.set(title=title, xlabel='World x (m)', ylabel='World y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.17)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT.parent/'results/runs/baseline')
    args = parser.parse_args()
    out = args.results
    report = ROOT.parent/'results' if out.resolve() == (ROOT.parent/'results/runs/baseline').resolve() else out
    figures, tables, audit_dir = (report/name for name in ['figures', 'tables', 'audit'])
    for directory in (figures, tables, audit_dir): directory.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(out/'metrics.csv')
    curves = pd.read_csv(out/'learning_curves.csv')
    per = pd.read_csv(out/'per_scene.csv')
    saved = np.load(out/'predictions.npz')
    variants = list(COLORS)
    style.apply()
    plt.rcParams.update({'font.size':9,'axes.titlesize':10})
    audit, example = neighbour_audit()
    audit.to_csv(tables/'neighbour_audit.csv', index=False)
    diagnostics = []
    for split in ['train','val','test']:
        scenes = load_scenes(split)
        speeds = [np.linalg.norm(np.diff(xy[:,0],axis=0),axis=-1).mean()*12.5 for _,_,xy in scenes]
        diagnostics.append({'split':split,'scenes':len(scenes),'time_windows':len(scenes)//4,
                            'mean_primary_step_speed_m_s':float(np.mean(speeds)),
                            'definition':'mean over primary tracks and all 19 intervals per 20-point window; descriptive only'})
    pd.DataFrame(diagnostics).to_csv(tables/'split_diagnostics.csv',index=False)
    grouped = metrics.groupby(['split','variant'], sort=False)[['ADE','FDE']].agg(['mean','std']).fillna(0)
    summary = []
    for (split, variant), r in grouped.iterrows():
        summary.append({'split':split,'variant':variant,'ADE_mean':r[('ADE','mean')],
                        'ADE_sd':r[('ADE','std')],'FDE_mean':r[('FDE','mean')],'FDE_sd':r[('FDE','std')]})
    pd.DataFrame(summary).to_csv(tables/'summary.csv', index=False)
    fig, axes = plt.subplots(2,3,figsize=(13.4,8.4))
    fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.86,wspace=.32,hspace=.52)
    fig.suptitle('Does neighbour information help Social LSTM?',x=.07,ha='left',y=.98,fontsize=19,fontweight='bold',color='#152A3A')
    fig.text(.07,.927,'Teacher code | 8 observed + 12 predicted | 4 matched models × 3 seeds × 10 epochs',fontsize=11,color='#53636B')
    for ax, metric, label in [(axes[0,0],'ADE','Average displacement error'),(axes[0,1],'FDE','Final displacement error')]:
        for i, variant in enumerate(variants):
            values = metrics[(metrics.split=='test')&(metrics.variant==variant)][metric].to_numpy()
            ax.bar(i,values.mean(),color=COLORS[variant],alpha=.72,width=.62)
            if len(values)>1:
                ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='none',ecolor='#25343C',capsize=3,lw=.8)
                ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color='#25343C',s=14,zorder=4)
            top = max(values.max(), values.mean() + (values.std(ddof=1) if len(values)>1 else 0))
            ax.text(i,top+(.025 if metric=='ADE' else .045),f'{values.mean():.3f}',ha='center',fontsize=8)
        ax.set_xticks(range(5),SHORT)
        ax.set(title=label,ylabel=f'Test {metric} (m)')
        ax.set_ylim(0, metrics[metrics.split=='test'][metric].max()*1.24);ax.grid(axis='y',alpha=.15)
    ax=axes[0,2]
    for v in variants[:-1]:
        stats=curves[curves.variant==v].groupby('epoch').ADE.agg(['mean','std'])
        ax.plot(stats.index,stats['mean'],color=COLORS[v],label=LABELS[v])
        ax.fill_between(stats.index,(stats['mean']-stats['std']).to_numpy(),(stats['mean']+stats['std']).to_numpy(),color=COLORS[v],alpha=.1)
    ax.set(title='Validation chooses each checkpoint',xlabel='Training epoch',ylabel='Validation ADE (m)',xticks=[1,3,5,7,10])
    ax.legend(fontsize=7,loc='upper right');ax.grid(alpha=.15)
    sid=int(saved['test_scene_ids'][0])
    draw_path(axes[1,0],saved,0,variants,f'Fixed first test scene {sid} / seed 42')
    axes[1,0].legend(fontsize=6.3,loc='best',framealpha=.9)
    ax=axes[1,1]
    truth=saved['test_truth'][:,:,0]
    for v in variants:
        keys=['test_constant_velocity'] if v=='constant_velocity' else [f'test_{v}_seed{s}' for s in [42,43,44]]
        error=np.stack([np.linalg.norm(saved[k][:,:,0]-truth,axis=-1) for k in keys]).mean(axis=(0,1))
        ax.plot(np.arange(1,13)*.08,error,color=COLORS[v],label=LABELS[v])
    ax.set(title='Error accumulates during rollout',xlabel='Future time from last observation (s)',ylabel='Mean position error (m)')
    ax.grid(alpha=.15)
    ax=axes[1,2]
    for i,side in enumerate([2.,4.,8.]):
        vals=audit[audit.side_m==side].groupby('split').visible_neighbours.mean().reindex(['train','val','test'])
        ax.bar(np.arange(3)+(i-1)*.24,vals,width=.22,label=f'{side:g} m square',color=[COLORS['small_grid'],COLORS['social'],COLORS['large_grid']][i])
    ax.set(xticks=np.arange(3),xticklabels=['Train','Validation','Test'],ylabel='Mean visible neighbours',title='Different phases expose different neighbours')
    ax.legend(fontsize=7);ax.grid(axis='y',alpha=.15)
    for i,ax in enumerate(axes.flat):style.panel_label(ax,chr(97+i),x=-.13,y=1.055)
    fig.text(.07,.052,'Bars: mean across 3 training seeds; error bars/shading: sample SD, not confidence intervals. CV is deterministic. All models score identical test scenes.',fontsize=8,color='#637078')
    fig.text(.07,.028,'Single circle experiment; overlapping windows and the same participants across time splits. A 12-step forecast spans 0.96 s; this is a classroom comparison.',fontsize=8,color='#637078')
    save_figure(fig,figures/'overview')
    # All four people in the first test window; chosen in advance, not by error.
    fig,axes=plt.subplots(2,2,figsize=(11.5,9))
    fig.subplots_adjust(left=.08,right=.97,top=.88,bottom=.13,wspace=.27,hspace=.30)
    fig.suptitle('Same history, different neighbour information',x=.08,ha='left',y=.975,fontsize=18,fontweight='bold',color='#152A3A')
    fig.text(.08,.925,'First test window: all four fixed primary pedestrians | seed 42 | cross = predicted endpoint',color='#53636B',fontsize=10)
    for i,ax in enumerate(axes.flat):
        sid=int(saved['test_scene_ids'][i])
        row=per[(per.split=='test')&(per.variant=='social')&(per.seed==42)&(per.scene_id==sid)].iloc[0]
        draw_path(ax,saved,i,variants,f'Person {int(row.primary_id)} / scene {sid}')
        style.panel_label(ax,chr(97+i),x=-.12,y=1.01)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,bbox_to_anchor=(.52,.048))
    fig.text(.08,.023,'Each panel has equal x/y scale. Panel extents differ. No true future of any neighbour is supplied to the forecasting model.',fontsize=8,color='#637078')
    save_figure(fig,figures/'trajectory_comparison')
    # Explain what the actual grid observes, using a training scene only.
    count,sid,relative=example
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.07,right=.97,top=.78,bottom=.19,wspace=.20)
    fig.suptitle('What changes when the neighbour grid changes?',x=.07,ha='left',y=.98,fontsize=18,fontweight='bold',color='#152A3A')
    fig.text(.07,.887,f'Training scene {sid}, last observed point | each grid has 4 × 4 cells | blue points = geometrically included neighbours',fontsize=10,color='#53636B')
    for i,(ax,side) in enumerate(zip(axes,[2.,4.,8.])):
        r=relative[1:]
        inside=((r>=-side/2)&(r<side/2)).all(axis=1)
        ax.scatter(r[~inside,0],r[~inside,1],c='#C1C6CA',s=12)
        ax.scatter(r[inside,0],r[inside,1],c='#0072B2',s=22,zorder=3)
        ax.scatter(0,0,marker='*',s=90,color='#D55E00',zorder=5)
        for edge in np.linspace(-side/2,side/2,5):
            ax.plot([edge,edge],[-side/2,side/2],c='#84949F',lw=.6)
            ax.plot([-side/2,side/2],[edge,edge],c='#84949F',lw=.6)
        ax.set(title=f'{side:g} m square / {inside.sum()} neighbours',xlim=(-5,5),ylim=(-5,5),xlabel='Relative x (m)')
        ax.set_aspect('equal');style.panel_label(ax,chr(97+i),x=-.12,y=1.02)
    axes[0].set_ylabel('Relative y (m)')
    fig.text(.07,.085,'Social grid: relative positions determine cells; compressed neighbour hidden states fill cells; the embedded grid enters the LSTM.',fontsize=8,color='#637078')
    fig.text(.07,.043,'Zero-neighbour experiment supplies an all-zero grid before embedding. Teacher code uses indexed assignment for shared cells, not sum pooling.',fontsize=8,color='#637078')
    save_figure(fig,figures/'neighbour_information')
    # Summary includes paired seed effects, never a significance claim.
    test=metrics[(metrics.split=='test')&(metrics.variant!='constant_velocity')]
    effects=[]
    for v in variants[1:-1]:
        base=test[test.variant=='social'].set_index('seed')
        other=test[test.variant==v].set_index('seed')
        effects.append({'variant':v,'ADE_delta_mean':float((other.ADE-base.ADE).mean()),
                        'ADE_improved_seeds':int((other.ADE<base.ADE).sum()),
                        'FDE_delta_mean':float((other.FDE-base.FDE).mean()),
                        'FDE_improved_seeds':int((other.FDE<base.FDE).sum())})
    (audit_dir/'figure_notes.json').write_text(json.dumps({'trajectory_scenes':saved['test_scene_ids'][:4].tolist(),
        'trajectory_seed':42,'trajectory_selection':'first test window, all four primary people, predeclared',
        'grid_example_train_scene':sid,'paired_effects':effects},indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('Three figures generated from saved predictions.')


if __name__=='__main__':main()
