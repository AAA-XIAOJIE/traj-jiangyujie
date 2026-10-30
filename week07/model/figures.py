"""Scientific 3D trajectory views and synchronous animation from saved arrays."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from week01.style import apply

COLORS={'apf':'#D55E00','vo':'#0072B2','cbf':'#009E73'}
LABELS={'apf':'Velocity potential field','vo':'3D velocity-obstacle sampling','cbf':'Joint barrier QP + circulation'}
SHORT={'apf':'APF','vo':'VO sampling','cbf':'Barrier QP'}


def decorate3d(ax,radius):
    lim=radius*1.12
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=35)
    ax.tick_params(labelsize=7,pad=0)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.fill=False
        axis._axinfo['grid'].update(color=(.6,.65,.7,.22))


def identity_colors(initial):
    return plt.get_cmap('viridis')((initial[:,2]/np.linalg.norm(initial,axis=1)+1)/2)


def save(fig,path):
    fig.savefig(path,dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_figures(cfg,metrics,traces,out):
    apply()
    names=cfg['models'];initial=traces[names[0]]['positions'][0]
    colors=identity_colors(initial)
    fig=plt.figure(figsize=(15,9.3),layout='constrained')
    grid=fig.add_gridspec(2,3,height_ratios=[1.5,1])
    fig.suptitle('100 UAVs | one 3D antipodal conflict, three velocity controllers',fontsize=19,fontweight='bold',color='#173243')
    for j,name in enumerate(names):
        t=traces[name];row=metrics[metrics.model==name].iloc[0]
        ax=fig.add_subplot(grid[0,j],projection='3d')
        for i in range(cfg['agents']):
            ax.plot(*t['positions'][:,i].T,color=colors[i],alpha=.5,lw=.65)
        ax.scatter(*initial.T,s=7,c=colors,alpha=.8)
        ax.scatter(*t['goals'].T,s=10,facecolors='none',edgecolors='#777777',linewidths=.4)
        decorate3d(ax,cfg['sphere_radius_m'])
        ax.set_title(f"{chr(97+j)}  {LABELS[name]}\n{int(row.arrived)}/100 arrived | {int(row.violating_unique_pairs)} violating pairs",fontsize=11,color=COLORS[name],pad=9)
    axes=[fig.add_subplot(grid[1,j]) for j in range(3)]
    for name in names:
        t=traces[name];color=COLORS[name]
        axes[0].plot(t['times'][1:],t['arrival_counts'],color=color,label=SHORT[name])
        axes[1].plot(t['times'][1:],t['minimum_separation'],color=color,lw=1.3)
        lengths=np.linalg.norm(np.diff(t['positions'],axis=0),axis=2).cumsum(axis=0).mean(axis=1)
        axes[2].plot(t['times'][1:],lengths,color=color)
    axes[0].set(title='d  Arrival over time',ylabel='Arrived UAVs',ylim=(0,104));axes[0].legend(frameon=False,fontsize=9)
    axes[1].axhline(1,color='#A83232',ls='--',lw=1,label='1 m requirement')
    axes[1].set(title='e  Minimum separation (0-4 m detail)',ylabel='Center-to-center distance (m)',ylim=(0,4));axes[1].legend(frameon=False,fontsize=8)
    axes[2].axhline(39,color='#888888',ls=':',lw=1,label='39 m direct-path lower bound')
    axes[2].set(title='f  Mean cumulative distance',ylabel='Mean path length (m)');axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.set_xlabel('Time (s)');ax.set_xlim(0,max(t['times'][-1] for t in traces.values())+1);ax.grid(alpha=.17)
    save(fig,out/'overview.png')
    fig,axs=plt.subplots(3,3,figsize=(13,12),layout='constrained')
    projections=[(0,1,'x','y'),(0,2,'x','z'),(1,2,'y','z')]
    for row,name in enumerate(names):
        for col,(u,v,ul,vl) in enumerate(projections):
            ax=axs[row,col];p=traces[name]['positions']
            for i in range(cfg['agents']):
                ax.plot(p[:,i,u],p[:,i,v],color=colors[i],lw=.65,alpha=.65)
            ax.scatter(initial[:,u],initial[:,v],c=colors,s=6)
            ax.set(xlim=(-22,22),ylim=(-22,22),xlabel=f'{ul} (m)',ylabel=f'{vl} (m)',aspect='equal',title=f'{SHORT[name]} | {ul}{vl} projection')
            ax.grid(alpha=.12)
    fig.suptitle('All 100 trajectories | complementary projections reveal 3D structure',fontsize=17,fontweight='bold')
    save(fig,out/'trajectory_projections.png')
    closest_encounters(cfg,traces,out)
    metrics_figure(metrics,out)


def worst_pair(trace):
    p=trace['positions'];dt=trace['times'][1]-trace['times'][0]
    i,j=np.triu_indices(p.shape[1],1);best=(np.inf,None)
    for k in range(len(p)-1):
        q=p[k,i]-p[k,j];dv=(p[k+1,i]-p[k,i]-p[k+1,j]+p[k,j])/dt
        tau=np.clip(-np.sum(q*dv,axis=1)/np.maximum(np.sum(dv*dv,axis=1),1e-12),0,dt)
        d=np.linalg.norm(q+tau[:,None]*dv,axis=1);index=np.argmin(d)
        if d[index]<best[0]:best=(d[index],(int(i[index]),int(j[index]),k,float(tau[index])))
    return best


def closest_encounters(cfg,traces,out):
    fig=plt.figure(figsize=(14,8),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1.25,1])
    fig.suptitle('Closest encounters | body spheres and interval-level checks',fontsize=17,fontweight='bold')
    uu=np.linspace(0,2*np.pi,24);vv=np.linspace(0,np.pi,14)
    sphere=np.array([np.outer(np.cos(uu),np.sin(vv)),np.outer(np.sin(uu),np.sin(vv)),np.outer(np.ones_like(uu),np.cos(vv))])*.5
    for col,name in enumerate(cfg['models']):
        trace=traces[name];d,(i,j,k,tau)=worst_pair(trace)
        p=trace['positions'];dt=cfg['dt_s'];when=trace['times'][k]+tau
        centers=p[k,[i,j]]+tau/dt*(p[k+1,[i,j]]-p[k,[i,j]])
        midpoint=centers.mean(axis=0)
        ax=fig.add_subplot(grid[0,col],projection='3d')
        for index,color in zip([i,j],['#0072B2','#D55E00']):
            c=centers[0 if index==i else 1]
            ax.plot_surface(sphere[0]+c[0],sphere[1]+c[1],sphere[2]+c[2],color=color,alpha=.65,linewidth=0,shade=True)
            start=max(0,k-10);end=min(len(p),k+12)
            ax.plot(*p[start:end,index].T,color=color,lw=1.5)
        for axis,value in zip(['x','y','z'],midpoint):getattr(ax,'set_'+axis+'lim')(value-1.5,value+1.5)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=35)
        ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'{SHORT[name]}: {d:.3f} m\nIDs {i}, {j} at t = {when:.3f} s')
        ax.tick_params(labelsize=7)
        bx=fig.add_subplot(grid[1,col])
        sep=np.linalg.norm(p[:,i]-p[:,j],axis=1)
        bx.plot(trace['times'],sep,color=COLORS[name],label='Sampled distance of this pair')
        bx.scatter([when],[d],color='#A83232',zorder=5,s=25,label='Continuous-step minimum')
        bx.axhline(1,color='#A83232',ls='--',lw=1)
        bx.axvline(when,color='#888888',ls=':',lw=.8)
        bx.set(xlim=(max(0,when-3),when+3),ylim=(0,4),xlabel='Time (s)',ylabel='Center separation (m)')
        bx.grid(alpha=.15)
        if col==0:bx.legend(frameon=False,fontsize=8)
    save(fig,out/'closest_encounters.png')


def metrics_figure(metrics,out):
    names=list(metrics.model);colors=[COLORS[n] for n in names];labels=[SHORT[n] for n in names]
    fig,axes=plt.subplots(1,5,figsize=(15,3.7),layout='constrained')
    spec=[('arrival_rate','Arrival (%)',100),('minimum_separation_m','Minimum separation (m)',1),
          ('violating_unique_pairs','Unique violating pairs',1),('all_arrival_time_s','All-arrival time (s)',1),('mean_path_length_m','Mean path length (m)',1)]
    for ax,(column,title,factor) in zip(axes,spec):
        values=np.asarray(metrics[column],dtype=float)*factor
        finite=values[np.isfinite(values)]
        top=float(finite.max()) if len(finite) and finite.max()>0 else 1.
        bars=ax.bar(labels,np.where(np.isfinite(values),values,0.),color=colors,width=.6)
        for bar,value in zip(bars,values):
            center=bar.get_x()+bar.get_width()/2
            if np.isfinite(value):
                ax.text(center,value+top*.035,f'{value:.2f}' if column not in ('arrival_rate','violating_unique_pairs') else f'{value:.0f}',ha='center',fontsize=9)
            else:
                label='Not all\narrived' if column=='all_arrival_time_s' else 'Unavailable'
                ax.text(center,.08,label,ha='center',fontsize=8,transform=ax.get_xaxis_transform())
        ax.set_title(title,fontsize=10);ax.set_ylim(0,top*1.25)
        ax.tick_params(axis='x',labelsize=8,rotation=25);ax.grid(axis='y',alpha=.15)
        if column=='minimum_separation_m':ax.axhline(1,color='#A83232',ls='--',lw=1)
    fig.suptitle('Arrival is not sufficient: safety and completion must be reported together',fontsize=15,fontweight='bold')
    save(fig,out/'metrics.png')


def make_animation(cfg,traces,out):
    names=cfg['models'];initial=traces[names[0]]['positions'][0];colors=identity_colors(initial)
    fig=plt.figure(figsize=(12,4.9),facecolor='white')
    fig.subplots_adjust(left=.055,right=.97,bottom=.10,top=.81,wspace=.12)
    fig.suptitle('100 UAVs flying to antipodal goals | synchronized 3D comparison',fontsize=15,fontweight='bold',y=.97)
    time_label=fig.text(.5,.895,'',ha='center',fontsize=11)
    artists=[]
    for index,name in enumerate(names):
        ax=fig.add_subplot(1,3,index+1,projection='3d');decorate3d(ax,cfg['sphere_radius_m'])
        ax.set_title(SHORT[name],color='#173243',fontsize=12,pad=10)
        ax.scatter(*traces[name]['goals'].T,s=4,c='#AAAAAA',alpha=.28)
        points=ax.scatter(*initial.T,s=11,c=colors,depthshade=False)
        lines=Line3DCollection(np.repeat(initial[:,None,:],2,axis=1),colors=colors,linewidths=.75,alpha=.55);ax.add_collection3d(lines)
        text=ax.text2D(.03,.95,'',transform=ax.transAxes,fontsize=9)
        artists.append((points,lines,text))
    end=max(t['times'][-1] for t in traces.values())
    times=np.r_[np.arange(0,end,.2),end]
    def update(time):
        time_label.set_text(f't = {time:05.1f} s | safety separation: 1 m | playback: approximately 2x')
        for name,(points,lines,text) in zip(names,artists):
            trace=traces[name];k=min(int(round(time/cfg['dt_s'])),len(trace['times'])-1)
            points._offsets3d=tuple(trace['positions'][k].T)
            lines.set_segments(np.transpose(trace['positions'][max(0,k-20):k+1],(1,0,2)))
            arrived=int(np.isfinite(trace['arrival_times']).sum()) if k==len(trace['times'])-1 else int(np.sum(trace['arrival_times']<=time))
            text.set_text(f'Arrived: {arrived}/100')
        return []
    movie=FuncAnimation(fig,update,frames=times,interval=100,blit=False)
    movie.save(out/'comparison.gif',writer=PillowWriter(fps=10),dpi=95)
    plt.close(fig)
    print('Saved synchronized animation',flush=True)
