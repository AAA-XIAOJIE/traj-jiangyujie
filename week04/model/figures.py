"""Scientific figures from the computed points, with shared comparison axes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

from week01 import style
from .measures import local_trend

NAMES = {"edie":"Edie space-time", "detector":"Virtual point detector",
         "voronoi":"1D Voronoi", "kernel":"Spatial triangular kernel"}
COLORS = {"edie":"#0072B2", "detector":"#D55E00", "voronoi":"#009E73", "kernel":"#8D68AC"}


def save(fig, output, name):
    fig.savefig(output/f"{name}.png", dpi=220, facecolor="white")
    plt.close(fig)


def trend(ax, frame, color="#243743", label=None, alpha=1, bandwidth=20., minimum=4):
    xx, yy = local_trend(frame.k_veh_km, frame.q_veh_h, bandwidth, minimum)
    ax.plot(xx, yy, color=color, lw=1.6, alpha=alpha, label=label)


def limits(frame):
    return (0, np.ceil(frame.k_veh_km.max()/10)*10+10), (0, np.ceil(frame.q_veh_h.max()/200)*200+200)


def plot_all(data, segments, points, summary, audit, out):
    from functools import partial
    config = audit["analysis_config"]
    fit = partial(trend, bandwidth=config["trend_bandwidth_veh_km"], minimum=config["trend_min_points"])
    style.apply()
    plt.rcParams.update({"font.size":10, "axes.labelsize":10, "axes.titlesize":11,
                         "xtick.labelsize":9, "ytick.labelsize":9, "legend.fontsize":8})
    methods = list(NAMES); main = points[points.window_s == 30]
    xlim, ylim = limits(main); time_norm=Normalize(0,14)
    fig, axes=plt.subplots(2,2,figsize=(11.4,7.7))
    fig.subplots_adjust(left=.085,right=.88,top=.855,bottom=.12,wspace=.27,hspace=.38)
    fig.suptitle("One traffic stream, four measurement operators",x=.085,y=.975,ha="left",fontsize=19,fontweight="bold",color="#152A3A")
    fig.text(.085,.917,"NGSIM I-80 | lane 2 | y = 150-250 m | 0-840 s | 30 s windows | all methods use the same data",color="#53636B",fontsize=10)
    for i,(ax,method) in enumerate(zip(axes.flat,methods)):
        part=main[main.method==method]
        sc=ax.scatter(part.k_veh_km,part.q_veh_h,c=(part.start_s+15)/60,cmap="viridis",norm=time_norm,s=32,alpha=.88,edgecolor="white",lw=.4)
        fit(ax,part)
        ax.set(xlim=xlim,ylim=ylim,xlabel="Density k (veh/km/lane)",ylabel="Flow q (veh/h/lane)",title=f"{NAMES[method]}  |  n = {len(part)}")
        ax.grid(alpha=.16);style.panel_label(ax,chr(97+i),x=-.14,y=1.04)
    fig.colorbar(sc,cax=fig.add_axes([.914,.21,.016,.59]),label="Minutes after 16:00")
    fig.text(.085,.057,"Black lines: descriptive local linear trends (density bandwidth 20 veh/km, minimum 4 windows); no extrapolation or capacity fit.",fontsize=8.5,color="#53636B")
    fig.text(.085,.026,"Detector density is inferred from passage-speed harmonic means. Edie, Voronoi and kernel integrate the fixed 100 m road segment.",fontsize=8.5,color="#53636B")
    save(fig,out,"methods")

    # All three fundamental relations, not only q-k.
    fig, axes=plt.subplots(2,4,figsize=(14,7))
    fig.subplots_adjust(left=.06,right=.985,top=.85,bottom=.12,wspace=.28,hspace=.30)
    fig.suptitle("Speed, density and flow on matched 30 s windows",x=.06,ha="left",y=.975,fontsize=19,fontweight="bold",color="#152A3A")
    fig.text(.06,.92,"Speed is q/k under each measurement definition; point-detector speeds use harmonic averaging.",fontsize=10,color="#53636B")
    vmax=np.ceil(main.v_km_h.max()/5)*5+5
    for col,method in enumerate(methods):
        p=main[main.method==method]
        axes[0,col].scatter(p.k_veh_km,p.v_km_h,c=COLORS[method],s=24,alpha=.7)
        axes[1,col].scatter(p.v_km_h,p.q_veh_h,c=COLORS[method],s=24,alpha=.7)
        axes[0,col].set(title=NAMES[method],xlim=xlim,ylim=(0,vmax),xlabel="k (veh/km/lane)")
        axes[1,col].set(xlim=(0,vmax),ylim=ylim,xlabel="v (km/h)")
        if col==0:axes[0,col].set_ylabel("v (km/h)");axes[1,col].set_ylabel("q (veh/h/lane)")
        for ax in axes[:,col]:ax.grid(alpha=.16)
    fig.text(.06,.045,"Different measurement operators change the averaging weights. A smoother cloud is not evidence of a more accurate capacity estimate.",fontsize=9,color="#53636B")
    save(fig,out,"speed_relations")

    fig, axes=plt.subplots(2,3,figsize=(13,8.2))
    fig.subplots_adjust(left=.065,right=.98,top=.86,bottom=.12,wspace=.29,hspace=.38)
    fig.suptitle("Sensitivity belongs within a measurement method",x=.065,ha="left",y=.975,fontsize=19,fontweight="bold",color="#152A3A")
    fig.text(.065,.925,"Temporal aggregation, spatial subdivision, kernel width and bin origin are separate controls.",fontsize=10,color="#53636B")
    ed=points[points.method=="edie"]; xl,yl=limits(ed)
    for ax,T in zip(axes[0],[10,30,60]):
        p=ed[ed.window_s==T];s=summary[(summary.method=="edie")&(summary.window_s==T)].iloc[0]
        ax.scatter(p.k_veh_km,p.q_veh_h,s=22,color=COLORS["edie"],alpha=.6);fit(ax,p)
        ax.set(title=f"Edie: T = {T} s, n = {len(p)}\nflow SD = {s.q_sd:.0f} veh/h",xlim=xl,ylim=yl,xlabel="k (veh/km/lane)",ylabel="q (veh/h/lane)")
    spatial=pd.read_csv(out/"spatial_sensitivity.csv")
    for length,color in zip([10,25,50,100],["#B7BEC5","#8D68AC","#009E73","#0072B2"]):
        p=spatial[spatial.cell_length_m==length]
        axes[1,0].scatter(p.k_veh_km,p.q_veh_h,s=6 if length<100 else 14,alpha=.18 if length<100 else .6,color=color,label=f"{length} m; n={len(p)}")
    sx,sy=limits(spatial)
    axes[1,0].set(xlim=sx,ylim=sy,title="Edie: split the same 100 m ROI (T = 10 s)",xlabel="k (veh/km/lane)",ylabel="q (veh/h/lane)")
    axes[1,0].legend(loc="upper right",fontsize=7)
    kernels=pd.read_csv(out/"kernel_sensitivity.csv")
    for h,color in zip([10,25,50],["#009E73","#8D68AC","#D55E00"]):
        p=kernels[kernels.halfwidth_m==h]
        axes[1,1].scatter(p.k_veh_km,p.q_veh_h,s=12,color=color,alpha=.28)
        fit(axes[1,1],p,color,label=f"halfwidth {h} m")
    axes[1,1].set(xlim=xlim,ylim=ylim,title="Kernel bandwidth (T = 30 s)",xlabel="k (veh/km/lane)",ylabel="q (veh/h/lane)");axes[1,1].legend(fontsize=7)
    shifts=pd.read_csv(out/"offset_sensitivity.csv")
    for phase,color in zip([0,15],["#0072B2","#D55E00"]):
        p=shifts[(shifts.window_s==30)&(shifts.offset_s==phase)]
        axes[1,2].scatter(p.k_veh_km,p.q_veh_h,s=17,color=color,alpha=.5)
        fit(axes[1,2],p,color,label=f"origin +{phase} s; n={len(p)}")
    axes[1,2].set(xlim=xlim,ylim=ylim,title="Edie window origin (T = 30 s)",xlabel="k (veh/km/lane)",ylabel="q (veh/h/lane)");axes[1,2].legend(fontsize=7)
    for i,ax in enumerate(axes.flat):ax.grid(alpha=.13);style.panel_label(ax,chr(97+i),x=-.13,y=1.05)
    fig.text(.065,.055,"Subcells reveal local heterogeneity and are not extra independent experiments. Area-weighted reaggregation exactly recovers the 100 m Edie values.",fontsize=8.5,color="#53636B")
    fig.text(.065,.028,"Shifted windows have a different initial boundary and a partial final window; actual durations are retained in the exported table.",fontsize=8.5,color="#53636B")
    save(fig,out,"sensitivity")

    # Trace a discrepancy back to trajectories, rather than inventing a mechanism.
    e=ed[ed.window_s==10].set_index("start_s")
    detector=points[(points.method=="detector")&(points.window_s==10)].set_index("start_s")
    example=float(abs(detector.q_veh_h-e.q_veh_h).idxmax());stop=example+10
    fig,axes=plt.subplots(3,1,figsize=(11.4,9.5),gridspec_kw={"height_ratios":[1.15,1,1]})
    fig.subplots_adjust(left=.085,right=.89,top=.90,bottom=.09,hspace=.55)
    fig.suptitle("Inspect the traffic states behind the scatter",x=.085,ha="left",y=.98,fontsize=19,fontweight="bold",color="#152A3A")
    cell=spatial[spatial.cell_length_m==10]
    grid=cell.pivot(index="x_min_m",columns="start_s",values="v_km_h")
    im=axes[0].imshow(grid,origin="lower",aspect="auto",extent=(0,14,150,250),cmap="viridis",vmin=0,vmax=60)
    axes[0].set(xlabel="Minutes after 16:00",ylabel="Longitudinal position (m)",title="10 m x 10 s Edie speed field within the 100 m segment")
    fig.colorbar(im,cax=fig.add_axes([.915,.686,.016,.203]),label="Speed (km/h)")
    for T,color in zip([10,60],["#85B8D5","#152A3A"]):
        p=ed[ed.window_s==T]
        axes[1].plot((p.start_s+p.end_s)/120,p.q_veh_h,color=color,lw=1 if T==10 else 2,label=f"Edie {T} s")
    axes[1].set(xlim=(0,14),ylabel="q (veh/h/lane)",xlabel="Minutes after 16:00",title="Long windows average short-lived high and low flows");axes[1].legend(ncol=2);axes[1].grid(alpha=.15)
    s=segments[(segments.t1>example)&(segments.t0<stop)&(segments.x1>100)&(segments.x0<300)]
    lines=np.stack([s[["t0","x0"]].to_numpy(),s[["t1","x1"]].to_numpy()],axis=1)
    axes[2].add_collection(LineCollection(lines,colors="#496A7B",lw=.75,alpha=.6))
    axes[2].axhspan(150,250,color="#E0ECF1",alpha=.45);axes[2].axhline(200,color="#D55E00",ls="--",lw=1.2)
    axes[2].set(xlim=(example,stop),ylim=(110,290),xlabel="Seconds after 16:00",ylabel="Longitudinal position (m)",
                title=f"Largest detector-Edie difference, selected after calculation: {example:.0f}-{stop:.0f} s")
    axes[2].text(.02,.96,f"Edie {e.loc[example].q_veh_h:.0f} veh/h | point detector {detector.loc[example].q_veh_h:.0f} veh/h",transform=axes[2].transAxes,va="top",fontsize=9)
    fig.text(.085,.029,"Highlighted ROI: 150-250 m; dashed line: detector at 200 m. This case explains measurement support and was not used to select the ROI.",fontsize=8.5,color="#53636B")
    save(fig,out,"diagnostics")

    # Compare raw coordinate derivatives with the supplied speed field explicitly.
    fig, axes=plt.subplots(1,3,figsize=(12,4.2))
    fig.subplots_adjust(left=.065,right=.98,top=.77,bottom=.19,wspace=.33)
    fig.suptitle("Upstream trajectory quality is separate from aggregation",x=.065,ha="left",y=.965,fontsize=18,fontweight="bold",color="#152A3A")
    fig.text(.065,.86,"Raw position-derived velocity is used consistently in all four main methods; no spike deletion or trajectory smoothing.",fontsize=9,color="#53636B")
    for ax,identifier in zip(axes[:2],audit["roi_anomaly_vehicle_ids"]):
        track=segments[segments.vehicle_id==identifier]
        peak=track.loc[track.velocity.idxmax()]
        track=track[(track.t0>=peak.t0-2)&(track.t1<=peak.t0+2)].sort_values("t0")
        time=(track.t0+track.t1)/2
        ax.plot(time,track.velocity,color="#D55E00",lw=1.2,label="Position derivative")
        ax.plot(time,track.reported_velocity,color="#0072B2",lw=1.2,label="Supplied speed")
        ax.set(xlabel="Time (s)",ylabel="Speed (m/s)",title=f"Vehicle {identifier}: raw speed spike")
        ax.grid(alpha=.15);ax.legend(fontsize=7);ax.ticklabel_format(axis="x",style="plain",useOffset=False)
    check=pd.read_csv(out/"speed_definition_check.csv")
    delta=check.q_speed_column-check.q_geometry
    axes[2].plot((check.start_s+15)/60,delta,color="#755299",lw=1.2,marker="o",ms=3)
    axes[2].axhline(0,color="#7B8B94",lw=.8)
    axes[2].set(xlabel="Minutes after 16:00",ylabel="Flow difference (veh/h)",title="Supplied-speed flux minus geometric flux\n30 s windows, same vehicle residence weights")
    axes[2].grid(alpha=.15)
    fig.text(.065,.045,f"Maximum absolute 30 s flow difference: {abs(delta).max():.2f} veh/h. Small aggregate differences do not establish accurate instantaneous velocities.",fontsize=9,color="#53636B")
    save(fig,out,"quality")

    # A schematic, explicitly separate from measured results.
    fig,axes=plt.subplots(2,2,figsize=(11.4,6.7))
    fig.subplots_adjust(left=.07,right=.97,top=.86,bottom=.11,wspace=.28,hspace=.40)
    fig.suptitle("What each measurement averages",x=.07,ha="left",y=.98,fontsize=19,fontweight="bold",color="#152A3A")
    fig.text(.07,.915,"Schematic only: positions and trajectories below are illustrative, not NGSIM measurements.",fontsize=10,color="#53636B")
    for offset in [-1,1,3,5,7]:
        t=np.linspace(0,10,100);axes[0,0].plot(t,offset+1.4*t,color="#6698B2",lw=1)
    axes[0,0].add_patch(Rectangle((2,4),5,6,facecolor="#D7E8F1",edgecolor="#0072B2",alpha=.55))
    axes[0,0].set(xlim=(0,10),ylim=(0,16),xlabel="time",ylabel="position",title="Edie: clip every trajectory to one L x T box")
    axes[0,0].text(.02,.97,"k = sum(residence time)/(L T)\nq = sum(distance)/(L T)",transform=axes[0,0].transAxes,va="top",fontsize=9)
    axes[0,1].axhline(.5,color="#D55E00",lw=2)
    for t in [1,3,4.2,7,8.5]:axes[0,1].plot([t,t],[.2,.8],color="#152A3A",lw=1.2)
    axes[0,1].set(xlim=(0,10),ylim=(0,1.2),xlabel="passage time",yticks=[],title="Detector: count passages at one fixed position")
    axes[0,1].text(.02,.96,"q = N/T;  v = harmonic mean of passage speeds\nk = q/v (inferred, not a segment vehicle count)",transform=axes[0,1].transAxes,va="top",fontsize=9)
    xx=np.array([0,2,4,5,8,10,13]);mid=(xx[1:]+xx[:-1])/2
    for lo,hi in zip(mid[:-1],mid[1:]):axes[1,0].add_patch(Rectangle((lo,0),hi-lo,1/(hi-lo),facecolor="#72BEAA",edgecolor="white",alpha=.7))
    axes[1,0].scatter(xx,np.zeros(len(xx)),color="#152A3A",s=22,zorder=4)
    axes[1,0].axvspan(3,9,color="#F1D69C",alpha=.4)
    axes[1,0].set(xlim=(0,13),ylim=(-.1,.85),xlabel="position",ylabel="local density",title="Voronoi: adaptive cell lengths; integrate ROI overlap")
    axes[1,0].text(.02,.98,"Use full cell length in 1/length;\nclip only the overlap with the ROI.",transform=axes[1,0].transAxes,va="top",fontsize=9)
    x=np.linspace(-2,15,300);total=np.zeros_like(x)
    for c in xx:
        k=np.maximum(0,1-abs(x-c)/2)/2;total+=k;axes[1,1].plot(x,k,color="#AA93BE",alpha=.7,lw=.8)
    axes[1,1].plot(x,total,color="#755299",lw=1.8);axes[1,1].axvspan(3,9,color="#F1D69C",alpha=.4)
    axes[1,1].set(xlim=(0,13),ylim=(0,max(total)*1.25),xlabel="position",ylabel="local density",title="Kernel: spread each vehicle with fixed bandwidth")
    axes[1,1].text(.02,.98,"Integrate unit-mass kernels over the same ROI;\nflow uses the same weights times each speed.",transform=axes[1,1].transAxes,va="top",fontsize=9)
    for i,ax in enumerate(axes.flat):style.panel_label(ax,chr(97+i),x=-.12,y=1.03)
    save(fig,out,"method_schematic")
    return example
