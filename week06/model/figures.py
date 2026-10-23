"""Scientific figures from saved trajectories; no hand-edited result values."""
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from matplotlib.animation import FuncAnimation,PillowWriter
from week01.style import apply
from .scenarios import narrow,mixed_circle
from .geometry import pair_geometry

COLORS={"sfm":"#0072B2","idm":"#D55E00","orca":"#009E73"}
LABELS={"sfm":"SFM","idm":"IDM-heading","orca":"ORCA adapted"}
PED="#2878B5";CAR="#D95F35"


def title(fig,heading,subtitle):
    fig.suptitle(heading,x=.07,y=.985,ha="left",fontsize=17,fontweight="bold",color="#173243")
    fig.text(.07,.985-27/(72*fig.get_figheight()),subtitle,va="top",fontsize=9,color="#546873")


def save(fig,path):
    fig.savefig(path,dpi=190,facecolor="white")
    plt.close(fig)


def key(scene,width,model,variant):
    return f"{scene}_{width}_{model}_{variant}"


def row_for(metrics,scene,model,variant,width=None):
    mask=(metrics.scene==scene)&(metrics.model==model)&(metrics.variant==variant)
    if width is not None:mask&=metrics.width_m==width
    return metrics[mask].iloc[0]


def road_axes(ax,width):
    ax.axhspan(-width/2,width/2,color="#F2F5F6",zorder=0)
    ax.axhline(width/2,color="#55626B",lw=1);ax.axhline(-width/2,color="#55626B",lw=1)
    ax.axhline(0,color="#A8B5BD",lw=.5,ls="--")
    ax.set_xlim(-3,34);ax.set_ylim(-2.5,2.5);ax.set_aspect("equal")
    ax.set_ylabel("y (m)");ax.grid(alpha=.12)


def trace_road(ax,archive,case,scene,at=6.,full=True):
    p=archive[case+"__positions"];times=archive[case+"__times"]
    k=np.argmin(abs(times-at))
    if full:
        for i in range(len(scene["p0"])):
            c=PED if scene["lateral"][i] else CAR
            ax.plot(p[:,i,0],p[:,i,1],color=c,alpha=.6,lw=.9)
            ax.plot(*p[0,i],"o",color=c,ms=2)
    for i in range(len(scene["p0"])):
        if scene["lateral"][i]:
            ax.add_patch(Circle(p[k,i],scene["radius"][i],facecolor=PED,edgecolor="white",lw=.3,zorder=5))
        else:
            extent=scene["extent"][i]
            ax.add_patch(Rectangle(p[k,i]-extent,2*extent[0],2*extent[1],facecolor=CAR,alpha=.85,zorder=4))


def narrow_trajectories(cfg,metrics,z,out):
    fig,axes=plt.subplots(6,2,figsize=(13,10.6))
    title(fig,"One interaction law for walkers and cars","Same six walkers + one 4.4 x 1.8 m straight car. Only road width changes between columns.")
    for r,(m,v) in enumerate((m,v) for m in cfg["models"] for v in cfg["variants"]):
        for c,w in enumerate([4.2,3.2]):
            ax=axes[r,c];s=narrow(cfg,w);road_axes(ax,w)
            trace_road(ax,z,key("narrow",w,m,v),s)
            row=row_for(metrics,"narrow",m,v,w)
            status="valid" if row.physically_feasible_complete else ("geometry only" if row.geometry_valid_complete else "failure / waiting")
            ax.set_title(f"{string.ascii_lowercase[r*2+c]}  {LABELS[m]} | {v} | W={w:g} m\nexit {int(row.arrived)}/7; {status}",loc="left",fontsize=9)
            if r==5:ax.set_xlabel("x (m)")
    fig.subplots_adjust(top=.89,bottom=.07,left=.07,right=.98,wspace=.12,hspace=.65)
    fig.text(.07,.025,"Paths are complete diagnostic rollouts; disks/rectangles show t=6 s. Exit counts alone do not certify physical validity.",fontsize=9,color="#546873")
    save(fig,out/"narrow_trajectories.png")


def narrow_metrics(cfg,metrics,z,out):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    title(fig,"Width, throughput, contact and intervention","A centered 1.8 m car requires W >= 2.8 m for a 0.5 m pedestrian to fit on one side (zero buffer).")
    columns=[("arrived","Agents fully exited / 7"),("body_overlap_pair_seconds","Overlap pair-seconds"),("wall_violation_agent_steps","Wall-violation agent-steps"),("max_realized_acceleration_m_s2","Peak realized acceleration (m/s²)")]
    for ax,(column,label) in zip(axes.flat,columns):
        for m in cfg["models"]:
            for v in cfg["variants"]:
                d=metrics[(metrics.scene=="narrow")&(metrics.model==m)&(metrics.variant==v)].sort_values("width_m")
                ax.plot(d.width_m,d[column],color=COLORS[m],ls="--" if v=="base" else "-",marker="o" if v=="base" else "s",ms=4,label=f"{LABELS[m]} / {v}")
        ax.axvline(2.8,color="#7B8184",ls=":",lw=1);ax.set_xlabel("Road width (m)");ax.set_ylabel(label);ax.grid(alpha=.15)
    axes[0,0].set_ylim(-.3,7.5);axes[0,0].legend(fontsize=7,ncol=2)
    axes[1,1].axhline(4,color="#333333",ls=":",lw=1)
    fig.subplots_adjust(top=.88,bottom=.11,left=.085,right=.98,hspace=.3,wspace=.25)
    fig.text(.085,.035,"Base ORCA can leave the road. A geometry veto can prevent overlap yet exceed the specified acceleration bound; these are recorded failures.",fontsize=8.5,color="#546873")
    save(fig,out/"narrow_metrics.png")


def failure_figure(cfg,metrics,z,out):
    scene=narrow(cfg,3.2)
    base=z[key("narrow",3.2,"sfm","base")+"__positions"]
    times=z[key("narrow",3.2,"sfm","base")+"__times"]
    arrival=z[key("narrow",3.2,"sfm","base")+"__arrival_times"]
    minima=[]
    for t,p in zip(times,base):
        gap=pair_geometry(p,scene["radius"],scene["extent"])[0]
        active=np.isnan(arrival)|(arrival>t)
        gap[~active,:]=np.inf;gap[:,~active]=np.inf
        minima.append(gap.min())
    worst=int(np.argmin(minima))
    at=float(z[key("narrow",3.2,"sfm","base")+"__times"][worst])
    specs=[(3.2,"base",at,"Overlap / boundary failure"),(3.2,"guided",at,"Route guidance: passes without veto"),(2.9,"guided",25.,"Space exists, but rule stalls"),(2.6,"guided",25.,"Too little space: waiting is appropriate")]
    fig,axes=plt.subplots(2,2,figsize=(12,6.8))
    title(fig,"Distinguish a rule failure from a space limitation","SFM comparisons: identical bodies, starts, desired speeds and integration step. Guidance changes are explicit.")
    for ax,(w,v,t,caption) in zip(axes.flat,specs):
        road_axes(ax,w);trace_road(ax,z,key("narrow",w,"sfm",v),narrow(cfg,w),at=t)
        ax.set_xlim(5,24);ax.set_ylim(-2,2)
        r=row_for(metrics,"narrow","sfm",v,w)
        ax.set_title(f"{caption}\nW={w:g} m; snapshot t={t:g} s; final exits={int(r.arrived)}/7",fontsize=9)
        ax.set_xlabel("x (m)")
        ax.text(.02,.03,f"min body gap {r.minimum_body_gap_m:.3f} m",transform=ax.transAxes,fontsize=8)
    fig.subplots_adjust(top=.84,bottom=.12,hspace=.4,wspace=.16)
    fig.text(.07,.04,"2.9 m has only 0.05 m spare space per side: geometric possibility does not ensure that the current local comfort/steering rules can use it.",fontsize=8.5,color="#546873")
    save(fig,out/"failure_and_repair.png")


def circle_trajectories(cfg,metrics,z,obs,out):
    fig,axes=plt.subplots(2,4,figsize=(13.4,7.7))
    title(fig,"Circle antipode: same observed starts, different mechanisms","64 pedestrians | radius approximately 10 m | 16.96 s | first 2 s initialize parameters; no later measured positions used")
    specifications=[("Observed",obs["positions"],None)]+[(f"{LABELS[m]} / {v}",z[key("circle",None,m,v)+"__positions"],row_for(metrics,"circle",m,v)) for m in cfg["models"] for v in cfg["variants"]]
    color=plt.cm.twilight(np.linspace(0,1,64,endpoint=False))
    for ax,(label,p,row) in zip(axes.flat,specifications):
        for i in range(64):ax.plot(p[:,i,0],p[:,i,1],color=color[i],lw=.65,alpha=.65)
        ax.scatter(p[0,:,0],p[0,:,1],s=5,color="#344754")
        ax.add_patch(Circle(obs["center"],10,fill=False,ls=":",ec="#9DA9AE",lw=.7))
        ax.set_xlim(-12,12);ax.set_ylim(-12,12);ax.set_aspect("equal")
        subtitle="64 observed arrivals" if row is None else f"{int(row.arrived)}/64 reached; gap min={row.minimum_body_gap_m:.3f} m"
        ax.set_title(label+"\n"+subtitle,fontsize=9);ax.set_xlabel("x (m)");ax.set_ylabel("y (m)")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.9,"Read the failures",fontsize=13,fontweight="bold",color="#173243")
    axes.flat[-1].text(0,.77,"Arrival is not collision-free success.\n\nSFM can overlap under compression.\nIDM needs usable lateral decisions.\nORCA loses its original guarantee\nafter acceleration/body constraints.\n\nGray circle marks start geometry;\nit is not a wall.",fontsize=10,va="top",linespacing=1.5)
    fig.subplots_adjust(top=.86,bottom=.1,wspace=.3,hspace=.35)
    fig.text(.07,.025,"Original 64-person data are the only empirical reference. Bodies are assumed disks (r=0.25 m); overlaps are geometry diagnostics.",fontsize=9,color="#546873")
    save(fig,out/"circle_trajectories.png")


def circle_diagnostics(cfg,metrics,series,out):
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    title(fig,"A population curve can fit while individual paths fail","Same-record mechanism comparison; first 2 s inform initialization / desired speeds. No independent validation claim.")
    for ax,column,ylabel in zip(axes[0],["mean_speed_mps","mean_radius_m","arrived_fraction"],["Mean speed (m/s)","Mean distance to center (m)","Cumulative arrived fraction"]):
        observed=series[series.case=="observed"]
        ax.plot(observed.time_s,observed[column],color="#253A47",lw=2,label="Observed")
        for m in cfg["models"]:
            for v in cfg["variants"]:
                d=series[series.case==key("circle",None,m,v)]
                ax.plot(d.time_s,d[column],color=COLORS[m],ls="--" if v=="base" else "-",lw=1.2,alpha=.85,label=f"{LABELS[m]} {v}")
        ax.set_xlabel("Time (s)");ax.set_ylabel(ylabel);ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.53,.915),fontsize=7.5,ncol=4,frameon=False)
    for ax,column,ylabel in zip(axes[1],["ade_after_calibration_m","minimum_body_gap_m","mean_speed_curve_rmse_m_s"],["Mean position error after 2 s (m)","Minimum swept body gap (m)","Mean-speed curve RMSE (m/s)"]):
        for j,v in enumerate(cfg["variants"]):
            d=metrics[(metrics.scene=="circle")&(metrics.variant==v)].set_index("model").loc[cfg["models"]]
            ax.bar(np.arange(3)+(j-.5)*.34,d[column],width=.33,color=[COLORS[m] for m in cfg["models"]],alpha=.35 if v=="base" else 1,label=v)
        ax.set_xticks(np.arange(3),[LABELS[m] for m in cfg["models"]],rotation=10)
        ax.set_ylabel(ylabel);ax.axhline(0,color="#8A969D",lw=.6);ax.legend(fontsize=8)
    fig.subplots_adjust(top=.82,bottom=.12,wspace=.35,hspace=.35)
    fig.text(.07,.025,"Negative model body gap = interpenetration of assumed footprints. Both body size and numerical step influence this diagnostic.",fontsize=9,color="#546873")
    save(fig,out/"circle_diagnostics.png")


def mixed_figure(cfg,metrics,z,obs,out):
    fig,axes=plt.subplots(3,3,figsize=(12.5,12))
    title(fig,"Mixed circle: a counterfactual stress test","32 walkers + 32 cars, strictly alternating | R=30 m | straight / guided straight / steering improvement")
    s=mixed_circle(cfg,obs["center"])
    for r,v in enumerate(["base","guided","steering"]):
        for c,m in enumerate(cfg["models"]):
            ax=axes[r,c];case=key("circle_mixed",None,m,v);p=z[case+"__positions"]
            for i in range(len(s["p0"])):
                color=PED if s["lateral"][i] else CAR
                ax.plot(p[:,i,0],p[:,i,1],color=color,alpha=.6,lw=.7 if s["lateral"][i] else 1.1)
                ax.scatter(*p[0,i],s=9,color=color)
            for i in np.flatnonzero(~s["lateral"]):
                ax.add_patch(Circle(p[0,i],s["radius"][i],fill=False,ec=CAR,lw=.35,ls=":"))
                theta=np.degrees(np.arctan2(s["heading"][i,1],s["heading"][i,0]))
                ax.add_patch(Rectangle(p[0,i]-[2.2,.9],4.4,1.8,angle=theta,rotation_point="center",fill=False,ec=CAR,lw=.6))
            row=row_for(metrics,"circle_mixed",m,v)
            ax.set_title(f"{LABELS[m]} / {v}\narrived pedestrians {int(row.pedestrians_arrived)}/32, cars {int(row.cars_arrived)}/32\nenvelope gap min {row.minimum_body_gap_m:.3f} m",fontsize=9)
            ax.set_aspect("equal");ax.set_xlim(-35,35);ax.set_ylim(-35,35);ax.set_xlabel("x (m)");ax.set_ylabel("y (m)")
    fig.subplots_adjust(top=.88,bottom=.07,hspace=.4,wspace=.3)
    fig.text(.07,.025,"Blue: pedestrians; orange: cars. Initial rectangles show real size; dotted circumdisks are conservative collision envelopes.",fontsize=8.5,color="#546873")
    save(fig,out/"mixed_circle.png")
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    title(fig,"Does bounded steering repair mixed-circle deadlock?","Same 32 walkers + 32 cars, R=30 m, initial states and speed targets. Steering is an explicit model improvement.")
    columns=[("pedestrians_arrived","Pedestrians arrived / 32"),("cars_arrived","Cars arrived / 32"),("minimum_body_gap_m","Minimum envelope gap (m)"),("max_realized_acceleration_m_s2","Peak realized acceleration (m/s²)")]
    for ax,(column,label) in zip(axes.flat,columns):
        for j,variant in enumerate(["base","guided","steering"]):
            d=metrics[(metrics.scene=="circle_mixed")&(metrics.variant==variant)].set_index("model").loc[cfg["models"]]
            ax.bar(np.arange(3)+(j-1)*.25,d[column],width=.24,alpha=[.3,.6,1][j],color=[COLORS[m] for m in cfg["models"]],label=variant)
        ax.set_xticks(np.arange(3),[LABELS[m] for m in cfg["models"]])
        ax.set_ylabel(label);ax.axhline(0,color="#777777",lw=.6)
    axes[0,0].legend(fontsize=8)
    fig.subplots_adjust(top=.85,bottom=.11,hspace=.3,wspace=.3)
    fig.text(.07,.035,"Envelope overlap is a conservative conflict diagnostic, not proof of physical rectangle contact; negative gaps still violate this model.",fontsize=8.5,color="#546873")
    save(fig,out/"mixed_diagnostics.png")


def robustness(cfg,metrics,studies,out):
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    title(fig,"Separate guidance, emergency veto and time discretization","A geometry filter is not a substitute for feasible acceleration. All explored outcomes remain in the CSV files.")
    labels=["base","guided_no_veto","veto_only","guided"]
    for j,m in enumerate(cfg["models"]):
        values=[]
        for variant in labels:
            d=(metrics if variant in ["base","guided"] else studies)
            r=row_for(d,"narrow",m,variant,3.2);values.append(r.arrived)
        axes[0].plot(np.arange(4),values,marker="o",color=COLORS[m],label=LABELS[m])
    axes[0].set_xticks(np.arange(4),["Base","Guide only","Veto only","Both"],rotation=25)
    axes[0].set_ylabel("Exits / 7 at W=3.2 m");axes[0].legend(fontsize=8)
    for m in cfg["models"]:
        main=row_for(metrics,"circle",m,"guided")
        fine=studies[(studies.study=="half_step")&(studies.scene=="circle")&(studies.model==m)].iloc[0]
        axes[1].plot([.02,.04],[fine.arrived,main.arrived],marker="o",color=COLORS[m])
    axes[1].set_xlabel("Integration step (s)");axes[1].set_ylabel("Circle arrivals / 64");axes[1].set_xticks([.02,.04])
    main=row_for(metrics,"circle","orca","guided")
    seed_rows=studies[studies.study=="side_seed"]
    axes[2].bar([42,43,44],[main.ade_after_calibration_m,*seed_rows.ade_after_calibration_m],color=COLORS["orca"])
    axes[2].set_xticks([42,43,44]);axes[2].set_xlabel("Persistent side-preference seed");axes[2].set_ylabel("ORCA guided mean position error (m)")
    for ax in axes:ax.grid(alpha=.15)
    fig.subplots_adjust(top=.78,bottom=.2,wspace=.35)
    fig.text(.07,.04,"Half-step tests check numerical sensitivity; they do not certify convergence or safety. Arrival counts must be read with overlap and acceleration diagnostics.",fontsize=8,color="#546873")
    save(fig,out/"robustness.png")


def animations(cfg,z,obs,out):
    # One animation compares the required failure and repair under identical conditions.
    specs=[(3.2,"base"),(3.2,"guided"),(2.6,"guided")]
    fig,axes=plt.subplots(3,1,figsize=(10,4.5));objects=[]
    for ax,(w,v) in zip(axes,specs):
        road_axes(ax,w);ax.set_title(f"SFM {v}; W={w:g} m",fontsize=10)
        case=key("narrow",w,"sfm",v);p=z[case+"__positions"]
        circles=[Circle(p[0,i],.25,fc=PED) for i in range(6)]
        for patch in circles:ax.add_patch(patch)
        car=Rectangle(p[0,-1]-[2.2,.9],4.4,1.8,fc=CAR);ax.add_patch(car)
        label=ax.text(.01,.9,"",transform=ax.transAxes,fontsize=8)
        objects.append((p,circles,car,label,z[case+"__arrival_times"]))
    fig.tight_layout(pad=.8)
    frames=range(0,1001,10)
    def update(k):
        for p,circles,car,label,arr in objects:
            for i,patch in enumerate(circles):
                patch.center=p[k,i];patch.set_visible(not(np.isfinite(arr[i]) and k*.04>=arr[i]))
            car.set_xy(p[k,-1]-[2.2,.9]);car.set_visible(not(np.isfinite(arr[-1]) and k*.04>=arr[-1]))
            label.set_text(f"t={k*.04:.1f} s")
    FuncAnimation(fig,update,frames=frames,interval=100).save(out/"narrow_comparison.gif",writer=PillowWriter(fps=10),dpi=90)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12,3.5));items=[]
    data=[("Observed",obs["positions"])]+[(LABELS[m]+" guided",z[key("circle",None,m,"guided")+"__positions"]) for m in cfg["models"]]
    for ax,(name,p) in zip(axes,data):
        ax.set_xlim(-12,12);ax.set_ylim(-12,12);ax.set_aspect("equal");ax.set_title(name,fontsize=10)
        ax.add_patch(Circle(obs["center"],10,fill=False,ls=":",ec="#BBBBBB"))
        scatter=ax.scatter(p[0,:,0],p[0,:,1],s=10,c=np.arange(64),cmap="twilight",vmin=0,vmax=63)
        label=ax.text(.03,.95,"",transform=ax.transAxes,fontsize=8);items.append((p,scatter,label))
    fig.tight_layout()
    def update_circle(k):
        for p,scatter,label in items:scatter.set_offsets(p[k]);label.set_text(f"t={k*.04:.2f} s")
    FuncAnimation(fig,update_circle,frames=range(0,425,5),interval=100).save(out/"circle_comparison.gif",writer=PillowWriter(fps=10),dpi=85)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6.8));items=[]
    scene=mixed_circle(cfg,obs["center"]);walkers=scene["lateral"]
    for r,variant in enumerate(["guided","steering"]):
        for c,m in enumerate(cfg["models"]):
            ax=axes[r,c];p=z[key("circle_mixed",None,m,variant)+"__positions"]
            ax.set_xlim(-35,35);ax.set_ylim(-35,35);ax.set_aspect("equal")
            ax.set_title(f"{LABELS[m]} / {variant}",fontsize=10)
            ax.add_patch(Circle(obs["center"],30,fill=False,ls=":",ec="#BBBBBB"))
            ped=ax.scatter(p[0,walkers,0],p[0,walkers,1],s=7,c=PED)
            car=ax.scatter(p[0,~walkers,0],p[0,~walkers,1],s=22,c=CAR,marker="s")
            label=ax.text(.02,.95,"",transform=ax.transAxes,fontsize=8)
            items.append((p,ped,car,label))
    fig.tight_layout()
    def update_mixed(k):
        for p,ped,car,label in items:
            ped.set_offsets(p[k,walkers]);car.set_offsets(p[k,~walkers]);label.set_text(f"t={k*cfg['dt_s']:.1f} s")
    FuncAnimation(fig,update_mixed,frames=range(0,len(items[0][0]),20),interval=100).save(out/"mixed_comparison.gif",writer=PillowWriter(fps=10),dpi=80)
    plt.close(fig)


def plot_all(cfg,metrics,z,obs,out,studies=None,gifs=False):
    apply()
    narrow_trajectories(cfg,metrics,z,out);narrow_metrics(cfg,metrics,z,out)
    failure_figure(cfg,metrics,z,out);circle_trajectories(cfg,metrics,z,obs,out)
    series=pd.read_csv(out/"circle_series.csv");circle_diagnostics(cfg,metrics,series,out)
    mixed_figure(cfg,metrics,z,obs,out)
    if studies is not None:robustness(cfg,metrics,studies,out)
    if gifs:animations(cfg,z,obs,out)
