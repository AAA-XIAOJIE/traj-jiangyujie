"""Matched scientific panels; counterfactual trajectories never get truth labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from week01 import style
from .models import DEFAULTS, optimal_velocity, acceleration

COLORS = {"OVM":"#0072B2","FVDM":"#D55E00","IDM":"#009E73"}
SCENARIOS = {"constant":"Constant-speed leader","braking":"Braking leader","observed":"Observed leader replay"}


def save(fig, out, name):
    fig.savefig(out/f"{name}.png",dpi=210,facecolor="white")
    plt.close(fig)


def heading(fig, title, subtitle, left=.075, subtitle_y=.925):
    fig.suptitle(title,x=left,y=.98,ha="left",fontsize=18,fontweight="bold",color="#152A3A")
    fig.text(left,subtitle_y,subtitle,color="#53636B",fontsize=9)


def common(ax):
    ax.grid(alpha=.15)
    ax.set_xlim(0,30)


def collision_marker(ax, row, y=0):
    if pd.notna(row.collision_time_s):
        ax.scatter([row.collision_time_s],[y],s=48,marker="x",lw=1.7,color=COLORS[row.model],zorder=7)


def plot_all(data, test, metrics, traces, parameters, config, out):
    style.apply()
    plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":10.5,
                         "legend.fontsize":8,"xtick.labelsize":9,"ytick.labelsize":9})
    models = list(COLORS)
    main = metrics[metrics.parameter_set=="calibrated"]
    fig,axes=plt.subplots(2,3,figsize=(13.2,7.2))
    fig.subplots_adjust(left=.065,right=.98,top=.85,bottom=.12,wspace=.28,hspace=.31)
    heading(fig,"Three models, one matched initial state",
            "NGSIM lane 2 | follower 378, leader 368 | held-out start 80.9 s | calibrated on the preceding 30 s only",.065)
    for col,scenario in enumerate(SCENARIOS):
        for model in models:
            f=traces[("calibrated",model,scenario)]
            axes[0,col].plot(f.time_s,f.v_m_s,color=COLORS[model],label=model,lw=1.7)
            axes[1,col].plot(f.time_s,f.gap_m,color=COLORS[model],lw=1.7)
            record=main[(main.model==model)&(main.scenario==scenario)].iloc[0]
            collision_marker(axes[1,col],record)
        axes[0,col].plot(f.time_s,f.leader_v_m_s,color="#243743",lw=1.1,ls="--",label="Leader input")
        if scenario=="observed":
            axes[0,col].plot(f.time_s,test.follower_v_raw_m_s.to_numpy(),color="#7D8386",ls=":",lw=2,label="Observed follower")
            axes[1,col].plot(f.time_s,test.net_gap_m.to_numpy(),color="#7D8386",ls=":",lw=2)
        if scenario=="braking":
            for ax in axes[:,col]:ax.axvline(5,color="#9BA9AE",lw=.8,ls=":")
        axes[0,col].set(title=SCENARIOS[scenario],ylabel="Follower speed (m/s)",ylim=(0,13))
        axes[1,col].set(ylabel="Net bumper gap (m)",xlabel="Seconds after validation start",ylim=(0,19))
        for ax in axes[:,col]:common(ax)
    axes[0,0].legend(ncol=2,fontsize=7.5)
    axes[0,2].legend(fontsize=7)
    for i,ax in enumerate(axes.flat):style.panel_label(ax,chr(97+i),x=-.15,y=1.03)
    fig.text(.065,.045,"Braking: -1.5 m/s² from t = 5 s until the leader stops. Crosses mark contact; trajectories end at collision.",fontsize=8.5,color="#53636B")
    fig.text(.065,.019,"The observed follower is a reference only in the replay column. Constant/braking interventions have no observed follower truth.",fontsize=8.5,color="#53636B")
    save(fig,out,"overview")

    fig,axes=plt.subplots(2,2,figsize=(11.6,7.4))
    fig.subplots_adjust(left=.08,right=.97,top=.85,bottom=.12,wspace=.28,hspace=.33)
    heading(fig,"Validation uses a freely simulated follower",
            "Observed leader position is prescribed; only the follower's initial x and v are observed inputs.",.08)
    t=test.relative_time_s.to_numpy()-30
    axes[0,0].plot(t,test.follower_x_m.to_numpy()-test.follower_x_m.iloc[0],color="#52616A",ls=":",lw=2,label="Observed follower")
    axes[0,1].plot(t,test.follower_v_raw_m_s,color="#52616A",ls=":",lw=2)
    axes[1,0].plot(t,test.net_gap_m,color="#52616A",ls=":",lw=2)
    for model in models:
        f=traces[("calibrated",model,"observed")]
        axes[0,0].plot(t,f.x_m-f.x_m.iloc[0],color=COLORS[model],label=model,lw=1.6)
        axes[0,1].plot(t,f.v_m_s,color=COLORS[model],lw=1.6)
        axes[1,0].plot(t,f.gap_m,color=COLORS[model],lw=1.6)
        axes[1,1].plot(t,f.x_m-test.follower_x_m.to_numpy(),color=COLORS[model],lw=1.6)
    axes[0,0].set(title="Measured and simulated trajectories",ylabel="Travelled distance (m)")
    axes[0,1].set(title="Measured and simulated speed",ylabel="Speed (m/s)")
    axes[1,0].set(title="Measured and simulated net gap",ylabel="Net gap (m)")
    axes[1,1].set(title="Signed follower position error",ylabel="Simulation - observation (m)")
    axes[1,1].axhline(0,color="#A2AAAE",lw=.8)
    axes[0,0].legend(ncol=2)
    for i,ax in enumerate(axes.flat):
        common(ax);ax.set_xlabel("Seconds after 80.9 s");style.panel_label(ax,chr(97+i),x=-.13,y=1.03)
    fig.text(.08,.035,"301 validation observations at 0.1 s; calibration uses the disjoint preceding 300 observations. No teacher forcing after initialization.",fontsize=8.5,color="#53636B")
    save(fig,out,"real_replay")

    fig,axes=plt.subplots(3,2,figsize=(11.6,9.1))
    fig.subplots_adjust(left=.08,right=.97,top=.86,bottom=.10,wspace=.27,hspace=.32)
    heading(fig,"Calibration accuracy and braking safety can disagree",
            "Solid: independently calibrated. Dashed: common-OV-function control baseline / IDM paper parameters.",.08)
    variables=[("v_m_s","Speed (m/s)"),("gap_m","Net gap (m)"),("a_m_s2","Model acceleration (m/s²)")]
    for col,scenario in enumerate(["constant","braking"]):
        for row,(var,label) in enumerate(variables):
            ax=axes[row,col]
            for model in models:
                for set_name,ls,lw in [("baseline","--",1.),("calibrated","-",1.7)]:
                    f=traces[(set_name,model,scenario)]
                    ax.plot(f.time_s,f[var],color=COLORS[model],ls=ls,lw=lw,alpha=.65 if set_name=="baseline" else 1)
                    if row==1:
                        record=metrics[(metrics.model==model)&(metrics.parameter_set==set_name)&(metrics.scenario==scenario)].iloc[0]
                        collision_marker(ax,record)
            if row==0:
                ax.plot(f.time_s,f.leader_v_m_s,color="#243743",ls=":",lw=1.4)
                ax.set_title(SCENARIOS[scenario])
            if scenario=="braking":ax.axvline(5,color="#9BA9AE",ls=":",lw=.8)
            ax.set_ylabel(label);common(ax)
            if row==2:ax.set_xlabel("Seconds after validation start")
            if var=="gap_m":ax.set_ylim(bottom=0)
    handles=[Line2D([0],[0],color=COLORS[m],lw=1.8,label=m) for m in models]
    handles += [Line2D([0],[0],color="#52616A",lw=1.5,label="Calibrated"),
                Line2D([0],[0],color="#52616A",lw=1,ls="--",label="Baseline")]
    fig.legend(handles=handles,loc="upper right",bbox_to_anchor=(.97,.919),ncol=5,fontsize=8,frameon=False)
    fig.text(.08,.032,"Negative-speed requests are counted even when the ballistic stop boundary holds v = 0. No gap floor, acceleration cap or collision repair is applied.",fontsize=8.5,color="#53636B")
    save(fig,out,"interventions")

    fig,axes=plt.subplots(2,2,figsize=(11.6,7.3))
    fig.subplots_adjust(left=.08,right=.97,top=.84,bottom=.12,wspace=.30,hspace=.40)
    heading(fig,"Robustness checks preserve failures",
            "Only the first 30 s fit parameters; validation, braking and convergence never refit them.",.08)
    observed=metrics[metrics.scenario=="observed"]
    x=np.arange(3);width=.34
    for ax,metric,label in [(axes[0,0],"position_rmse_m","Position RMSE (m)"),
                            (axes[0,1],"speed_rmse_m_s","Speed RMSE (m/s)")]:
        for set_name,offset,color in [("baseline",-width/2,"#AFC1CC"),("calibrated",width/2,"#366980")]:
            p=observed[observed.parameter_set==set_name].set_index("model").loc[models]
            bars=ax.bar(x+offset,p[metric],width,color=color,label=set_name)
            ax.bar_label(bars,fmt="%.2f",fontsize=8,padding=3)
        ax.set(xticks=x,xticklabels=models,ylabel=label,title="Held-out observed-leader replay")
        ax.set_ylim(0,ax.get_ylim()[1]*1.16);ax.legend(fontsize=7);ax.grid(axis="y",alpha=.12)
    stress=pd.read_csv(out/"braking_sensitivity.csv")
    for model in models:
        p=stress[stress.model==model]
        axes[1,0].plot(p.leader_deceleration_m_s2,p.minimum_gap_m,marker="o",ms=4,color=COLORS[model],label=model)
        fail=p[p.status=="collision"]
        if model=="OVM":
            axes[1,0].scatter(fail.leader_deceleration_m_s2,fail.minimum_gap_m,marker="o",s=80,facecolors="none",edgecolors=COLORS[model],zorder=6)
        else:
            axes[1,0].scatter(fail.leader_deceleration_m_s2,fail.minimum_gap_m,marker="x",s=45,color=COLORS[model],zorder=7)
    axes[1,0].set(xlabel="Leader braking magnitude (m/s²)",ylabel="Minimum accepted gap (m)",title="Braking sensitivity: crosses are collisions",ylim=(-.3,5))
    axes[1,0].legend(fontsize=7);axes[1,0].grid(alpha=.15)
    convergence=pd.read_csv(out/"convergence.csv")
    for model in models:
        p=convergence[(convergence.model==model)&(convergence.scenario=="observed")].sort_values("step_s")
        axes[1,1].plot(p.step_s,p.max_position_difference_vs_fine_m,marker="o",color=COLORS[model],label=model)
    axes[1,1].set(xlabel="Maximum integration step (s)",ylabel="Maximum position difference (m)",title="Observed replay vs 0.025 s reference",xticks=[.025,.05,.1])
    axes[1,1].legend(fontsize=7);axes[1,1].grid(alpha=.15)
    for i,ax in enumerate(axes.flat):style.panel_label(ax,chr(97+i),x=-.13,y=1.03)
    fig.text(.08,.034,"Single-pair results describe this driver and regime. The fitted IDM b reaches its 4 m/s² upper bound; parameter identifiability is limited.",fontsize=8.5,color="#53636B")
    save(fig,out,"robustness")

    fig,axes=plt.subplots(1,3,figsize=(12.6,4.7))
    fig.subplots_adjust(left=.065,right=.98,top=.80,bottom=.19,wspace=.30)
    heading(fig,"What information changes each model's response?",
            "Formula-based illustrations in SI units; these curves are not additional NGSIM observations.",.065,subtitle_y=.87)
    gap=np.linspace(.1,35,250)
    for model in ["OVM","FVDM"]:
        axes[0].plot(gap,optimal_velocity(gap,parameters["calibrated"][model]),color=COLORS[model],label=model)
    axes[0].plot(gap,optimal_velocity(gap,DEFAULTS["OVM"]),color="#64727C",ls="--",label="Common baseline V")
    axes[0].axhline(0,color="#A2AAAE",lw=.8)
    axes[0].set(xlabel="Net gap (m)",ylabel="Optimal speed (m/s)",title="Spacing-dependent desired speed");axes[0].legend(fontsize=7)
    closing=np.linspace(-5,5,200)
    for model in models:
        a=acceleration(model,10.,6.,6.-closing,parameters["calibrated"][model])
        axes[1].plot(closing,a,color=COLORS[model],label=model)
    axes[1].axvline(0,color="#A2AAAE",lw=.8)
    axes[1].set(xlabel="Closing speed v - v_leader (m/s)",ylabel="Acceleration (m/s²)",title="Fixed net gap 10 m, follower speed 6 m/s")
    axes[1].legend(fontsize=7)
    velocity=np.linspace(0,12,200)
    for model in models:
        p=parameters["calibrated"][model]
        if model=="IDM":
            seq=(p["s0"]+velocity*p["T"])/np.sqrt(1-(velocity/p["v0"])**p["delta"])
        else:
            argument=(velocity-p["V1"])/p["V2"]
            seq=(p["C2"]+np.arctanh(argument))/p["C1"]
        axes[2].plot(velocity,seq,color=COLORS[model],label=model)
    axes[2].set(xlabel="Equilibrium speed (m/s)",ylabel="Equilibrium net gap (m)",title="Equal-speed equilibrium assumptions")
    axes[2].legend(fontsize=7)
    for i,ax in enumerate(axes):ax.grid(alpha=.15);style.panel_label(ax,chr(97+i),x=-.15,y=1.04)
    fig.text(.065,.045,"OVM: spacing relaxation. FVDM: additional relative-speed feedback. IDM: free acceleration and speed/closing-dependent desired spacing.",fontsize=8.5,color="#53636B")
    save(fig,out,"mechanisms")

    fig,axes=plt.subplots(3,1,figsize=(10.8,7.6),sharex=True)
    fig.subplots_adjust(left=.085,right=.97,top=.84,bottom=.10,hspace=.27)
    heading(fig,"A stable leader pair, selected before model simulation",
            "NGSIM SI source | lane 2 | front-centre coordinates | measured leader length 3.017526 m",.085)
    t=data.relative_time_s
    for ax in axes:ax.axvspan(0,29.9,color="#D7E8F1",alpha=.65);ax.axvline(30,color="#53636B",ls="--",lw=1);ax.grid(alpha=.15)
    axes[0].plot(t,data.leader_x_m,color="#243743",label="Leader 368")
    axes[0].plot(t,data.follower_x_m,color="#0072B2",label="Follower 378")
    axes[0].set_ylabel("Front position (m)");axes[0].legend(ncol=2)
    axes[1].plot(t,data.leader_v_raw_m_s,color="#243743",label="Leader raw speed")
    axes[1].plot(t,data.follower_v_raw_m_s,color="#0072B2",label="Follower raw speed")
    axes[1].set_ylabel("Speed (m/s)");axes[1].legend(ncol=2,fontsize=7)
    axes[2].plot(t,data.net_gap_m,color="#755299")
    axes[2].set(ylabel="Net gap (m)",xlabel="Seconds after source time 50.9 s",xlim=(0,60))
    fig.text(.085,.905,"Blue shading: 300 calibration observations. Unshaded: 301 held-out observations, with initialization only at t = 30 s.",fontsize=9,color="#53636B")
    fig.text(.085,.027,"Nearest observed same-lane leader is inferred at every frame; the pair is stable throughout. No smoothing, unit reconversion or model-based case selection.",fontsize=8.3,color="#53636B")
    save(fig,out,"data_protocol")
