"""Fixed synthetic scene and audited reuse of the Week02 circle observation."""
import json
from pathlib import Path
import numpy as np
from week02.model.analysis import read_experiment,initial_conditions

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def scene_arrays(p,v,goals,desired,radius,extent,lateral,cfg,name,width,duration):
    d=cfg["dynamics"];fixed=~lateral
    tau=np.where(fixed,d["car_tau_s"],d["pedestrian_tau_s"])
    accel=np.where(fixed,d["car_accel_m_s2"],d["pedestrian_accel_m_s2"])
    brake=np.where(fixed,d["car_brake_m_s2"],d["pedestrian_accel_m_s2"])
    heading=(goals-p)/np.linalg.norm(goals-p,axis=1)[:,None]
    side=np.where(np.random.default_rng(cfg["seed"]).random(len(p))<cfg["guidance"]["right_probability"],1.,-1.)
    return dict(p0=p,v0=v,goals=goals,desired=desired,radius=radius,extent=extent,lateral=lateral,
        tau=tau,accel=accel,brake=brake,heading=heading,side=side,
        max_speed=np.maximum(desired*d["max_speed_factor"],np.linalg.norm(v,axis=1)),
        name=name,width=width,duration=duration)


def narrow(cfg,width):
    c=cfg["narrow"];p=np.array(c["pedestrian_positions_m"]+[c["car_position_m"]],float);n=len(p)
    desired=np.r_[np.full(n-1,c["pedestrian_speed_m_s"]),c["car_speed_m_s"]]
    v=np.column_stack([desired*np.r_[np.ones(n-1),-1.],np.zeros(n)])
    goals=p.copy();goals[:-1,0]=40.;goals[-1,0]=-10.
    radius=np.r_[np.full(n-1,c["pedestrian_radius_m"]),0.]
    extent=np.zeros_like(p);extent[-1]=[c["car_length_m"]/2,c["car_width_m"]/2]
    lateral=np.r_[np.ones(n-1,bool),False]
    return scene_arrays(p,v,goals,desired,radius,extent,lateral,cfg,"narrow",width,c["duration_s"])


def circle(cfg):
    previous=json.loads((REPO/"week02/model/config.json").read_text(encoding="utf8"))
    exp=read_experiment(REPO/"week02/model/data/circle-10m-64-1.txt",previous["fps"],previous["coordinate_scale_to_m"])
    p,v,desired=initial_conditions(exp,previous);n=len(p)
    scene=scene_arrays(p,v,exp["goals"],desired,np.full(n,.25),np.zeros_like(p),
        np.ones(n,bool),cfg,"circle",None,float(exp["times"][-1]))
    return scene,exp,previous


def mixed_circle(cfg,center):
    """Strictly alternating walkers/cars; conservative car circumdisks, no truth."""
    c=cfg["mixed_circle"];n=c["agents"]
    assert n%2==0
    angle=np.arange(n)*2*np.pi/n
    p=center+c["radius_m"]*np.column_stack([np.cos(angle),np.sin(angle)])
    goals=2*center-p;lateral=np.arange(n)%2==1
    extent=np.zeros_like(p)
    car_radius=np.hypot(c["car_length_m"]/2,c["car_width_m"]/2)
    radius=np.where(lateral,.25,car_radius);desired=np.where(lateral,1.4,3.)
    heading=(goals-p)/np.linalg.norm(goals-p,axis=1)[:,None]
    return scene_arrays(p,heading*desired[:,None],goals,desired,radius,extent,lateral,cfg,
        "circle_mixed",None,c["duration_s"])
