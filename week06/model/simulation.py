"""Synchronous SI integration, explicit capability constraints and diagnostic failures."""
import numpy as np
from .geometry import unit,pair_geometry,swept_gaps,lateral_clearance,EPS
from .kernels import social_force,idm_heading,orca_velocity


def preferred(p,v,scene,cfg,t,guided):
    toward=scene["goals"]-p;remaining=np.linalg.norm(toward,axis=1)
    heading=unit(toward)
    desired=scene["desired"]+(np.linalg.norm(scene["v0"],axis=1)-scene["desired"])*np.exp(-t/.5)
    pref=heading*np.minimum(desired,remaining/.5)[:,None]
    if not guided:return pref,0
    if scene["width"] is None:
        rel=p[None,:,:]-p[:,None,:];rv=pref[:,None,:]-v[None,:,:]
        tca=np.sum(rel*rv,axis=-1)/np.maximum(np.sum(rv*rv,axis=-1),EPS)
        horizon=cfg["guidance"]["anticipation_s"]
        miss=np.linalg.norm(rel-np.clip(tca,0,horizon)[...,None]*rv,axis=-1)
        risk=np.exp(-(miss/1.)**2)*np.clip(1-tca/horizon,0,1)
        risk*=(tca>0)&(tca<horizon)&(np.sum(rel*heading[:,None,:],axis=-1)>0)
        np.fill_diagonal(risk,0)
        angle=scene["side"]*cfg["guidance"]["side_bias_rad"]*risk.max(axis=1)
        right=np.column_stack([heading[:,1],-heading[:,0]])
        pref=(np.cos(angle)[:,None]*heading+np.sin(angle)[:,None]*right)*np.minimum(desired,remaining/.5)[:,None]
        return pref,0
    waits=0
    # Mobility-aware waypoint: the same rule uses the neighbor's footprint and
    # lateral capability. It does not switch to a different human/car kernel.
    for j in np.flatnonzero(~scene["lateral"]&scene["active"]):
        for i in np.flatnonzero(scene["lateral"]&scene["active"]):
            forward=p[j,0]+scene["extent"][j,0]-p[i,0]
            if not (-.8<forward<15):continue
            max_y=scene["width"]/2-scene["radius"][i]-cfg["guidance"]["wall_margin_m"]
            min_y=scene["extent"][j,1]+scene["radius"][i]
            if max_y<=min_y+.005:
                if forward<11:
                    pref[i]=0;pref[j]=0;waits+=1
                continue
            margin=min(cfg["guidance"]["passing_margin_m"],.5*(max_y-min_y))
            side=1. if scene["p0"][i,1]>=0 else -1.
            target_y=side*(min_y+margin)
            # A stalled longitudinal model needs a lateral feasible heading:
            # a shallow diagonal still treats the same car as its blocking leader.
            lookahead=0. if abs(p[i,1])<min_y+margin-.01 else 3.0
            waypoint=np.array([p[i,0]+lookahead,target_y])
            pref[i]=unit(waypoint-p[i])*desired[i]
        # Both participants react: no forward advance into an occupied straight corridor.
        direction=scene["heading"][j,0]
        ahead=(p[:,0]-p[j,0])*direction
        side_overlap=np.abs(p[:,1]-p[j,1])<scene["extent"][j,1]+scene["radius"]+.10
        relevant=(ahead>0)&side_overlap&scene["active"];relevant[j]=False
        if relevant.any():
            gap=np.min(ahead[relevant]-scene["extent"][j,0]-scene["radius"][relevant])
            cap=np.sqrt(2*scene["brake"][j]*max(0,gap-.8))
            pref[j]*=min(1.,cap/max(np.linalg.norm(pref[j]),EPS))
    return pref,waits


def constrain_velocity(candidate,v,scene,dt):
    """Every agent uses its admissible velocity/acceleration set; car cannot sidestep."""
    before=candidate.copy()
    fixed=~scene["lateral"];steering=scene.get("steering",np.zeros(len(v),bool))
    straight=fixed&~steering
    forward=np.sum(candidate[straight]*scene["heading"][straight],axis=1)
    candidate[straight]=np.maximum(forward,0)[:,None]*scene["heading"][straight]
    delta=candidate-v
    # Pedestrians have a vector acceleration bound; straight vehicles distinguish braking.
    cap=scene["accel"].copy()
    reference=scene["heading"].copy()
    moving=steering&(np.linalg.norm(v,axis=1)>1e-8)
    reference[moving]=unit(v[moving])
    slowing=np.sum(delta*reference,axis=1)<0
    cap[fixed&slowing]=scene["brake"][fixed&slowing]
    norm=np.linalg.norm(delta,axis=1)
    candidate=v+delta*np.minimum(1,cap*dt/np.maximum(norm,EPS))[:,None]
    speed=np.linalg.norm(candidate,axis=1)
    candidate*=np.minimum(1,scene["max_speed"]/np.maximum(speed,EPS))[:,None]
    if steering.any():
        c=scene["steering_config"]
        index=np.flatnonzero(steering)
        old_angle=np.arctan2(reference[index,1],reference[index,0])
        new_angle=np.arctan2(candidate[index,1],candidate[index,0])
        turn=(new_angle-old_angle+np.pi)%(2*np.pi)-np.pi
        speed=np.linalg.norm(candidate[index],axis=1)
        yaw=np.minimum(c["max_yaw_rate_rad_s"],speed*np.tan(c["max_steer_angle_rad"])/c["wheelbase_m"])
        yaw=np.minimum(yaw,c["max_lateral_accel_m_s2"]/np.maximum(speed,.1))
        angle=old_angle+np.clip(turn,-yaw*dt,yaw*dt)
        candidate[index]=speed[:,None]*np.column_stack([np.cos(angle),np.sin(angle)])
    return candidate,int(np.sum(np.linalg.norm(before-candidate,axis=1)>1e-7))


def guard(p,candidate,scene,dt):
    """Explicit pre-movement geometry veto, not a position repair or collision-free model claim."""
    result=candidate.copy();changed=np.zeros(len(p),bool)
    if scene["width"] is not None:
        half=np.where(scene["extent"][:,0]>0,scene["extent"][:,1],scene["radius"])
        bound=scene["width"]/2-half-.001
        lo=(-bound-p[:,1])/dt;hi=(bound-p[:,1])/dt
        old=result[:,1].copy();result[:,1]=np.clip(result[:,1],lo,hi)
        changed|=abs(old-result[:,1])>1e-10
    ii,jj=np.triu_indices(len(p),1)
    for _ in range(4):
        gap=swept_gaps(p,p+dt*result,scene["radius"],scene["extent"])
        gap[~(scene["active"][ii]&scene["active"][jj])]=np.inf
        bad=gap<.002
        if not bad.any():break
        involved=np.unique(np.r_[ii[bad],jj[bad]])
        changed[involved]|=np.linalg.norm(result[involved],axis=1)>EPS
        result[involved]=0
    return result,int(changed.sum())


def simulate(scene,cfg,model,variant,dt=None):
    scene={**scene,"active":np.ones(len(scene["p0"]),bool)}
    scene["steering"]=(~scene["lateral"])&(scene["name"]=="circle_mixed")&(variant=="steering")
    scene["steering_config"]=cfg["mixed_circle"]
    dt=cfg["dt_s"] if dt is None else dt
    duration=scene["duration"];steps=int(round(duration/dt))
    times=np.arange(steps+1)*dt;p=scene["p0"].copy();v=scene["v0"].copy()
    pos=np.empty((steps+1,len(p),2));vel=np.empty_like(pos);pos[0]=p;vel[0]=v
    pair_count=0;wall_count=0;guard_count=0;clip_count=0;fallback=0;waiting=0
    min_gap=float("inf");min_wall=float("inf");max_acc=0.;first_contact=None
    dynamic_violations=0;max_raw_acc=0.;max_yaw=0.;max_lateral=0.;steering_violations=0
    pair_i,pair_j=np.triu_indices(len(p),1)
    active_overlap=np.zeros(len(p)*(len(p)-1)//2,bool);events=0
    arrival=np.full(len(p),np.nan);halt_seconds=np.zeros(len(p))
    for k in range(steps):
        gaps,normals=pair_geometry(p,scene["radius"],scene["extent"])
        gaps[:,~scene["active"]]=np.inf
        pref,wait=preferred(p,v,scene,cfg,times[k],variant in ("guided","guided_no_veto","steering"));waiting+=wait
        pref[~scene["active"]]=0
        if model=="sfm":raw=v+dt*social_force(p,v,pref,scene,gaps,normals,cfg)
        elif model=="idm":
            raw=v+dt*idm_heading(p,v,pref,scene,gaps,normals,cfg)
            direction=unit(pref);forward=np.sum(raw*direction,axis=1)
            raw-=np.minimum(forward,0)[:,None]*direction
        elif model=="orca":
            raw,failed=orca_velocity(p,v,pref,scene,cfg,dt);fallback+=failed
        else:raise ValueError(model)
        raw[~scene["active"]]=0
        raw_acc=np.linalg.norm(raw-v,axis=1)/dt
        max_raw_acc=max(max_raw_acc,float(raw_acc[scene["active"]].max(initial=0)))
        new,clipped=constrain_velocity(raw,v,scene,dt);clip_count+=clipped
        if variant in ("guided","veto_only") and scene["width"] is not None:
            new,guarded=guard(p,new,scene,dt);guard_count+=guarded
        next_p=p+dt*new
        if not np.isfinite(next_p).all() or not np.isfinite(new).all():
            raise FloatingPointError(f"Nonfinite state in {scene['name']}/{model}/{variant} at step {k}")
        swept=swept_gaps(p,next_p,scene["radius"],scene["extent"])
        swept[~(scene["active"][pair_i]&scene["active"][pair_j])]=np.inf
        overlaps=swept< -1e-5
        events+=int(np.sum(overlaps&~active_overlap));active_overlap=overlaps
        pair_count+=int(overlaps.sum())
        if overlaps.any() and first_contact is None:first_contact=float(times[k])
        min_gap=min(min_gap,float(swept.min()))
        wall=lateral_clearance(next_p,scene["radius"],scene["extent"],scene["width"])
        wall[~scene["active"]]=np.inf
        min_wall=min(min_wall,float(wall.min()));wall_count+=int(np.sum(wall< -1e-6))
        actual_acc=np.linalg.norm(new-v,axis=1)/dt
        max_acc=max(max_acc,float(actual_acc[scene["active"]].max(initial=0)))
        cap=scene["accel"].copy()
        reference=scene["heading"].copy();moving=scene["steering"]&(np.linalg.norm(v,axis=1)>1e-8)
        reference[moving]=unit(v[moving])
        slowing=np.sum((new-v)*reference,axis=1)<0
        cap[(~scene["lateral"])&slowing]=scene["brake"][(~scene["lateral"])&slowing]
        dynamic_violations+=int(np.sum((actual_acc>cap+1e-5)&scene["active"]))
        if scene["steering"].any():
            idx=np.flatnonzero(scene["steering"])
            angle0=np.arctan2(reference[idx,1],reference[idx,0]);angle1=np.arctan2(new[idx,1],new[idx,0])
            speed=np.linalg.norm(new[idx],axis=1)
            yaw=np.abs((angle1-angle0+np.pi)%(2*np.pi)-np.pi)/dt
            yaw[speed<1e-8]=0
            lat=speed*yaw
            max_yaw=max(max_yaw,float(yaw.max()));max_lateral=max(max_lateral,float(lat.max()))
            c=cfg["mixed_circle"]
            steering_violations+=int(np.sum((yaw>c["max_yaw_rate_rad_s"]+1e-6)|(lat>c["max_lateral_accel_m_s2"]+1e-6)|
                (yaw>speed*np.tan(c["max_steer_angle_rad"])/c["wheelbase_m"]+1e-6)))
        remaining=np.linalg.norm(scene["goals"]-next_p,axis=1)
        if scene["width"] is None:
            reached=(remaining<.5)&np.isnan(arrival)
        else:
            reached=((scene["lateral"]&(next_p[:,0]-scene["radius"]>=cfg["narrow"]["road_length_m"]))|
                (~scene["lateral"]&(next_p[:,0]+scene["extent"][:,0]<=0)))&np.isnan(arrival)
        arrival[reached]=times[k+1]
        halt_seconds+=(np.linalg.norm(new,axis=1)<.1)*np.isnan(arrival)*dt
        if scene["width"] is not None:
            scene["active"][reached]=False
            new[~scene["active"]]=0
        p,v=next_p,new;pos[k+1]=p;vel[k+1]=v
    reached=np.isfinite(arrival);fixed=~scene["lateral"]
    metric=dict(model=model,variant=variant,scene=scene["name"],width_m=scene["width"],
        duration_s=duration,dt_s=dt,agents=len(p),arrived=int(reached.sum()),
        pedestrians_arrived=int(reached[~fixed].sum()),cars_arrived=int(reached[fixed].sum()),
        all_arrival_time_s=float(arrival.max()) if reached.all() else None,
        median_arrival_completed_s=float(np.nanmedian(arrival)) if reached.any() else None,
        minimum_body_gap_m=min_gap,body_overlap_pair_steps=pair_count,
        body_overlap_pair_seconds=pair_count*dt,overlap_episode_starts=events,first_overlap_interval_start_s=first_contact,
        wall_violation_agent_steps=wall_count,minimum_wall_clearance_m=min_wall if scene["width"] is not None else None,
        wall_violation_agent_seconds=wall_count*dt,
        geometry_veto_agent_steps=guard_count,capability_clipped_agent_steps=clip_count,orca_lp_fallback_agent_steps=fallback,
        geometry_veto_agent_seconds=guard_count*dt,
        max_realized_acceleration_m_s2=max_acc,max_requested_acceleration_m_s2=max_raw_acc,
        dynamic_bound_violation_agent_steps=dynamic_violations,mean_waiting_s=float(halt_seconds.mean()),
        max_car_yaw_rate_rad_s=max_yaw,max_car_lateral_accel_m_s2=max_lateral,
        steering_bound_violation_agent_steps=steering_violations,
        car_waiting_s=float(halt_seconds[fixed].sum()) if fixed.any() else None,
        passability_wait_pair_steps=waiting,
        geometry_valid_complete=bool(reached.all() and pair_count==0 and wall_count==0),
        physically_feasible_complete=bool(reached.all() and pair_count==0 and wall_count==0 and dynamic_violations==0 and steering_violations==0),
        finite=True)
    return dict(times=times,positions=pos,velocities=vel,arrival_times=arrival,metrics=metric)
