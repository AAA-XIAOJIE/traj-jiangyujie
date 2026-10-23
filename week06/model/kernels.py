"""One interaction kernel per experiment, used for every traffic participant.

SFM is a circular exponential acceleration-form simplification, not a verbatim
reproduction of Helbing's elliptical-potential 1995 model. IDM is an explicit
goal-heading extension to 2D, using the modern nonnegative dynamic-gap term.
ORCA reuses Week02's Apache-2.0 solver; directional body supports below are a
heuristic for rectangles, so original circular-agent guarantees do not transfer.
"""
import numpy as np
from week02.model.orca import lp2, lp3
from .geometry import EPS,ray_headways


def social_force(p,v,pref,scene,gaps,normals,cfg):
    a=(pref-v)/scene["tau"][:,None]
    magnitude=cfg["sfm"]["A_m_s2"]*np.exp(np.clip(-gaps/cfg["sfm"]["B_m"],-60,8))
    a+=np.sum(magnitude[...,None]*normals,axis=1)
    if scene["width"] is not None:
        halfwidth=np.where(scene["extent"][:,0]>0,scene["extent"][:,1],scene["radius"])
        top=scene["width"]/2-p[:,1]-halfwidth
        bottom=scene["width"]/2+p[:,1]-halfwidth
        wall_B=cfg["sfm"]["wall_B_m"]
        a[:,1]+=cfg["sfm"]["wall_A_m_s2"]*(np.exp(np.clip(-bottom/wall_B,-60,8))-np.exp(np.clip(-top/wall_B,-60,8)))
    return a


def idm_heading(p,v,pref,scene,gaps,normals,cfg):
    desired=np.linalg.norm(pref,axis=1)
    heading=pref/np.maximum(desired[:,None],EPS)
    empty=desired<EPS
    heading[empty]=scene["heading"][empty]
    speed=np.maximum(np.sum(v*heading,axis=1),0)
    candidates=ray_headways(p,heading,scene["radius"],scene["extent"],cfg["idm"]["lateral_buffer_m"])
    candidates[:,~scene["active"]]=np.inf
    neighbor=np.argmin(candidates,axis=1);s=candidates[np.arange(len(p)),neighbor]
    leader_speed=np.sum(v[neighbor]*heading,axis=1)
    closing=speed-leader_speed
    amax=scene["accel"];brake=scene["brake"]
    target=cfg["idm"]["s0_m"]+np.maximum(0,speed*cfg["idm"]["T_s"]+speed*closing/(2*np.sqrt(amax*brake)))
    scalar=amax*(1-(speed/np.maximum(desired,.05))**cfg["idm"]["delta"]-(target/np.maximum(s,.02))**2)
    scalar=np.where(empty,-speed/scene["tau"],scalar)
    # Same goal-heading dynamics for all; the base heading has no avoidance steering.
    perpendicular=v-speed[:,None]*heading
    return scalar[:,None]*heading-perpendicular/scene["tau"][:,None]


def orca_velocity(p,v,pref,scene,cfg,dt):
    n=len(p);rp=p[None,:,:]-p[:,None,:];rv=v[:,None,:]-v[None,:,:]
    d2=np.sum(rp**2,axis=-1);dist=np.sqrt(np.maximum(d2,EPS))
    u=rp/dist[...,None]
    support_i=np.broadcast_to(scene["radius"][:,None],(n,n)).copy()
    support_j=np.broadcast_to(scene["radius"][None,:],(n,n)).copy()
    for i in np.flatnonzero(scene["extent"][:,0]>0):
        support_i[i]=np.abs(u[i])@scene["extent"][i]
        support_j[:,i]=np.abs(u[:,i])@scene["extent"][i]
    combined=support_i+support_j
    horizon=cfg["orca"]["horizon_s"]
    w=rv-rp/horizon;w2=np.sum(w*w,axis=-1);dot=np.sum(w*rp,axis=-1)
    cut=(dot<0)&(dot**2>combined**2*w2)
    length=np.sqrt(np.maximum(w2,EPS));normal=w/length[...,None]
    directions=np.stack([normal[...,1],-normal[...,0]],axis=-1)
    correction=(combined/horizon-length)[...,None]*normal
    leg=np.sqrt(np.maximum(d2-combined**2,0));x,y=rp[...,0],rp[...,1]
    sign=np.where(x*w[...,1]-y*w[...,0]>0,1.,-1.)
    legs=np.stack([sign*x*leg-y*combined,x*combined+sign*y*leg],axis=-1)/np.maximum(d2,EPS)[...,None]
    leg_u=np.sum(rv*legs,axis=-1)[...,None]*legs-rv
    directions=np.where(cut[...,None],directions,legs);correction=np.where(cut[...,None],correction,leg_u)
    overlap=d2<=combined**2;ow=rv-rp/dt;ol=np.linalg.norm(ow,axis=-1)
    ou=ow/np.maximum(ol[...,None],EPS)
    od=np.stack([ou[...,1],-ou[...,0]],axis=-1);oc=(combined/dt-ol)[...,None]*ou
    directions=np.where(overlap[...,None],od,directions);correction=np.where(overlap[...,None],oc,correction)
    points=v[:,None,:]+.5*correction
    result=np.zeros_like(v);fallback=0
    for i in range(n):
        if not scene["active"][i]:continue
        order=np.argsort(d2[i],kind="stable");order=order[order!=i]
        order=order[scene["active"][order]]
        lines=list(zip(points[i,order],directions[i,order]))
        failed,candidate=lp2(lines,scene["max_speed"][i],pref[i])
        if failed<len(lines):
            fallback+=1;candidate=lp3(lines,failed,scene["max_speed"][i],candidate)
        result[i]=candidate
    return result,fallback
