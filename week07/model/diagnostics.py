"""Explain the first unsafe interval using only the stored state and commands."""
import numpy as np


def encounter_diagnostics(cfg, traces):
    reports={}
    for name,trace in traces.items():
        p=trace['positions'];dt=cfg['dt_s'];i,j=np.triu_indices(p.shape[1],1)
        report={'first_unsafe_interval':None}
        for k in range(len(p)-1):
            command=(p[k+1]-p[k])/dt
            q=p[k,j]-p[k,i];w=command[i]-command[j]
            close=np.clip(np.sum(q*w,axis=1)/np.maximum(np.sum(w*w,axis=1),1e-12),0,dt)
            distance=np.linalg.norm(q-close[:,None]*w,axis=1)
            bad=np.flatnonzero(distance<2*cfg['body_radius_m']-cfg['distance_tolerance_m'])
            if len(bad):
                index=bad[np.argmin(distance[bad])];a,b=int(i[index]),int(j[index])
                forecasts=[]
                for own,neighbor in [(a,b),(b,a)]:
                    rel=p[k,neighbor]-p[k,own]
                    relative_velocity=command[own]-trace['velocities'][k,neighbor]
                    ca=np.clip(np.dot(rel,relative_velocity)/max(np.dot(relative_velocity,relative_velocity),1e-12),0,cfg['horizon_s'])
                    forecasts.append(float(np.linalg.norm(rel-ca*relative_velocity)))
                report['first_unsafe_interval']={'step':k,'time_start_s':float(trace['times'][k]),
                      'ids':[a,b],'initial_separation_m':float(np.linalg.norm(q[index])),
                      'continuous_step_minimum_m':float(distance[index]),
                      'time_of_minimum_s':float(trace['times'][k]+close[index]),
                      'frozen_neighbor_velocity_prediction_m':forecasts,
                      'interpretation':'Prediction keeps the neighbor previous velocity; actual update changes both velocities simultaneously.'}
                break
        reports[name]=report
    return reports
