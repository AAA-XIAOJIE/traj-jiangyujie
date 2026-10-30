"""Run from repository root: python -X utf8 -m week07.model.run."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import osqp
from .core import fibonacci_sphere
from .simulation import simulate

ROOT = Path(__file__).resolve().parents[2]
WEEK = ROOT/'week07'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(out):
    # Git stores LF text. Hash canonical bytes so archives and Windows/Linux
    # checkouts validate the same evidence instead of platform newline changes.
    text_outputs=['metrics.csv','agent_metrics.csv','initial_conditions.csv',
                  'time_series.csv','sensitivity.csv','experiment.json','failure_diagnostics.json']
    for name in text_outputs:
        path=out/name
        if path.exists():
            path.write_text(path.read_text(encoding='utf8'),encoding='utf8',newline='\n')
    receipt = {'python': sys.version, 'platform': platform.platform(),
               'numpy': np.__version__, 'scipy': scipy.__version__, 'osqp': osqp.__version__,
               'config_sha256': digest(WEEK/'config.json'),
               'numerical_source_sha256': {p.name:digest(p) for p in [WEEK/'model/core.py', WEEK/'model/simulation.py']},
               'output_sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='reproducibility.json'}}
    (out/'reproducibility.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')


def studies(cfg):
    records = []
    for name in cfg['models']:
        variant = copy.deepcopy(cfg); variant['dt_s'] = cfg['dt_s']/2
        row, _ = simulate(variant, name)
        row.update(study='half_timestep',dt_s=variant['dt_s'],swirl_speed_m_s=variant['cbf']['swirl_speed_m_s'] if name=='cbf' else None)
        records.append(row)
        print('study', name, 'half dt', row['arrived'], row['minimum_separation_m'],flush=True)
    variant=copy.deepcopy(cfg);variant['cbf']['swirl_speed_m_s']=0
    row, _=simulate(variant,'cbf')
    row.update(study='without_circulation',dt_s=variant['dt_s'],swirl_speed_m_s=0)
    records.append(row)
    print('study cbf without circulation', row['arrived'], row['minimum_separation_m'],flush=True)
    return pd.DataFrame(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WEEK/'results')
    parser.add_argument('--no-animation',action='store_true')
    parser.add_argument('--no-studies',action='store_true')
    parser.add_argument('--no-figures',action='store_true')
    parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((WEEK/'config.json').read_text(encoding='utf8'))
    if args.figures_only:
        receipt=json.loads((out/'reproducibility.json').read_text(encoding='utf8'))
        assert receipt['config_sha256']==digest(WEEK/'config.json'), 'Configuration changed: rerun simulations.'
        for name,h in receipt['numerical_source_sha256'].items():
            assert digest(WEEK/'model'/name)==h, 'Numerical code changed: rerun simulations.'
        metrics=pd.read_csv(out/'metrics.csv')
        archive=np.load(out/'trajectories.npz')
        traces={name:{key.split('__')[1]:archive[key] for key in archive.files if key.startswith(name+'__')} for name in cfg['models']}
    else:
        rows=[];traces={}
        for name in cfg['models']:
            row,trace=simulate(cfg,name,verbose=True)
            rows.append(row);traces[name]=trace
            print(json.dumps(row,ensure_ascii=False),flush=True)
        metrics=pd.DataFrame(rows)
        metrics.to_csv(out/'metrics.csv',index=False,float_format='%.10g')
        np.savez_compressed(out/'trajectories.npz',**{name+'__'+key:value for name,t in traces.items() for key,value in t.items()})
        p=cfg['sphere_radius_m']*fibonacci_sphere(cfg['agents'])
        initial=pd.DataFrame(np.c_[np.arange(cfg['agents']),p,-p,np.zeros_like(p)],columns=['agent_id','x_m','y_m','z_m','goal_x_m','goal_y_m','goal_z_m','vx_m_s','vy_m_s','vz_m_s'])
        initial['agent_id']=initial['agent_id'].astype(int)
        initial.to_csv(out/'initial_conditions.csv',index=False,float_format='%.12g')
        agents=[];histories=[]
        for name,t in traces.items():
            for i in range(cfg['agents']):
                agents.append({'model':name,'agent_id':i,'arrival_time_s':t['arrival_times'][i],'path_length_m':t['path_lengths'][i],
                               'final_goal_distance_m':float(np.linalg.norm(t['positions'][-1,i]-t['goals'][i]))})
            for k,time in enumerate(t['times'][1:]):
                histories.append({'model':name,'time_s':time,'arrived':t['arrival_counts'][k],
                                  'interval_minimum_separation_m':t['minimum_separation'][k]})
        pd.DataFrame(agents).to_csv(out/'agent_metrics.csv',index=False,float_format='%.10g')
        pd.DataFrame(histories).to_csv(out/'time_series.csv',index=False,float_format='%.10g')
        if not args.no_studies:
            studies(cfg).to_csv(out/'sensitivity.csv',index=False,float_format='%.10g')
        settings=copy.deepcopy(cfg)
        settings.update(initialization='Fibonacci sphere: equal-area z bands and golden-angle azimuth',
                        simulation_type='single-integrator kinematics, direct velocity commands',
                        no_mass_force_acceleration_model=True,
                        stopped_agents_remain_obstacles=True,
                        intrusion_definition='unique unordered pairs with continuous-step center distance < 1 m minus 1e-6 m tolerance',
                        shortest_direct_path_to_goal_region_m=2*cfg['sphere_radius_m']-cfg['goal_tolerance_m'])
        (out/'experiment.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if not args.no_figures:
        from .figures import make_figures, make_animation
        make_figures(cfg,metrics,traces,out)
        if not args.no_animation:
            make_animation(cfg,traces,out)
    from .diagnostics import encounter_diagnostics
    (out/'failure_diagnostics.json').write_text(json.dumps(encounter_diagnostics(cfg,traces),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    save_manifest(out)
    print('Saved',out,flush=True)


if __name__=='__main__':
    main()
