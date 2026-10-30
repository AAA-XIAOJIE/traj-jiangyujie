"""Independent metric reconstruction and safety tests for 3D kinematics."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('scipy', reason='Week07 has its own pinned simulation dependencies and CI job')
pytest.importorskip('osqp', reason='Week07 has its own pinned simulation dependencies and CI job')
from week07.model.core import fibonacci_sphere, swept_distances, CBF

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'week07/config.json').read_text())
OUT=Path(os.environ.get('WEEK07_RESULTS',ROOT/'week07/results'))


def test_fibonacci_sphere_is_surface_distribution_and_antipodes_are_distinct():
    p=20*fibonacci_sphere(100)
    np.testing.assert_allclose(np.linalg.norm(p,axis=1),20,atol=1e-12)
    np.testing.assert_allclose(np.sort(p[:,2]),np.linspace(-19.8,19.8,100),atol=1e-12)
    i,j=np.triu_indices(100,1)
    assert np.linalg.norm(p[i]-p[j],axis=1).min()>1.05
    assert len(np.unique(-p,axis=0))==100


def test_swept_check_catches_crossing_with_safe_endpoints():
    p=np.array([[-2.,0,0],[2.,0,0]])
    v=np.array([[4.,0,0],[-4.,0,0]])
    d=swept_distances(p,v,1.,(np.array([0]),np.array([1])))
    assert d[0]==pytest.approx(0)


def test_stationary_pair_and_parallel_motion():
    p=np.array([[0.,0,0],[0.,2,0]])
    for v in [np.zeros((2,3)),np.array([[2.,0,0],[2.,0,0]])]:
        assert swept_distances(p,v,.1,(np.array([0]),np.array([1])))[0]==pytest.approx(2)


def test_joint_barrier_respects_a_stopped_obstacle_and_speed():
    p=np.array([[-.8,0.,0],[.8,0,0]])
    goal=-p;stopped=np.array([False,True])
    ctl=CBF(CFG,p)
    for _ in range(12):
        v,diagnostic=ctl.step(p,np.zeros_like(p),goal,stopped)
        np.testing.assert_array_equal(v[1],0)
        assert np.linalg.norm(v,axis=1).max()<=2+1e-8
        assert swept_distances(p,v,.1,(np.array([0]),np.array([1]))).min()>=1.05-1e-6
        assert diagnostic['solver_failure']==0
        p+=.1*v


@pytest.mark.parametrize('name',['apf','vo','cbf'])
def test_published_metrics_reconstructed_from_positions(name):
    z=np.load(OUT/'trajectories.npz');p=z[name+'__positions'];t=z[name+'__times'];goals=z[name+'__goals']
    row=pd.read_csv(OUT/'metrics.csv').set_index('model').loc[name]
    dt=CFG['dt_s'];pairs=np.triu_indices(100,1)
    assert p.shape[1:]==(100,3) and np.isfinite(p).all()
    np.testing.assert_allclose(p[0],20*fibonacci_sphere(100),atol=1e-12)
    np.testing.assert_allclose(goals,-p[0],atol=1e-12)
    np.testing.assert_array_equal(z[name+'__velocities'][0],0)
    np.testing.assert_allclose(np.diff(t),dt,atol=1e-12)
    path=np.linalg.norm(np.diff(p,axis=0),axis=2)
    assert path.max()/dt<=2+1e-7
    assert path.sum(axis=0).mean()==pytest.approx(row.mean_path_length_m,abs=1e-6)
    np.testing.assert_allclose(path.sum(axis=0),z[name+'__path_lengths'],atol=1e-9)
    encountered=np.zeros(len(pairs[0]),bool);minimum=np.inf;count=0
    for k in range(len(p)-1):
        # Independent 3D point-to-segment geometry, without the simulator helper:
        # use endpoint distances unless the perpendicular foot lies inside;
        # interior distance equals triangle area / segment length.
        q0=p[k,pairs[0]]-p[k,pairs[1]]
        q1=p[k+1,pairs[0]]-p[k+1,pairs[1]]
        segment=q1-q0;norm2=np.sum(segment*segment,axis=1)
        dot=np.sum(q0*segment,axis=1)
        inside=(dot<0)&(-dot<norm2)&(norm2>0)
        d=np.minimum(np.linalg.norm(q0,axis=1),np.linalg.norm(q1,axis=1))
        d[inside]=np.linalg.norm(np.cross(q0[inside],q1[inside]),axis=1)/np.sqrt(norm2[inside])
        bad=d<1-CFG['distance_tolerance_m'];encountered|=bad;count+=int(bad.sum());minimum=min(minimum,d.min())
    assert minimum==pytest.approx(row.minimum_separation_m,abs=1e-7)
    assert int(encountered.sum())==row.violating_unique_pairs
    np.testing.assert_array_equal(z[name+'__violation_pairs'],np.column_stack(pairs)[encountered])
    assert count==row.violating_pair_steps
    arrival=[]
    for i in range(100):
        indices=np.flatnonzero(np.linalg.norm(p[:,i]-goals[i],axis=1)<=1+1e-10)
        if len(indices):
            first=indices[0];arrival.append(t[first])
            np.testing.assert_allclose(p[first:,i],np.broadcast_to(p[first,i],p[first:,i].shape),atol=1e-12)
    assert len(arrival)==row.arrived
    if len(arrival)==100:assert max(arrival)==pytest.approx(row.all_arrival_time_s)
    else:assert pd.isna(row.all_arrival_time_s)
    assert bool(row.collision_free_complete)==(len(arrival)==100 and not encountered.any())


def test_partial_rerun_rejects_stale_outputs(tmp_path):
    from types import SimpleNamespace
    from week07.model.run import check_output_reuse
    (tmp_path/'comparison.gif').write_bytes(b'old animation')
    args=SimpleNamespace(figures_only=False,no_figures=False,no_animation=True,no_studies=False)
    with pytest.raises(ValueError,match='retain old outputs'):
        check_output_reuse(tmp_path,args)
    assert (tmp_path/'comparison.gif').read_bytes()==b'old animation'


def test_figures_only_rejects_tampered_saved_data(tmp_path):
    import shutil
    from week07.model.run import verify_saved_results
    receipt=json.loads((OUT/'reproducibility.json').read_text())
    receipt['output_sha256']={k:v for k,v in receipt['output_sha256'].items() if k in ('metrics.csv','trajectories.npz')}
    for name in receipt['output_sha256']:
        shutil.copy2(OUT/name,tmp_path/name)
    (tmp_path/'reproducibility.json').write_text(json.dumps(receipt))
    verify_saved_results(tmp_path)
    with (tmp_path/'metrics.csv').open('a') as f:
        f.write('\nchanged')
    with pytest.raises(ValueError,match='metrics.csv'):
        verify_saved_results(tmp_path)


def test_timeout_is_incomplete_and_still_plots(tmp_path):
    from copy import deepcopy
    from week07.model.simulation import simulate
    from week07.model.figures import metrics_figure
    cfg=deepcopy(CFG);cfg['agents']=6;cfg['duration_s']=.2
    rows=[]
    for name in ('apf','vo','cbf'):
        row,trace=simulate(cfg,name)
        assert row['arrived']==0 and row['arrival_rate']==0
        assert row['all_arrival_time_s'] is None and not row['collision_free_complete']
        assert row['simulated_time_s']==pytest.approx(.2)
        assert np.isnan(trace['arrival_times']).all()
        rows.append(row)
    metrics_figure(pd.DataFrame(rows),tmp_path)
    assert (tmp_path/'metrics.png').stat().st_size>1000


def test_barrier_main_case_completes_safely():
    row=pd.read_csv(OUT/'metrics.csv').set_index('model').loc['cbf']
    assert row.arrived==100 and row.violating_unique_pairs==0
    assert row.minimum_separation_m>=1 and row.all_arrival_time_s<=120


def test_manifest_matches_sources_and_outputs():
    import hashlib
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=json.loads((OUT/'reproducibility.json').read_text())
    assert digest(ROOT/'week07/config.json')==receipt['config_sha256']
    for name,h in receipt['numerical_source_sha256'].items():assert digest(ROOT/'week07/model'/name)==h
    for name,h in receipt['output_sha256'].items():assert digest(OUT/name)==h
