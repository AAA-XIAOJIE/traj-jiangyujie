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
    encountered=np.zeros(len(pairs[0]),bool);minimum=np.inf;count=0
    for k in range(len(p)-1):
        d=swept_distances(p[k],(p[k+1]-p[k])/dt,dt,pairs)
        bad=d<1-CFG['distance_tolerance_m'];encountered|=bad;count+=int(bad.sum());minimum=min(minimum,d.min())
    assert minimum==pytest.approx(row.minimum_separation_m,abs=1e-7)
    assert int(encountered.sum())==row.violating_unique_pairs
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
