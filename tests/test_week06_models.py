"""Geometric and behavioral contracts for the shared interaction experiments."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from week06.model.geometry import pair_geometry,swept_gaps,ray_headways
from week06.model.scenarios import narrow,circle,mixed_circle
from week06.model.simulation import simulate,constrain_velocity,guard
from week06.model.kernels import social_force,idm_heading

ROOT=Path(__file__).resolve().parents[1]/"week06"
CFG=json.loads((ROOT/"config.json").read_text(encoding="utf8"))


def test_circle_rectangle_gap_and_corner_distance():
    p=np.array([[0.,0.],[3.,0.],[0.,2.],[3.,2.]])
    extent=np.zeros_like(p);extent[0]=[2.,1.]
    radii=np.array([0.,.25,.25,.25])
    gap,n=pair_geometry(p,radii,extent)
    np.testing.assert_allclose(gap[0,1:],[.75,.75,np.sqrt(2)-.25])
    np.testing.assert_allclose(gap,gap.T)
    np.testing.assert_allclose(n,-n.transpose(1,0,2))


def test_swept_rectangle_catches_tunneling_between_valid_endpoints():
    p=np.array([[0.,0.],[-4.,0.]]);end=np.array([[0.,0.],[4.,0.]])
    r=np.array([0.,.25]);e=np.array([[2.,1.],[0.,0.]])
    assert pair_geometry(p,r,e)[0][0,1]>0
    assert pair_geometry(end,r,e)[0][0,1]>0
    assert swept_gaps(p,end,r,e)[0]==pytest.approx(-1.25)


def test_swept_circle_and_rectangle_tangent_are_not_penetrations():
    p=np.array([[0.,0.],[-4.,1.25]]);end=np.array([[0.,0.],[4.,1.25]])
    assert swept_gaps(p,end,np.array([0.,.25]),np.array([[2.,1.],[0.,0.]]))[0]==pytest.approx(0.)
    p=np.array([[-1.,0.],[1.,0.]]);end=-p
    assert swept_gaps(p,end,np.array([.25,.25]),np.zeros((2,2)))[0]==pytest.approx(-.5)


def test_two_rectangles_use_both_vehicle_extents():
    p=np.array([[0.,0.],[5.,0.]]);r=np.zeros(2);e=np.array([[2.,1.],[2.,1.]])
    assert pair_geometry(p,r,e)[0][0,1]==pytest.approx(1.)
    end=np.array([[5.,0.],[0.,0.]])
    assert swept_gaps(p,end,r,e)[0]==pytest.approx(-2.)


def test_ray_projection_respects_clear_side_and_negative_longitudinal_direction():
    p=np.array([[0.,0.],[5.,0.],[0.,5.]])
    heading=np.array([[1.,0.],[-1.,0.],[0.,1.]])
    r=np.full(3,.25);e=np.zeros_like(p)
    gaps=ray_headways(p,heading,r,e)
    assert gaps[0,1]==pytest.approx(4.5)
    assert np.isinf(gaps[0,2]) and np.isinf(gaps[2,0])


def test_width_is_the_only_input_changed_in_main_narrow_comparison():
    a=narrow(CFG,4.2);b=narrow(CFG,3.2)
    for key in a:
        if key=="width":continue
        if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key])
        else:assert a[key]==b[key]
    assert CFG["narrow"]["car_width_m"]+4*CFG["narrow"]["pedestrian_radius_m"]==pytest.approx(2.8)


def test_initial_scenes_are_geometrically_valid():
    ring,obs,_=circle(CFG)
    for s in [narrow(CFG,2.6),ring,mixed_circle(CFG,obs["center"])]:
        gaps,_=pair_geometry(s["p0"],s["radius"],s["extent"])
        assert gaps.min()>0
    assert len(ring["p0"])==64


def test_car_never_gets_lateral_velocity_or_reverse_velocity():
    scene=narrow(CFG,3.2);v=np.zeros_like(scene["v0"])
    candidate=np.full_like(v,10.)
    out,_=constrain_velocity(candidate,v,scene,.04)
    assert out[-1,1]==0 and out[-1,0]<=0
    candidate[-1]=[-10.,10.]
    out,_=constrain_velocity(candidate,v,scene,.04)
    assert out[-1,1]==0 and out[-1,0]==pytest.approx(-.06)


@pytest.mark.parametrize("model",["sfm","idm","orca"])
def test_short_rollouts_are_finite_and_car_stays_on_axis(model):
    scene=narrow(CFG,3.2);scene["duration"]=1.
    result=simulate(scene,CFG,model,"guided")
    assert np.isfinite(result["positions"]).all()
    np.testing.assert_allclose(result["positions"][:,-1,1],0,atol=1e-12)
    assert (result["velocities"][:,-1,0]<=1e-12).all()


def test_mixed_scenario_has_no_empirical_reference_and_no_initial_overlap():
    _,obs,_=circle(CFG);s=mixed_circle(CFG,obs["center"])
    assert s["name"]=="circle_mixed" and len(s["p0"])==64
    assert (~s["lateral"]).sum()==32
    np.testing.assert_array_equal(s["lateral"],np.arange(64)%2==1)
    np.testing.assert_allclose(np.linalg.norm(s["p0"]-obs["center"],axis=1),30)
    assert "observations" not in s


def test_turning_is_only_available_in_the_improved_mixed_model():
    _,obs,_=circle(CFG);s=mixed_circle(CFG,obs["center"]);s["duration"]=1.
    straight=simulate(s,CFG,"sfm","guided")
    turning=simulate(s,CFG,"sfm","steering")
    cars=~s["lateral"]
    v=straight["velocities"][:,cars];h=s["heading"][cars]
    np.testing.assert_allclose(v[...,0]*h[:,1]-v[...,1]*h[:,0],0,atol=1e-9)
    assert turning["metrics"]["max_car_yaw_rate_rad_s"]>0
    assert turning["metrics"]["steering_bound_violation_agent_steps"]==0


def test_mutual_reaction_same_force_formula():
    scene=narrow(CFG,3.2)
    scene["active"]=np.ones(7,bool)
    p=scene["p0"].copy();p[-1]=[12.,0.]
    gap,n=pair_geometry(p,scene["radius"],scene["extent"])
    isolated=np.full_like(gap,np.inf);isolated[4,-1]=gap[4,-1];isolated[-1,4]=gap[-1,4]
    pref=scene["v0"].copy()
    a=social_force(p,scene["v0"],pref,scene,isolated,n,CFG)
    # Opponents both react: forward walkers decelerate, left-going vehicle brakes.
    assert a[4,0]<0 and a[-1,0]>0
