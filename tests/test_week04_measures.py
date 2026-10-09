"""Analytical and conservation checks for traffic measurement definitions."""
import numpy as np
import pandas as pd
import pytest

from week04.model.measures import (trajectory_segments, clip_space, edie, detector,
                                  voronoi_state, kernel_state, windows, triangle_cdf)


def uniform_stream(speed=10., spacing=20.):
    rows=[]
    for identifier,initial in enumerate(np.arange(-160,301,spacing)):
        for frame,t in enumerate(np.arange(0,10.001,.1)):
            rows.append(dict(vehicle_id=identifier,frame_id=frame,time_s=t,
                             lane_id=2,local_y_m=initial+3.7+speed*t,speed_m_s=speed))
    return pd.DataFrame(rows)


def test_uniform_stream_matches_analytic_density_flow_speed():
    segments=trajectory_segments(uniform_stream())
    e=edie(segments,0,100,[(0,10)]).iloc[0]
    d,_=detector(segments,0,100,[(0,10)])
    for result in [e,d.iloc[0]]:
        assert result.k_veh_km==pytest.approx(50,abs=1e-8)
        assert result.q_veh_h==pytest.approx(1800,abs=1e-8)
        assert result.v_km_h==pytest.approx(36,abs=1e-8)
    x=np.arange(-100,201,20.)+3.7;v=np.full(len(x),10.)
    k,q,coverage=voronoi_state(x,v,0,100)
    assert (k,q,coverage)==pytest.approx((.05,.5,1.))
    k,q=kernel_state(x,v,0,100,20.)
    assert (k,q)==pytest.approx((.05,.5))


def test_stopped_vehicles_count_as_density_not_flow():
    segment=pd.DataFrame([dict(vehicle_id=1,t0=0.,t1=10.,x0=50.,x1=50.,velocity=0.)])
    e=edie(segment,0,100,[(0,10)]).iloc[0]
    assert (e.k_veh_km,e.q_veh_h,e.v_km_h)==pytest.approx((10,0,0))
    d,events=detector(segment,0,100,[(0,10)])
    assert events.empty and d.iloc[0].q_veh_h==0 and np.isnan(d.iloc[0].k_veh_km)


def test_exact_entry_exit_and_temporal_clipping():
    segment=pd.DataFrame([dict(vehicle_id=1,t0=0.,t1=10.,x0=-50.,x1=150.,velocity=20.)])
    clipped=clip_space(segment,0,100).iloc[0]
    assert (clipped.t0,clipped.t1,clipped.x0,clipped.x1)==pytest.approx((2.5,7.5,0,100))
    row=edie(segment,0,100,[(1,5)]).iloc[0]
    assert row.vehicle_seconds==pytest.approx(2.5)
    assert row.vehicle_metres==pytest.approx(50.)


def test_lane_change_is_split_at_midpoint_and_gaps_not_joined():
    d=pd.DataFrame(dict(vehicle_id=[1,1,1],time_s=[0,.1,1.],lane_id=[1,2,2],
                        local_y_m=[10,11,20],speed_m_s=[10,10,10]))
    segments=trajectory_segments(d)
    assert len(segments)==1
    assert segments.iloc[0].t0==pytest.approx(.05)
    assert segments.iloc[0].t1==pytest.approx(.1)


def test_voronoi_uses_full_cells_and_requires_external_neighbours():
    x=np.array([-10.,10.,30.]);v=np.array([5.,5.,5.])
    k,q,coverage=voronoi_state(x,v,5,15)
    assert (k,q,coverage)==pytest.approx((.05,.25,1.))
    assert np.isnan(voronoi_state(x,v,-5,15)[0])


def test_kernel_unit_mass_and_empty_regions():
    assert triangle_cdf(np.array([-100,-1,0,1,100])).tolist()==pytest.approx([0,0,.5,1,1])
    k,q=kernel_state(np.array([50.]),np.array([5.]),0,100,10)
    assert (k,q)==pytest.approx((.01,.05))
    assert kernel_state(np.array([500.]),np.array([5.]),0,100,10)==(0.,0.)


def test_time_and_space_additivity_and_units():
    s=trajectory_segments(uniform_stream());full=edie(s,0,100,[(0,10)]).iloc[0]
    temporal=edie(s,0,100,windows(3,0,10))
    spatial=pd.concat([edie(s,a,a+25,[(0,10)]) for a in [0,25,50,75]])
    for metric in ["vehicle_seconds","vehicle_metres"]:
        assert temporal[metric].sum()==pytest.approx(full[metric],abs=1e-8)
        assert spatial[metric].sum()==pytest.approx(full[metric],abs=1e-8)
    np.testing.assert_allclose(temporal.q_veh_h,temporal.k_veh_km*temporal.v_km_h)


def test_detector_boundary_crossing_counted_once():
    s=pd.DataFrame([dict(vehicle_id=1,t0=0.,t1=1.,x0=0.,x1=10.,velocity=10.),
                    dict(vehicle_id=1,t0=1.,t1=2.,x0=10.,x1=20.,velocity=10.)])
    d,events=detector(s,0,20,[(0,1),(1,2)],point=10)
    assert len(events)==1
    assert d.samples.tolist()==[0,1]
