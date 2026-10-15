"""Three matched leader experiments and held-out NGSIM validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import DEFAULTS, bando_velocity, optimal_velocity, acceleration, idm_desired_gap
from .calibration import fit
from .simulation import simulate, from_observations

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    def sanitize(x):
        if isinstance(x, dict): return {k:sanitize(v) for k,v in x.items()}
        if isinstance(x, list): return [sanitize(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, (np.bool_,)): return bool(x)
        return x
    Path(path).write_text(json.dumps(sanitize(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")


def table(frame, path):
    frame.to_csv(path,index=False,float_format="%.12g",lineterminator="\n")


def metrics(frame, meta, data, scenario, config):
    complete = bool(meta["completed"])
    result = dict(meta)
    result.update(position_rmse_m=np.nan, speed_rmse_m_s=np.nan, gap_rmse_m=np.nan)
    if scenario == "observed" and complete:
        result.update(position_rmse_m=float(np.sqrt(np.mean((frame.x_m-data.follower_x_m.to_numpy())**2))),
                      speed_rmse_m_s=float(np.sqrt(np.mean((frame.v_m_s-data.follower_v_raw_m_s.to_numpy())**2))),
                      gap_rmse_m=float(np.sqrt(np.mean((frame.gap_m-data.net_gap_m.to_numpy())**2))))
    valid = frame.dropna(subset=["v_m_s"])
    moving = valid.v_m_s > .1
    closing = valid.v_m_s-valid.leader_v_m_s
    approaching = closing > .1
    result.update(valid_output_samples=len(valid),
                  min_time_gap_s=float((valid.loc[moving,"gap_m"]/valid.loc[moving,"v_m_s"]).min()) if moving.any() else np.nan,
                  min_ttc_s=float((valid.loc[approaching,"gap_m"]/closing[approaching]).min()) if approaching.any() else np.nan,
                  response_delay_s=np.nan, follower_stop_time_s=np.nan,
                  equilibrium_speed_error_last5_m_s=np.nan)
    if scenario == "braking":
        # Predeclared numerical threshold: first a<=-0.2 after the brake begins.
        start = config["braking_start_s"]
        braking = valid[(valid.time_s >= start) & (valid.a_m_s2 <= -.2)]
        result["response_delay_s"] = float(braking.time_s.iloc[0]-start) if len(braking) else np.nan
        stopped = valid[(valid.time_s >= start) & (valid.v_m_s <= .1)]
        result["follower_stop_time_s"] = float(stopped.time_s.iloc[0]) if len(stopped) else np.nan
    if scenario == "constant" and complete:
        tail = valid[valid.time_s >= config["validation_duration_s"]-5.]
        result["equilibrium_speed_error_last5_m_s"] = float(np.mean(abs(tail.v_m_s-tail.leader_v_m_s)))
    return result


def compute(out):
    config = json.loads((ROOT/"config.json").read_text(encoding="utf8"))
    manifest = json.loads((ROOT/"data/manifest.json").read_text(encoding="utf8"))
    source = ROOT/"data"/manifest["subset_file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["subset_sha256"]:
        raise ValueError("NGSIM input checksum changed")
    data = pd.read_csv(source)
    if data.isna().any().any() or data.duplicated(["time_s"]).any():
        raise ValueError("Missing or duplicate observations")
    np.testing.assert_allclose(data.net_gap_m,
                               data.leader_x_m-data.follower_x_m-data.leader_length_m,atol=1e-8)
    train = data[data.relative_time_s < config["calibration_end_s"]].copy()
    test = data[data.relative_time_s >= config["calibration_end_s"]].copy()
    times = test.relative_time_s.to_numpy()-test.relative_time_s.iloc[0]
    calibration = {}; parameters = {"baseline":{m:dict(DEFAULTS[m]) for m in config["models"]}, "calibrated":{}}
    for model in config["models"]:
        p, detail = fit(model, train, config)
        parameters["calibrated"][model] = p; calibration[model] = detail
        print(f"{model} fit: loss {detail['baseline_objective']:.4f} -> {detail['objective']:.4f}; {p}",flush=True)
    write_json(out/"calibration.json",dict(parameters=parameters,fit=calibration,
        training_samples=len(train),validation_samples=len(test),
        objective="mean(((x_sim-x_observed)/10 m)^2+((v_sim-v_raw)/5 m/s)^2)",
        invalid_fit_policy="collisions, nonfinite and raw negative-speed proposals receive loss 1e6",
        optimization="deterministic 257 candidate initialization, eight multistarts, bounded coordinate pattern search; not a proof of a global optimum"))
    frames = []; records = []; traces = {}; fine_results = {}
    for set_name, models in parameters.items():
        for model, p in models.items():
            for scenario in ["constant","braking","observed"]:
                leader = from_observations(test,scenario,config["braking_deceleration_m_s2"])
                leader.brake_start = config["braking_start_s"]
                frame,meta = simulate(model,p,times,leader,float(test.follower_x_m.iloc[0]),
                                      float(test.follower_v_raw_m_s.iloc[0]),config["main_step_s"])
                record = dict(model=model,parameter_set=set_name,scenario=scenario,
                              **metrics(frame,meta,test,scenario,config))
                records.append(record)
                frame = frame.assign(model=model,parameter_set=set_name,scenario=scenario)
                frames.append(frame);traces[(set_name,model,scenario)]=frame
    summary = pd.DataFrame(records)
    table(summary,out/"metrics.csv");table(pd.concat(frames,ignore_index=True),out/"trajectories.csv")
    # Same calibrated parameters and observations at three steps: no refitting.
    convergence = []
    for model,p in parameters["calibrated"].items():
        for scenario in ["constant","braking","observed"]:
            leader = from_observations(test,scenario,config["braking_deceleration_m_s2"])
            leader.brake_start=config["braking_start_s"]
            runs={}
            for step in config["convergence_steps_s"]:
                runs[step]=simulate(model,p,times,leader,float(test.follower_x_m.iloc[0]),
                                    float(test.follower_v_raw_m_s.iloc[0]),step)
            fine, fine_meta = runs[min(runs)]
            for step,(frame,meta) in runs.items():
                common=np.isfinite(frame.x_m)&np.isfinite(fine.x_m)
                convergence.append(dict(model=model,scenario=scenario,step_s=step,
                    max_position_difference_vs_fine_m=float(abs(frame.loc[common,"x_m"]-fine.loc[common,"x_m"]).max()),
                    max_speed_difference_vs_fine_m_s=float(abs(frame.loc[common,"v_m_s"]-fine.loc[common,"v_m_s"]).max()),
                    minimum_gap_m=meta["minimum_gap_m"],status=meta["status"],
                    collision_time_s=meta["collision_time_s"],
                    raw_negative_speed_requests=meta["raw_negative_speed_requests"]))
    table(pd.DataFrame(convergence),out/"convergence.csv")
    # Stronger braking is a scenario-sensitivity study, not additional observed truth.
    stress=[]
    for braking in config["braking_sensitivity_m_s2"]:
        for model,p in parameters["calibrated"].items():
            leader=from_observations(test,"braking",braking);leader.brake_start=config["braking_start_s"]
            frame,meta=simulate(model,p,times,leader,float(test.follower_x_m.iloc[0]),
                                float(test.follower_v_raw_m_s.iloc[0]),config["main_step_s"])
            stress.append(dict(model=model,leader_deceleration_m_s2=braking,**metrics(frame,meta,test,"braking",config)))
    table(pd.DataFrame(stress),out/"braking_sensitivity.csv")
    # A controlled ablation isolates lambda from independent parameter calibration.
    ablation=[]
    for gain in [0.,.5]:
        p=dict(DEFAULTS["FVDM"]);p["lambda"]=gain
        for scenario in ["constant","braking","observed"]:
            leader=from_observations(test,scenario,config["braking_deceleration_m_s2"])
            leader.brake_start=config["braking_start_s"]
            frame,meta=simulate("FVDM",p,times,leader,float(test.follower_x_m.iloc[0]),
                                float(test.follower_v_raw_m_s.iloc[0]),config["main_step_s"])
            ablation.append(dict(velocity_difference_gain_s_inv=gain,scenario=scenario,**metrics(frame,meta,test,scenario,config)))
    table(pd.DataFrame(ablation),out/"feedback_ablation.csv")
    # Bando's exact scaled equation and SI dimensional reference checks.
    formula_checks=dict(bando_V_zero=float(bando_velocity(0)),bando_V_two=float(bando_velocity(2)),
        Jiang_SI_V_at_gap_10_m=float(optimal_velocity(10.,DEFAULTS["OVM"])),
        FVDM_minus_OVM_at_gap10_v8_vL6=float(acceleration("FVDM",10.,8.,6.,DEFAULTS["FVDM"])-acceleration("OVM",10.,8.,6.,DEFAULTS["OVM"])),
        IDM_equilibrium_rest_acceleration=float(acceleration("IDM",2.,0.,0.,DEFAULTS["IDM"])),
        IDM_original_negative_desired_gap_example_m=float(idm_desired_gap(10.,40.,dict(DEFAULTS["IDM"],a=1.,b=1.,T=1.))))
    write_json(out/"formula_checks.json",formula_checks)
    write_json(out/"experiment.json",dict(source_manifest=manifest,
        calibration_time_s=[float(train.time_s.min()),float(train.time_s.max())],
        validation_time_s=[float(test.time_s.min()),float(test.time_s.max())],
        shared_initial_position_m=float(test.follower_x_m.iloc[0]),shared_initial_speed_m_s=float(test.follower_v_raw_m_s.iloc[0]),
        initial_leader_position_m=float(test.leader_x_m.iloc[0]),initial_leader_raw_speed_m_s=float(test.leader_v_raw_m_s.iloc[0]),
        initial_net_gap_m=float(test.net_gap_m.iloc[0]),leader_length_m=float(test.leader_length_m.iloc[0]),
        observed_minimum_gap_m=float(test.net_gap_m.min()),config=config,
        speed_semantics="Observed leader: piecewise-linear position derivative; initial follower and speed RMSE: original supplied speed. No follower state reset during rollout.",
        counterfactual_truth="No observed follower truth exists for constant/braking interventions; errors are NaN.",
        minimum_gap_semantics="minimum accepted internal-step gap, including estimated first contact 0; rejected overlap proposals stored separately",
        TTC_semantics="s/(v-vL) only if closing>0.1 m/s; time gap s/v only if v>0.1 m/s",
        main_step_quadrature="ballistic explicit acceleration updates; dt=0.05s; exact zero-speed event, no acceleration cap/gap repair",
        fit_parameter_counts={model:len(config["parameter_bounds"][model]) for model in config["models"]}))
    print(summary[["model","parameter_set","scenario","position_rmse_m","speed_rmse_m_s","minimum_gap_m","status"]].to_string(index=False),flush=True)
    return data,test,summary,traces,parameters,config
