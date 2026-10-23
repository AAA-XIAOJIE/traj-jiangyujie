"""Run unified pedestrian/vehicle experiments from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from week02.model.analysis import evaluate
from .scenarios import ROOT,narrow,circle,mixed_circle
from .simulation import simulate


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8",newline="\n")


def run(output,only=None):
    output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/"config.json").read_text(encoding="utf8"));rows=[];saved={};series=[];begin=time.perf_counter()
    specifications=[(narrow(cfg,w),m,v) for w in cfg["narrow"]["widths_m"] for m in cfg["models"] for v in cfg["variants"]]
    ring,obs,previous=circle(cfg)
    specifications += [(ring,m,v) for m in cfg["models"] for v in cfg["variants"]]
    mixed=mixed_circle(cfg,obs["center"])
    specifications += [(mixed,m,v) for m in cfg["models"] for v in [*cfg["variants"],"steering"]]
    observed=evaluate(obs["positions"],obs["times"],obs)
    initial_rows=[]
    for s in [narrow(cfg,cfg["narrow"]["widths_m"][0]),ring,mixed]:
        for i in range(len(s["p0"])):
            initial_rows.append(dict(scene=s["name"],agent_id=int(obs["ids"][i]) if s["name"]=="circle" else i,
                baseline_lateral_allowed=bool(s["lateral"][i]),x_m=s["p0"][i,0],y_m=s["p0"][i,1],
                vx_m_s=s["v0"][i,0],vy_m_s=s["v0"][i,1],goal_x_m=s["goals"][i,0],goal_y_m=s["goals"][i,1],
                desired_speed_m_s=s["desired"][i],radius_m=s["radius"][i],
                rectangle_length_m=cfg["narrow"]["car_length_m"] if not s["lateral"][i] else 0,
                rectangle_width_m=cfg["narrow"]["car_width_m"] if not s["lateral"][i] else 0,
                collision_geometry="car_circumdisk" if s["name"]=="circle_mixed" and not s["lateral"][i] else ("rectangle" if not s["lateral"][i] else "disk"),
                tau_s=s["tau"][i],acceleration_bound_m_s2=s["accel"][i],braking_bound_m_s2=s["brake"][i],
                side_preference_sign=s["side"][i]))
    pd.DataFrame(initial_rows).to_csv(output/"initial_conditions.csv",index=False,float_format="%.9g",lineterminator="\n")
    for scene,model,variant in specifications:
        key=f"{scene['name']}_{scene['width']}_{model}_{variant}"
        if only and only not in key:continue
        start=time.perf_counter();result=simulate(scene,cfg,model,variant);row=result["metrics"]
        saved[key+"__positions"]=result["positions"];saved[key+"__velocities"]=result["velocities"];saved[key+"__times"]=result["times"]
        saved[key+"__arrival_times"]=result["arrival_times"]
        if scene["name"]=="circle":
            ev=evaluate(result["positions"],result["times"],obs)
            row.update({k:ev["stats"][k] for k in ["ade_after_calibration_m","fde_m","median_max_lateral_m","right_fraction","closest_approach_speed_mps","slower_at_closest_fraction"]})
            row["mean_speed_curve_rmse_m_s"]=float(np.sqrt(np.mean((ev["series"].mean_speed_mps-observed["series"].mean_speed_mps)**2)))
            row["mean_radius_curve_rmse_m"]=float(np.sqrt(np.mean((ev["series"].mean_radius_m-observed["series"].mean_radius_m)**2)))
            series.append(ev["series"].assign(case=key))
        rows.append(row)
        pd.DataFrame(rows).to_csv(output/"metrics.csv",index=False,float_format="%.9g",lineterminator="\n")
        print(key,f"{time.perf_counter()-start:.1f}s",json.dumps({k:row[k] for k in ["arrived","minimum_body_gap_m","body_overlap_pair_steps","wall_violation_agent_steps","geometry_veto_agent_steps"]}),flush=True)
    np.savez_compressed(output/"trajectories.npz",**saved)
    series.append(observed["series"].assign(case="observed"))
    pd.concat(series,ignore_index=True).to_csv(output/"circle_series.csv",index=False,float_format="%.9g",lineterminator="\n")
    write_json(output/"experiment.json",dict(config=cfg,observed=observed["stats"],source_sha256=obs["sha256"],
        source_file="week02/model/data/circle-10m-64-1.txt",circle_initialization=previous,
        dependency_sha256={name:hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest()
            for name in ["week02/model/analysis.py","week02/model/orca.py","week02/model/config.json","week01/style.py"]},
        geometric_minimum_road_width_m=cfg["narrow"]["car_width_m"]+4*cfg["narrow"]["pedestrian_radius_m"],
        numerical_semantics="Synchronous x_next=x+dt*v_next. Swept geometry assumes this piecewise-linear segment. Overlap simulations continue diagnostically; arrived counts after overlap are not safe throughput.",
        source_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"model").glob("*.py"))}))
    print(f"Completed {len(rows)} runs in {time.perf_counter()-begin:.1f}s",flush=True)
    return pd.DataFrame(rows),saved,obs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--only",help="Development subset; not a complete release")
    parser.add_argument("--figures-only",action="store_true")
    parser.add_argument("--no-gif",action="store_true")
    parser.add_argument("--no-studies",action="store_true",help="Skip additional ablation/sensitivity simulations")
    args=parser.parse_args();begin=time.perf_counter()
    cfg=json.loads((ROOT/"config.json").read_text(encoding="utf8"))
    if args.figures_only:
        metrics=pd.read_csv(args.output/"metrics.csv");saved=np.load(args.output/"trajectories.npz")
        _,obs,_=circle(cfg)
        receipt=json.loads((args.output/"experiment.json").read_text(encoding="utf8"))
        if receipt["config"]!=cfg or receipt["source_sha256"]!=obs["sha256"]:
            raise ValueError("Inputs changed; rerun calculations before replotting")
        for name,digest in receipt["source_code_sha256"].items():
            if hashlib.sha256((ROOT/"model"/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f"Source changed ({name}); rerun calculations before replotting")
    else:
        metrics,saved,obs=run(args.output,args.only)
    if args.only:return
    from .figures import plot_all
    from .studies import additional_studies
    from .report import one_page
    study_path=args.output/"additional_studies.csv"
    studies=pd.read_csv(study_path) if args.figures_only and study_path.exists() else None
    if not args.figures_only and not args.no_studies:
        studies=additional_studies(cfg,args.output,metrics)
    plot_all(cfg,metrics,saved,obs,args.output,studies,gifs=not args.no_gif)
    pdf=one_page(args.output,metrics)
    from pypdf import PdfReader
    assert len(PdfReader(pdf).pages)==1
    write_json(args.output/"reproducibility.json",dict(
        environment={"python":platform.python_version(),**{name:version(name) for name in ["numpy","pandas","matplotlib","Pillow","reportlab","pypdf"]}},
        elapsed_seconds=round(time.perf_counter()-begin,2),
        config_sha256=hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest(),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file() and p.name not in ["reproducibility.json","one_page.png"]}))
    print(f"All outputs complete in {time.perf_counter()-begin:.1f}s: {args.output}",flush=True)


if __name__=="__main__":main()
