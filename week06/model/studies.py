"""Predeclared ablation and numerical sensitivity; all outcomes are retained."""
import copy
import numpy as np
import pandas as pd
from week02.model.analysis import evaluate
from .scenarios import narrow,circle
from .simulation import simulate


def additional_studies(cfg,output,main_metrics):
    rows=[]
    for model in cfg["models"]:
        for variant in ["guided_no_veto","veto_only"]:
            result=simulate(narrow(cfg,3.2),cfg,model,variant)
            rows.append(dict(study="component_ablation",**result["metrics"]))
    for width in [3.2,2.6]:
        for model in ["sfm","idm","orca"]:
            result=simulate(narrow(cfg,width),cfg,model,"guided",dt=.02)
            rows.append(dict(study="half_step",**result["metrics"]))
    for model,variant in [("sfm","guided"),("idm","guided"),("orca","guided")]:
        scene,obs,_=circle(cfg)
        result=simulate(scene,cfg,model,variant,dt=.02)
        # Compare every second fine-grid sample with the same empirical 25 Hz frames.
        ev=evaluate(result["positions"][::2],result["times"][::2],obs)
        rows.append(dict(study="half_step",**result["metrics"],ade_after_calibration_m=ev["stats"]["ade_after_calibration_m"]))
    for seed in [43,44]:
        setting=copy.deepcopy(cfg);setting["seed"]=seed;scene,obs,_=circle(setting)
        result=simulate(scene,setting,"orca","guided")
        ev=evaluate(result["positions"],result["times"],obs)
        rows.append(dict(study="side_seed",seed=seed,**result["metrics"],ade_after_calibration_m=ev["stats"]["ade_after_calibration_m"]))
    pd.DataFrame(rows).to_csv(output/"additional_studies.csv",index=False,float_format="%.9g",lineterminator="\n")
    return pd.DataFrame(rows)
