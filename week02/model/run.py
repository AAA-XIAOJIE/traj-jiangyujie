"""Run with python -m week02.model.run; no future trajectory fed to simulator."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .analysis import evaluate, initial_conditions, read_experiment
from .simulation import simulate
from .plots import observed_figure, overview, failures, replay
from .interaction import make_encounter_figure

WEEK = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--figures-only", action="store_true", help="Replot the saved simulation without rerunning ORCA.")
    args = parser.parse_args()
    config = json.loads((WEEK / "model/config.json").read_text(encoding="utf-8"))
    experiment = read_experiment(WEEK / "model/data/circle-10m-64-1.txt", config["fps"], config["coordinate_scale_to_m"])
    output = WEEK / "results"
    output.mkdir(exist_ok=True)
    if args.figures_only:
        receipt_path = output / "experiment.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        numerical = lambda c: {k: v for k, v in c.items() if k != "source_assumptions"}
        if receipt["source_sha256"] != experiment["sha256"] or numerical(receipt["config"]) != numerical(config):
            raise ValueError("Input or numerical configuration changed; rerun the full simulation.")
        receipt["config"] = config
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    cases = {"observed": evaluate(experiment["positions"], experiment["times"], experiment)}
    observed_figure(experiment, cases["observed"], output / "observed_trajectories")
    make_encounter_figure(experiment, output)
    print("Observed trajectories and speed figure generated.", flush=True)
    initial, initial_v, desired = initial_conditions(experiment, config)
    metrics = [{"case": "observed", **cases["observed"]["stats"]}]
    saved = {}
    if not args.figures_only:
        main_model = {"bias": config["selected_right_bias_rad"], "side_seed": config["side_seed"],
                      "right_probability": config["right_preference_probability"]}
        specifications = [
            ("straight", {"avoid": False}),
            ("orca", {}),
            ("orca_right", main_model),
            ("uniform_right", {"bias": .15}),
            ("bias_050", {**main_model, "bias": .50}),
            ("bias_070", {**main_model, "bias": .70}),
            ("horizon_1", {**main_model, "horizon": 1.0}),
            ("horizon_3", {**main_model, "horizon": 3.0}),
            ("radius_020", {**main_model, "radius": .20}),
            ("radius_030", {**main_model, "radius": .30}),
            ("speed_seed_1", {**main_model, "speed_seed": 1}),
            ("speed_seed_2", {**main_model, "speed_seed": 2}),
            ("side_seed_43", {**main_model, "side_seed": 43}),
            ("side_seed_44", {**main_model, "side_seed": 44}),
            ("dt_002", {**main_model, "substeps": 2}),
        ]
        for name, settings in specifications:
            start = time.perf_counter()
            positions, diagnostics = simulate(initial, initial_v, experiment["goals"], desired, experiment["times"], config, **settings)
            result = evaluate(positions, experiment["times"], experiment,
                              body_radius=settings.get("radius", config["body_radius_m"]),
                              goal_tolerance=config["goal_tolerance_m"])
            row = {"case": name, **result["stats"], **diagnostics}
            row["speed_curve_rmse_mps"] = float(np.sqrt(np.mean((result["series"].mean_speed_mps - cases["observed"]["series"].mean_speed_mps)**2)))
            row["radius_curve_rmse_m"] = float(np.sqrt(np.mean((result["series"].mean_radius_m - cases["observed"]["series"].mean_radius_m)**2)))
            metrics.append(row)
            if name in ("straight", "orca", "orca_right", "radius_030"):
                saved[name] = positions
                cases[name] = result
            print(f"{name}: {time.perf_counter() - start:.1f} s; ADE={row['ade_after_calibration_m']:.2f} m, right={row['right_fraction']:.1%}, arrived={row['arrived_by_end']}, disk overlaps={row['disk_overlap_pair_intervals']}", flush=True)
            pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False, float_format="%.8f", lineterminator="\n")
        np.savez_compressed(output / "simulation.npz", times=experiment["times"], ids=experiment["ids"], **saved)
        side_sign = np.where(np.random.default_rng(config["side_seed"]).random(len(desired)) < config["right_preference_probability"], 1, -1)
        pd.DataFrame({"id": experiment["ids"], "x0_m": initial[:, 0], "y0_m": initial[:, 1],
                      "vx0_mps": initial_v[:, 0], "vy0_mps": initial_v[:, 1], "desired_speed_mps": desired,
                      "model_side_sign": side_sign,
                      "goal_x_m": experiment["goals"][:, 0], "goal_y_m": experiment["goals"][:, 1]}).to_csv(output / "initial_conditions.csv", index=False, float_format="%.8f", lineterminator="\n")
        evidence = {"source_sha256": experiment["sha256"], "rows": int(np.prod(experiment["positions"].shape[:2])),
                    "people": len(experiment["ids"]), "frames": len(experiment["times"]),
                    "first_frame": int(experiment["frames"][0]), "last_frame": int(experiment["frames"][-1]),
                    "fitted_center_m": experiment["center"].tolist(), "fitted_radius_m": experiment["radius"],
                    "config": config, "observed": cases["observed"]["stats"]}
        (output / "experiment.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    else:
        metrics = pd.read_csv(output / "metrics.csv").to_dict("records")
        with np.load(output / "simulation.npz") as archive:
            for name in ("straight", "orca", "orca_right", "radius_030"):
                cases[name] = evaluate(archive[name], experiment["times"], experiment,
                                       body_radius=.3 if name == "radius_030" else config["body_radius_m"])
    pd.concat([c["series"].assign(case=name) for name, c in cases.items()]).to_csv(output / "time_series.csv", index=False, float_format="%.6f", lineterminator="\n")
    pd.concat([c["individual"].assign(case=name) for name, c in cases.items()]).to_csv(output / "individual_metrics.csv", index=False, float_format="%.6f", lineterminator="\n")
    overview(experiment, cases, output / "model_comparison")
    failures(experiment, cases, metrics, output / "failure_cases")
    if not args.no_gif:
        replay(experiment, cases, output / "comparison.gif")
    print("Week 02 complete: figures, simulation, metrics and reproducible initial conditions.", flush=True)


if __name__ == "__main__":
    main()
