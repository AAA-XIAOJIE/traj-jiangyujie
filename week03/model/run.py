"""Train fixed variants, freeze validation choices, evaluate the fixed test set."""
import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from .experiment import ROOT, make_model, load_scenes, evaluate, constant_velocity, score
from trajnetbaselines.lstm.trainer import Trainer
from trajnetbaselines.lstm.loss import PredictionLoss


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_hash(model):
    digest = hashlib.sha256()
    for key, val in model.state_dict().items():
        digest.update(key.encode())
        digest.update(val.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'results')
    parser.add_argument('--smoke', action='store_true', help='One epoch/seed in separate output; not reportable')
    parser.add_argument('--evaluate-only', action='store_true', help='Verify saved checkpoints on fixed splits')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.smoke:
        if args.output.resolve() == (ROOT.parent / 'results').resolve():
            raise ValueError('Smoke output must be a separate directory')
        config['epochs'], config['seeds'] = 1, [42]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    train_scenes = load_scenes('train')
    val_scenes = load_scenes('val')
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    for split, expected in manifest['data_sha256'].items():
        assert hashlib.sha256((ROOT / 'data' / split / 'circle.ndjson').read_bytes()).hexdigest() == expected
    for _, _, xy in train_scenes + val_scenes:
        assert xy.shape == (20, 64, 2) and np.isfinite(xy).all()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    history_rows, runs = [], []
    begin = time.perf_counter()
    if args.evaluate_only:
        runs = json.loads((out / 'training.json').read_text(encoding='utf-8'))['runs']
        assert all(r['config_sha256'] == config_hash for r in runs)
    else:
        if (out / 'training.json').exists():
            raise FileExistsError('Existing experiment; choose --output or --evaluate-only')
        for seed in config['seeds']:
            hashes = []
            orders = []
            for variant in config['variants']:
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                model = make_model(config, variant)
                initial_hash = state_hash(model); hashes.append(initial_hash)
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_step'])
                trainer = Trainer(model, criterion=PredictionLoss(), optimizer=optimizer,
                                  lr_scheduler=scheduler, batch_size=config['batch_size'],
                                  obs_length=8, pred_length=12, augment=False, normalize_scene=False)
                train = [('circle', sid, paths) for sid, paths, _ in train_scenes]
                best, best_state, best_epoch = float('inf'), None, None
                epoch_orders = []
                started = time.perf_counter()
                for epoch in range(config['epochs']):
                    with contextlib.redirect_stdout(io.StringIO()):
                        trainer.train(train, None, epoch)
                    metrics, _, _ = evaluate(model, val_scenes)
                    epoch_orders.append(hashlib.sha256(str([x[1] for x in train]).encode()).hexdigest())
                    history_rows.append({'seed': seed, 'variant': variant, 'epoch': epoch+1, **metrics})
                    if metrics['ADE'] < best:
                        best, best_epoch = metrics['ADE'], epoch+1
                        best_state = copy.deepcopy(model.state_dict())
                    print(f'{variant} seed={seed} epoch={epoch+1}/{config["epochs"]} val ADE={metrics["ADE"]:.4f} FDE={metrics["FDE"]:.4f}', flush=True)
                orders.append(epoch_orders)
                model.load_state_dict(best_state)
                checkpoint = checkpoints / f'{variant}_seed{seed}.pt'
                torch.save(best_state, checkpoint)
                run = {'variant': variant, 'seed': seed, 'selected_epoch': best_epoch,
                       'best_val_ADE': best, 'elapsed_seconds': time.perf_counter()-started,
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'initial_state_sha256': initial_hash, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       'epoch_order_sha256': epoch_orders, 'config_sha256': config_hash}
                runs.append(run)
                write_csv(out / 'learning_curves.csv', history_rows)
                write_json(out / 'training.json', {'config':config, 'runs':runs,
                           'environment':{'python':platform.python_version(), 'torch':torch.__version__,
                                          'numpy':np.__version__, 'device':'cpu', 'threads':config['threads']}})
                print(f'COMPLETED {variant} seed={seed}: {run["elapsed_seconds"]:.1f}s; selected epoch {best_epoch}', flush=True)
            assert len(set(hashes)) == 1, 'Unmatched initial parameters'
            assert all(o == orders[0] for o in orders), 'Unmatched scene order'
    # No test arrays or test errors were loaded/used by the training loop.
    val_means = {variant:float(np.mean([r['best_val_ADE'] for r in runs if r['variant']==variant])) for variant in config['variants']}
    selected_variant = min(val_means, key=val_means.get)
    write_json(out / 'validation_selection.json', {'selection':'lowest mean validation ADE across fixed seeds',
               'mean_ADE':val_means,'selected_variant':selected_variant,'config_sha256':config_hash,
               'test_used_for_selection':False})
    aggregate, per_scene, saved = [], [], {}
    for split in ['val','test']:
        scenes = load_scenes(split)
        truth = np.stack([xy[8:] for _, _, xy in scenes])
        saved[split+'_history'] = np.stack([xy[:8] for _, _, xy in scenes])
        saved[split+'_truth'] = truth
        saved[split+'_scene_ids'] = np.array([sid for sid, _, _ in scenes])
        cv = constant_velocity(scenes)
        saved[split+'_constant_velocity'] = cv
        metrics, errors = score(cv, truth)
        aggregate.append({'split':split,'variant':'constant_velocity','seed':-1,'selected_epoch':0,'scenes':len(scenes),**metrics})
        for (sid, paths, _), e in zip(scenes, errors):
            per_scene.append({'split':split,'variant':'constant_velocity','seed':-1,'scene_id':sid,
                              'primary_id':paths[0][0].pedestrian,'ADE':float(e.mean()),'FDE':float(e[-1])})
        for run in runs:
            variant, seed = run['variant'], run['seed']
            model = make_model(config, variant)
            ckpt = checkpoints/f'{variant}_seed{seed}.pt'
            assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == run['checkpoint_sha256']
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
            metrics, pred, errors = evaluate(model, scenes)
            saved[f'{split}_{variant}_seed{seed}'] = pred
            aggregate.append({'split':split,'variant':variant,'seed':seed,'selected_epoch':run['selected_epoch'],'scenes':len(scenes),**metrics})
            for (sid, paths, _), e in zip(scenes, errors):
                per_scene.append({'split':split,'variant':variant,'seed':seed,'scene_id':sid,
                                  'primary_id':paths[0][0].pedestrian,'ADE':float(e.mean()),'FDE':float(e[-1])})
    np.savez_compressed(out/'predictions.npz', **saved)
    write_csv(out/'metrics.csv', aggregate)
    write_csv(out/'per_scene.csv', per_scene)
    write_json(out/'evaluation.json', {'config_sha256':config_hash,'protocol':config['evaluation'],
               'wall_seconds_this_invocation':time.perf_counter()-begin,
               'test_scene_ids':saved['test_scene_ids'].tolist(),
               'selection_frozen_before_test':True,'seed_count':len(config['seeds'])})
    print('Training and fixed evaluation complete. Selected by validation:', selected_variant, flush=True)


if __name__ == '__main__':
    main()
