"""Causal input, matched ablation and fixed split contracts for Week 03."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('trajnetplusplustools')
from week03.model.experiment import ROOT, NeighbourPooling, make_model, load_scenes, predict, score
from trajnetbaselines.lstm import GridBasedPooling


@pytest.fixture
def config():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def test_equal_parameter_shapes_and_initial_values(config):
    states = []
    for variant in config['variants']:
        torch.manual_seed(42)
        states.append(make_model(config, variant).state_dict())
    for other in states[1:]:
        assert other.keys() == states[0].keys()
        for key in other:
            torch.testing.assert_close(other[key], states[0][key], rtol=0, atol=0)


def test_baseline_exact_teacher_pool_output():
    torch.manual_seed(8)
    original = GridBasedPooling(type_='social', hidden_dim=4, latent_dim=2, n=4, cell_side=1, out_dim=3)
    modified = NeighbourPooling(type_='social', hidden_dim=4, latent_dim=2, n=4, cell_side=1, out_dim=3)
    modified.load_state_dict(original.state_dict())
    h = torch.randn(1, 3, 4)
    xy = torch.tensor([[[0., 0.], [.8, .2], [-.9, 1.2]]])
    torch.testing.assert_close(original(h, xy.clone(), xy.clone()), modified(h, xy.clone(), xy.clone()), rtol=0, atol=0)


def test_masked_prediction_independent_of_neighbours(config):
    torch.manual_seed(5)
    model = make_model(config, 'no_neighbours')
    xy = np.stack([np.array([[i*.05, 0], [.6, i*.04], [-.6, .2]], dtype=np.float32) for i in range(8)])
    changed = xy.copy()
    changed[:, 1:] = np.random.default_rng(99).normal(size=(8, 2, 2)).astype(np.float32) * 10
    np.testing.assert_allclose(predict(model, xy)[:, 0], predict(model, changed)[:, 0], rtol=0, atol=1e-7)


def test_causal_api_rejects_future_window(config):
    model = make_model(config, 'social')
    with pytest.raises(ValueError, match='8 history'):
        predict(model, np.zeros((20, 3, 2), dtype=np.float32))


def test_pool_excludes_self_and_range():
    pool = NeighbourPooling(mask_neighbours=False, type_='social', hidden_dim=2, latent_dim=1, out_dim=2, n=4, cell_side=1)
    xy = torch.tensor([[[0., 0.], [10., 10.]]])
    h = torch.tensor([[[10., 10.], [20., 20.]]])
    grid = pool.social(h, xy.clone(), xy.clone())
    assert torch.count_nonzero(grid) == 0
    xy[:, 1] = torch.tensor([.5, .5])
    grid = pool.social(h, xy.clone(), xy.clone())
    assert torch.count_nonzero(grid) > 0


def test_teacher_files_and_split_hashes():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT / 'data/teacher_audit.json').read_text(encoding='utf-8'))
    raw = ROOT.parents[1] / 'week02/model/data/circle-10m-64-1.txt'
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == audit['source_sha256']
    for name, expected in manifest['unchanged_teacher_source_sha256'].items():
        assert hashlib.sha256((ROOT/'vendor/trajnetbaselines'/name).read_bytes()).hexdigest() == expected
    frame_sets = []
    for split, expected in manifest['data_sha256'].items():
        path = ROOT/'data'/split/'circle.ndjson'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        frame_sets.append({r['track']['f'] for r in rows if 'track' in r})
    for i in range(3):
        for j in range(i):
            assert frame_sets[i].isdisjoint(frame_sets[j])


def test_full_scenes_fixed_primary_and_windows():
    for split, count in [('train',56), ('val',12), ('test',12)]:
        scenes = load_scenes(split)
        assert len(scenes) == count
        assert {paths[0][0].pedestrian for _, paths, _ in scenes} == {1,17,33,49}
        for _, _, xy in scenes:
            assert xy.shape == (20,64,2) and np.isfinite(xy).all()


def test_metrics_primary_only_in_physical_distance():
    truth = np.zeros((2,12,3,2))
    pred = truth.copy()
    pred[:, :, 0] = [3,4]
    pred[:, :, 1:] = 1000
    metrics, _ = score(pred, truth)
    assert metrics == {'ADE':5., 'FDE':5.}


def test_saved_evidence_matches_predictions_and_validation_selection(config):
    import csv
    out = ROOT.parent / 'results/runs/baseline'
    if not (out / 'metrics.csv').exists():
        pytest.skip('Full training has not finished yet')
    rows = list(csv.DictReader((out/'metrics.csv').open(encoding='utf-8')))
    curves = list(csv.DictReader((out/'learning_curves.csv').open(encoding='utf-8')))
    receipt = json.loads((out/'training.json').read_text(encoding='utf-8'))
    assert len(receipt['runs']) == 12
    assert len(rows) == 26 and len(curves) == 120
    with np.load(out/'predictions.npz') as saved:
        for row in rows:
            split, variant, seed = row['split'], row['variant'], int(row['seed'])
            key = f'{split}_constant_velocity' if seed == -1 else f'{split}_{variant}_seed{seed}'
            assert saved[key].shape == (12,12,64,2)
            metrics, _ = score(saved[key], saved[split+'_truth'])
            for metric in ['ADE','FDE']:
                assert metrics[metric] == pytest.approx(float(row[metric]), abs=1e-7)
        for variant in config['variants']:
            model = make_model(config, variant)
            model.load_state_dict(torch.load(out/'checkpoints'/f'{variant}_seed42.pt', weights_only=True, map_location='cpu'))
            pred = predict(model, saved['test_history'][0])
            np.testing.assert_allclose(pred, saved[f'test_{variant}_seed42'][0], rtol=1e-5, atol=1e-4)
    for run in receipt['runs']:
        trace = [r for r in curves if r['variant']==run['variant'] and int(r['seed'])==run['seed']]
        chosen = min(trace, key=lambda r:float(r['ADE']))
        assert int(chosen['epoch']) == run['selected_epoch']
        checkpoint = out/'checkpoints'/f'{run["variant"]}_seed{run["seed"]}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == run['checkpoint_sha256']
    for seed in config['seeds']:
        group = [r for r in receipt['runs'] if r['seed']==seed]
        assert len({r['initial_state_sha256'] for r in group}) == 1
        assert len({r['parameters'] for r in group}) == 1
        assert all(r['epoch_order_sha256']==group[0]['epoch_order_sha256'] for r in group)
