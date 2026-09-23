"""Matched-architecture neighbour ablations using the teacher's actual LSTM."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from trajnetbaselines.lstm import GridBasedPooling, LSTM
import trajnetplusplustools


class NeighbourPooling(GridBasedPooling):
    """Keep the upstream baseline; ablate grid content before its embedding."""
    def __init__(self, mask_neighbours=False, **kwargs):
        super().__init__(**kwargs)
        self.mask_neighbours = mask_neighbours

    def forward(self, hidden_state, obs1, obs2):
        if not self.mask_neighbours:
            return super().forward(hidden_state, obs1, obs2)
        # Identical parameter shapes. Zero grid carries no neighbour geometry or
        # hidden-state content; the embedding bias remains trainable and constant.
        b, n = obs2.shape[:2]
        grid = hidden_state.new_zeros((b * n, self.n * self.n * self.pooling_dim))
        return self.embedding(grid)


def make_model(config, variant):
    pool = NeighbourPooling(type_='social', hidden_dim=config['hidden_dim'],
                            out_dim=config['pool_dim'], latent_dim=config['latent_dim'],
                            n=config['grid_n'], **config['variants'][variant])
    return LSTM(pool=pool, embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'], goal_flag=False)


def load_scenes(split):
    reader = trajnetplusplustools.Reader(str(ROOT / 'data' / split / 'circle.ndjson'), scene_type='paths')
    scenes = sorted(list(reader.scenes()), key=lambda item: item[0])
    return [(int(sid), paths, np.asarray(reader.paths_to_xy(paths), dtype=np.float32))
            for sid, paths in scenes]


def predict(model, history):
    """This interface cannot receive the target future or a future-based mask."""
    if history.shape[0] != 8 or history.ndim != 3 or history.shape[-1] != 2:
        raise ValueError('Expected exactly 8 history points for all agents')
    if not np.isfinite(history).all():
        raise ValueError('Missing/non-finite history')
    xy = torch.tensor(history.copy(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        _, positions = model(xy, torch.zeros(xy.shape[1], 2),
                             torch.tensor([0, xy.shape[1]]), n_predict=12)
    output = positions[-12:].detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError('Non-finite prediction')
    return output


def score(predictions, truth):
    errors = np.linalg.norm(np.asarray(predictions)[:, :, 0] - np.asarray(truth)[:, :, 0], axis=-1)
    return {'ADE': float(errors.mean()), 'FDE': float(errors[:, -1].mean())}, errors


def evaluate(model, scenes):
    predictions = np.stack([predict(model, xy[:8]) for _, _, xy in scenes])
    truth = np.stack([xy[8:] for _, _, xy in scenes])
    metrics, errors = score(predictions, truth)
    return metrics, predictions, errors


def constant_velocity(scenes):
    return np.stack([xy[7:8] + np.arange(1, 13, dtype=np.float32)[:, None, None] *
                     (xy[7:8] - xy[6:7]) for _, _, xy in scenes])
