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


class SumNeighbourPooling(NeighbourPooling):
    """Add all valid neighbours in a cell instead of indexed overwrite.

    The experiment retains the teacher's 4x4, zero-background, unblurred grid.
    Out-of-range neighbours contribute zero, so they cannot erase cell (0,0).
    """
    def occupancy(self, obs, other_values=None, past_obs=None):
        if self.front or self.pool_size != 1 or self.blur_size != 1 or self.constant != 0:
            raise ValueError('Sum ablation supports the fixed classroom grid preset')
        b, n, _ = obs.shape
        if n == 1:
            return obs.new_zeros((b, self.pooling_dim, self.n, self.n))
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=obs.device)
        relative = (obs[:, None, :, :] - obs[:, :, None, :])[:, off_diagonal].reshape(b,n,n-1,2)
        index_float = relative / self.cell_side + self.n / 2
        valid = torch.isfinite(index_float).all(-1) & (index_float >= 0).all(-1) & (index_float < self.n).all(-1)
        cells = torch.where(valid[...,None], index_float, torch.zeros_like(index_float)).long()
        flat = (cells[...,0]*self.n + cells[...,1]).reshape(b*n,n-1)
        if other_values is None:
            other_values = obs.new_ones((b,n,n-1,self.pooling_dim))
        values = torch.where(valid[...,None], torch.nan_to_num(other_values), torch.zeros_like(other_values))
        values = values.reshape(b*n,n-1,self.pooling_dim)
        grid = obs.new_zeros((b*n,self.n*self.n,self.pooling_dim))
        grid = grid.scatter_add(1,flat[...,None].expand_as(values),values)
        return grid.transpose(1,2).reshape(b*n,self.pooling_dim,self.n,self.n)


class DirectionalSumPooling(SumNeighbourPooling):
    """Heading-aware weights, computed from current/previous positions only.

    w = .25 + .75 max(0, cos(angle)); stationary agents (<0.1 m/s at
    the fixed 0.08 s sampling interval) use equal weights instead.
    """
    def occupancy(self, obs, other_values=None, past_obs=None):
        b,n,_=obs.shape
        if n<=1 or past_obs is None:
            return super().occupancy(obs,other_values,past_obs)
        mask=~torch.eye(n,dtype=torch.bool,device=obs.device)
        relative=(obs[:,None,:,:]-obs[:,:,None,:])[:,mask].reshape(b,n,n-1,2)
        displacement=obs-past_obs
        norm=torch.linalg.vector_norm(displacement,dim=-1)
        distance=torch.linalg.vector_norm(relative,dim=-1)
        cosine=(relative*displacement[:,:,None,:]).sum(-1)/(distance*norm[:,:,None]).clamp_min(1e-8)
        weights=.25+.75*torch.nan_to_num(cosine).clamp(0,1)
        weights=torch.where((norm<.008)[:,:,None],torch.ones_like(weights),weights)
        if other_values is None:
            other_values=obs.new_ones((b,n,n-1,self.pooling_dim))
        return super().occupancy(obs,other_values*weights[...,None],past_obs)


def make_model(config, variant):
    pool_class = {'sum_pool':SumNeighbourPooling,'directional_sum':DirectionalSumPooling}.get(variant,NeighbourPooling)
    pool = pool_class(type_='social', hidden_dim=config['hidden_dim'],
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
