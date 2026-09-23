"""Sum-pooling mechanism and subsequent exploratory experiment contracts."""
import json
import numpy as np
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('trajnetplusplustools')
from week03.model.experiment import ROOT, SumNeighbourPooling, make_model, predict


def test_sum_shared_cell_and_gradients():
    pool = SumNeighbourPooling(type_='social', hidden_dim=2,latent_dim=1,out_dim=2,n=4,cell_side=1)
    xy=torch.tensor([[[0.,0.],[.2,.2],[.3,.3]]])
    values=torch.tensor([[[[2.],[3.]],[[4.],[5.]],[[6.],[7.]]]],requires_grad=True)
    grid=pool.occupancy(xy,values)
    assert grid[0,0,2,2].item()==5
    grid[0].sum().backward()
    torch.testing.assert_close(values.grad[0,0,:,0],torch.ones(2))


def test_sum_invalid_neighbour_does_not_erase_corner_and_inputs_unchanged():
    pool=SumNeighbourPooling(type_='social',hidden_dim=2,latent_dim=1,out_dim=2,n=4,cell_side=1)
    xy=torch.tensor([[[0.,0.],[-1.5,-1.5],[20.,20.],[float('nan'),float('nan')]]])
    before=xy.clone()
    grid=pool.occupancy(xy,torch.ones(1,4,3,1))
    assert grid[0,0,0,0].item()==1
    assert grid[0].sum().item()==1
    torch.testing.assert_close(xy,before,equal_nan=True)
    single=pool.occupancy(torch.zeros(2,1,2))
    assert single.shape==(2,1,4,4) and single.sum()==0


def test_sum_neighbour_permutation_and_matching_architecture():
    torch.set_num_threads(2)
    config=json.loads((ROOT/'sum_config.json').read_text(encoding='utf-8'))
    torch.manual_seed(42); model=make_model(config,'sum_pool')
    base_config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    torch.manual_seed(42); base=make_model(base_config,'social')
    for key,v in model.state_dict().items():
        torch.testing.assert_close(v,base.state_dict()[key],rtol=0,atol=0)
    xy=np.random.default_rng(9).normal(size=(8,4,2)).astype(np.float32)
    perm=[0,3,1,2]
    a=predict(model,xy);b=predict(model,xy[:,perm])
    np.testing.assert_allclose(a[:,perm],b,rtol=1e-5,atol=1e-6)


def test_extension_matches_original_initializations_and_orders():
    output=ROOT.parent/'results/sum_pool'
    if not (output/'metrics.csv').exists():
        pytest.skip('Extension training not finished')
    original=json.loads((ROOT.parent/'results/training.json').read_text(encoding='utf-8'))
    extra=json.loads((output/'training.json').read_text(encoding='utf-8'))
    assert len(extra['runs'])==3
    for run in extra['runs']:
        base=next(r for r in original['runs'] if r['variant']=='social' and r['seed']==run['seed'])
        for field in ['parameters','initial_state_sha256','epoch_order_sha256']:
            assert run[field]==base[field]
    import csv
    from week03.model.experiment import score
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config=json.loads((ROOT/'sum_config.json').read_text(encoding='utf-8'))
    metrics=list(csv.DictReader((output/'metrics.csv').open(encoding='utf-8')))
    with np.load(output/'predictions.npz') as arrays:
        for row in metrics:
            key=f'{row["split"]}_constant_velocity' if row['seed']=='-1' else f'{row["split"]}_sum_pool_seed{row["seed"]}'
            actual,_=score(arrays[key],arrays[row['split']+'_truth'])
            for metric in ['ADE','FDE']:
                assert actual[metric]==pytest.approx(float(row[metric]),abs=1e-7)
        model=make_model(config,'sum_pool')
        model.load_state_dict(torch.load(output/'checkpoints/sum_pool_seed42.pt',weights_only=True,map_location='cpu'))
        np.testing.assert_allclose(predict(model,arrays['test_history'][0]),arrays['test_sum_pool_seed42'][0],rtol=1e-5,atol=1e-4)
