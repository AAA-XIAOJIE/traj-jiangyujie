import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('trajnetplusplustools')
from week03.model.experiment import ROOT,DirectionalSumPooling


def test_heading_front_back_and_stationary_fallback():
    pool=DirectionalSumPooling(type_='social',hidden_dim=2,latent_dim=1,out_dim=2,n=4,cell_side=1)
    obs=torch.tensor([[[0.,0.],[.8,0.],[-.8,0.]]])
    past=obs.clone();past[0,0,0]=-.08
    values=torch.ones(1,3,2,1)
    grid=pool.occupancy(obs,values,past)
    assert grid[0,0,2,2].item()==pytest.approx(1.)
    assert grid[0,0,1,2].item()==pytest.approx(.25)
    stationary=pool.occupancy(obs,values,obs.clone())
    assert stationary[0].sum().item()==pytest.approx(2.)
    # Rotate both geometry and history by 90 degrees: total weights unchanged.
    rot=torch.tensor([[0.,-1.],[1.,0.]])
    rotated=pool.occupancy(obs@rot,values,past@rot)
    torch.testing.assert_close(grid.sum(dim=(1,2,3)),rotated.sum(dim=(1,2,3)))


def test_direction_followup_record_and_causal_reload():
    from week03.model.experiment import make_model,predict,score
    import csv
    out=ROOT.parent/'results/runs/directional_sum'
    if not (out/'metrics.csv').exists():pytest.skip('Training pending')
    config=json.loads((ROOT/'direction_config.json').read_text(encoding='utf8'))
    main=json.loads((ROOT.parent/'results/runs/baseline/training.json').read_text(encoding='utf8'))
    extra=json.loads((out/'training.json').read_text(encoding='utf8'))
    assert len(extra['runs'])==3
    for r in extra['runs']:
        original=next(x for x in main['runs'] if x['variant']=='social' and x['seed']==r['seed'])
        for key in ['initial_state_sha256','epoch_order_sha256','parameters']:assert r[key]==original[key]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=make_model(config,'directional_sum')
    model.load_state_dict(torch.load(out/'checkpoints/directional_sum_seed42.pt',weights_only=True,map_location='cpu'))
    with np.load(out/'predictions.npz') as data:
        np.testing.assert_allclose(predict(model,data['test_history'][0]),data['test_directional_sum_seed42'][0],rtol=1e-5,atol=1e-4)
        for row in csv.DictReader((out/'metrics.csv').open(encoding='utf8')):
            key=f'{row["split"]}_constant_velocity' if row['seed']=='-1' else f'{row["split"]}_directional_sum_seed{row["seed"]}'
            actual,_=score(data[key],data[row['split']+'_truth'])
            assert actual['ADE']==pytest.approx(float(row['ADE']),abs=1e-7)
            assert actual['FDE']==pytest.approx(float(row['FDE']),abs=1e-7)
