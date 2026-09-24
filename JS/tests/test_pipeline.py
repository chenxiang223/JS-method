import numpy as np
import torch
from jev.experiment import parser, run
from jev.inference import Predictor


def test_complete_pipeline_and_saved_inference(tmp_path):
    rng=np.random.default_rng(6)
    gt=np.tile(np.arange(1,4),(15,5)); cube=rng.normal(size=(15,15,8)).astype('float32')
    data=tmp_path/'data';data.mkdir()
    np.save(data/'PaviaU.npy',cube);np.save(data/'PaviaU_gt.npy',gt)
    args=parser().parse_args(['--data-root',str(data),'--output',str(tmp_path/'out'),'--dataset','PaviaU',
        '--epochs','1','--reliability-epochs','1','--channels','16','--patch-size','5',
        '--stress-samples','6','--device','cpu','--threads','2'])
    run(args)
    directory=tmp_path/'out/PaviaU/random/seed0/full'
    pred=Predictor(directory/'model.pt')
    output=pred.predict(cube[:5,:5])
    assert output['probabilities'].shape==(1,3)
    assert np.isclose(output['probabilities'].sum(),1.)
    assert output['original_label'][0] in (1,2,3)
    z=np.load(directory/'predictions.npz'); assert len(z['labels'])>0
    assert (directory/'result.json').exists()
    run(args)  # identical complete run is idempotent
