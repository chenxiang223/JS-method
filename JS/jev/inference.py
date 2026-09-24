"""Load a complete model checkpoint and predict raw, unnormalized HWC patches."""
import numpy as np
import torch
from .model import JevSPARC

class Predictor:
    def __init__(self, checkpoint, device='cpu'):
        data=torch.load(checkpoint,map_location=device,weights_only=False)
        self.device=torch.device(device)
        self.model=JevSPARC(**data['model_kwargs']).to(self.device).eval()
        self.model.load_state_dict(data['model'],strict=True)
        self.mean=data['mean'].numpy(); self.std=data['std'].numpy()
        self.label_values=np.asarray(data['label_values'])
        self.patch_size=data['config']['patch_size']

    @torch.inference_mode()
    def predict(self, raw_patches):
        x=np.asarray(raw_patches,dtype=np.float32)
        if x.ndim==3: x=x[None]
        if x.ndim!=4 or x.shape[1:3]!=(self.patch_size,self.patch_size) or x.shape[-1]!=len(self.mean):
            raise ValueError('Expected [B, patch_size, patch_size, original_bands]')
        if not np.isfinite(x).all(): raise ValueError('Nonfinite input')
        x=torch.from_numpy(((x-self.mean)/self.std).transpose(0,3,1,2).copy()).to(self.device)
        out=self.model(x)
        result={k:getattr(out,k).cpu().numpy() for k in
                ('logits','probabilities','prediction','class_evidence','reliability','uncertainty','abstain')}
        result['original_label']=self.label_values[result['prediction']]
        return result
