from pathlib import Path
import os
os.environ['OMP_NUM_THREADS']='2'
import json,sys,hashlib
import numpy as np,torch
from rgb_identity_model import RoofAppearanceEncoder
from native_temporal_contract import O,save,sha
sys.path.insert(0,str(Path(__file__).parent/'official_supcontrast'))
from losses import SupConLoss
torch.set_num_threads(2);torch.manual_seed(20261115)
method=Path(__file__).parent/'official_supcontrast';receipt=json.loads((method/'SOURCE_RECEIPT.json').read_text())
for r in receipt['files']:assert sha(method/r['path'])==r['sha256']
m=json.loads((O/'rgb_identity/CROP_MANIFEST.json').read_text());path=O/'rgb_identity'/m['records'][0]['path'];z=np.load(path);x=torch.from_numpy(z['crops'][:4]);assert len(x)==4;model=RoofAppearanceEncoder().train();out=model(x);loss=SupConLoss(.1,base_temperature=.1)(out.reshape(2,2,64),torch.tensor([0,1]));loss.backward();assert torch.isfinite(loss);assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.rgb.layer4.parameters());assert all(p.grad is None for p in model.rgb.layer1.parameters());assert all(not b.training for b in model.rgb.modules() if isinstance(b,torch.nn.BatchNorm2d))
save(O/'rgb_identity/CPU_MODEL_LOSS_SMOKE.json',dict(status='PASS_REAL_PREDICTED_ROOF_CROP_ENCODER_SUPCON_GRADIENT',loss=float(loss),projection_dim=64,pretrained_dim=512,layer4_trainable=True,early_layers_frozen=True,BN_running_statistics_frozen=True,CPU_backward=1,optimizer_updates=0,functional_labels='SYNTHETIC_PAIRING_ONLY_NOT_ACCURACY_OR_TRAINING',protected_reads=0,ImageNet_checkpoint_sha256=sha('/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'),official_loss_commit=receipt['commit']));print('PASS_RGB_IDENTITY_CPU',float(loss))
