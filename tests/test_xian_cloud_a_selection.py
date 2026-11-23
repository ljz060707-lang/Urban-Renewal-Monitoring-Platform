"""Synthetic reproduction of unknown-exterior selection leakage (no files)."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from xian_cloud_a_train import evaluate_a


class ArrayContext:
    def __init__(self,array):self.array=array
    def __enter__(self):return {'reference_roof':self.array}
    def __exit__(self,*args):return False


def test_unknown_exterior_cannot_improve_A_selection_by_disappearing():
    hr=torch.zeros(1,3,16,16);lr=torch.zeros(1,3,4,4)
    masks=torch.zeros(1,2,16,16);masks[0,0,8,8]=1;masks[0,1,1,1]=1
    gt=np.zeros((16,16),np.uint8);gt[8,8]=1
    class Predictor:
        def __init__(self,remove_unknown):self.remove=remove_unknown
        def __call__(self,*args,**kwargs):
            return {'total_shift_px':torch.tensor([[[0.,0.],[32. if self.remove else 0.,0.]]])}
    rows=[dict(training_supervision='SYNTHETIC_ONLY_NOT_A_PATH')]
    with patch('xian_cloud_a_train.load_scene',return_value=(hr,lr,masks,torch.ones(1,1,16,16,dtype=torch.bool))),\
         patch('xian_cloud_a_train.np.load',return_value=ArrayContext(gt)),\
         patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self):
        before=evaluate_a(Predictor(False),Path('SYNTHETIC'),rows)
        disappeared=evaluate_a(Predictor(True),Path('SYNTHETIC'),rows)
    assert before['selection_positive_support_fraction_error']==disappeared['selection_positive_support_fraction_error']==0
    assert disappeared['full_canvas_reference_polygon_set_MAE_diagnostic']<before['full_canvas_reference_polygon_set_MAE_diagnostic']
    assert before['selection_known_support_denominator_per_region']==disappeared['selection_known_support_denominator_per_region']
