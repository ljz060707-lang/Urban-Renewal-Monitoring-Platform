import numpy as np
from surface_state_v1.datasets.cwld_codec import CWLDCodec

def test_exact_anchors_and_ambiguous_are_not_background():
    rgba=np.array([[[0,0,0,255],[255,255,255,255],[0,0,255,255],[255,0,0,255],[127,0,127,255]]],dtype=np.uint8)
    result=CWLDCodec(max_anchor_distance=32,min_distance_margin=16).encode_array(rgba)
    assert result[0,:4].tolist()==[0,1,2,3]
    assert result[0,4]==255
