"""Spatial-block consistency of one frozen image-vector, never a metric bound."""
import numpy as np
from .supported_ground_features import directional_features,_paired
from .base_tie import _pearson_correlation

def replay_check_blocks(reference_rgb,moving_rgb,reference_mask,moving_mask,component,offset,
                        *,block_size=16,minimum_positive_blocks=3,minimum_gain=.01):
    if (not isinstance(offset,tuple) or len(offset)!=2 or any(type(x) is not int for x in offset)
            or type(block_size) is not int or block_size<4 or minimum_positive_blocks<1):
        raise ValueError('Frozen integer vector and spatial-block policy required')
    yy,xx=np.indices(component.shape);check=(xx//block_size+yy//block_size)%2==1
    ref,rv=directional_features(reference_rgb,reference_mask&component&check)
    mov,mv=directional_features(moving_rgb,moving_mask&component&check)
    r,m,common,_,bounds=_paired(ref,mov,rv,mv,*offset,also_zero=True)
    y0,y1,x0,x1=bounds;blocks=[]
    rows=(yy[y0:y1,x0:x1]//block_size);cols=(xx[y0:y1,x0:x1]//block_size)
    for rr,cc in sorted({(int(a),int(b)) for a,b in zip(rows[common.any(axis=0)],cols[common.any(axis=0)])}):
        population=common&((rows==rr)&(cols==cc))[None]
        n=int(population.any(axis=0).sum())
        if n<3:
            blocks.append({'row':rr,'col':cc,'n':n,'gain':None,'status':'TOO_FEW_FOR_SCALAR_CORRELATION'});continue
        zero=_pearson_correlation(r[population],mov[:,y0:y1,x0:x1][population])
        candidate=_pearson_correlation(r[population],m[population])
        gain=None if zero is None or candidate is None else candidate-zero
        blocks.append({'row':rr,'col':cc,'n':n,'zero_corr':zero,'candidate_corr':candidate,'gain':gain,
                       'status':'INFORMATIVE_INTERNAL_BLOCK' if gain is not None else 'UNINFORMATIVE_BLOCK'})
    informative=[b for b in blocks if b['gain'] is not None]
    positive=sum(b['gain']>=minimum_gain for b in informative)
    materially_negative=sum(b['gain'] < -minimum_gain for b in informative)
    return {'blocks':blocks,'positive_blocks':positive,'materially_negative_blocks':materially_negative,
        'consistent_internal_experiment':positive>=minimum_positive_blocks and materially_negative==0,
        'independent_building_groups':False,'external_accuracy_validation':False,
        'formal_fit_holdout_review':False,'minimum_scalar_block_samples':3,
        'minimum_positive_blocks':minimum_positive_blocks,'minimum_gain':minimum_gain}
