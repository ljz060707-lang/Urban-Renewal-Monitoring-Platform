"""Explicit ground candidate support and whole-structure independence.

No semantic classification, localization bound or roof GT is inferred here.
Candidate ground support is not equivalent to certified stable ground.
"""
import numpy as np


def candidate_ground_support(ground_candidate, excluded_objects, observed_valid):
    arrays=[np.asarray(v) for v in (ground_candidate,excluded_objects,observed_valid)]
    if any(v.dtype.kind!='b' or v.ndim!=2 or v.shape!=arrays[0].shape for v in arrays):
        raise ValueError('Shared-grid explicit boolean masks required')
    return arrays[0] & ~arrays[1] & arrays[2]


def independent_groups(features):
    groups={'FIT':set(),'CHECK':set()}
    for row in features:
        group=row.get('independence_group');fold=row.get('fold')
        if not isinstance(group,str) or not group.strip() or fold not in groups:
            raise ValueError('Explicit whole-structure group and FIT/CHECK required')
        groups[fold].add(group)
    if groups['FIT'] & groups['CHECK']:
        raise ValueError('Same ground structure leaks into FIT and CHECK')
    return {k:sorted(v) for k,v in groups.items()}
