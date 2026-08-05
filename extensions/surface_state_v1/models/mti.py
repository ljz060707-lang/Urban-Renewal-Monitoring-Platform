from __future__ import annotations
import torch

class MultiTaskIntegration:
    """Explainable two-state temporal Viterbi integration; no external model import."""
    def __init__(self,change_weight=.5): self.change_weight=float(change_weight)
    def __call__(self,building_probabilities,change_probabilities,timestamps=None,temporal_valid_mask=None,full20_evidence=None,rsbuilding_evidence=None):
        p=building_probabilities.clamp(1e-6,1-1e-6); b,t,h,w=p.shape; unary=torch.stack([(1-p).log(),p.log()],-1); states=[unary[:,0]]
        for i in range(1,t):
            cp=change_probabilities[:,i-1].clamp(0,1)[...,None,None]*self.change_weight; transition=torch.zeros((b,h,w,2,2),device=p.device); transition[...,0,1]=cp[...,0,0]; transition[...,1,0]=cp[...,0,0]; states.append((states[-1][...,None]+transition).max(-2).values+unary[:,i])
        sequence=torch.stack(states,1).argmax(-1); new=(sequence[:,1:]==1)&(sequence[:,:-1]==0); demo=(sequence[:,1:]==0)&(sequence[:,:-1]==1)
        return {"optimal_building_sequence":sequence,"new_building_sequence":new,"demolition_sequence":demo,"persistence_sequence":(sequence[:,1:]==sequence[:,:-1]),"integration_confidence":p,"transition_costs":{"change_weight":self.change_weight}}
