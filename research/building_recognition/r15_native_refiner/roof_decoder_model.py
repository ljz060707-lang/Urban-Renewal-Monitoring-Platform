"""Per-image batched SAM decoder; original Meta modules and weights stay intact.

The upstream predictor repeats ONE image for many prompts. Training batches have
different images, so add each image to its own dense prompt without B-by-B repeat.
Based on the Apache-2.0 SAM mask decoder kept in external/sam/source.
"""
import torch
import torch.nn.functional as F
def forward_roof(sam,embedding,points,boxes,priors,use_prior):
 b=len(embedding)
 with torch.no_grad():
  sparse,dense=sam.prompt_encoder(points=(points[:,None],torch.ones((b,1),device=points.device,dtype=torch.int64)),boxes=boxes,masks=priors);empty=sam.prompt_encoder.no_mask_embed.weight.reshape(1,-1,1,1).expand(b,-1,64,64);dense=torch.where(use_prior[:,None,None,None],dense,empty);pos=sam.prompt_encoder.get_dense_pe().expand(b,-1,-1,-1)
 decoder=sam.mask_decoder;tokens=torch.cat([decoder.iou_token.weight,decoder.mask_tokens.weight],dim=0)[None].expand(b,-1,-1);tokens=torch.cat([tokens,sparse],dim=1);hidden,spatial=decoder.transformer(embedding+dense,pos,tokens);spatial=spatial.transpose(1,2).reshape_as(embedding);upscaled=decoder.output_upscaling(spatial);weight=decoder.output_hypernetworks_mlps[0](hidden[:,1]);logits=torch.einsum('bc,bchw->bhw',weight,upscaled)[:,None];quality=decoder.iou_prediction_head(hidden[:,0])[:,0];return logits,quality

def roof_loss(native_logits,quality,target,valid,roi,enabled=None):
 if enabled is None:enabled=torch.ones(len(target),dtype=torch.bool,device=target.device)
 if not enabled.any():return native_logits.sum()*0+quality.sum()*0,dict(enabled=0)
 logits=native_logits[enabled];q=quality[enabled];g=target[enabled].float();v=valid[enabled].float();r=roi[enabled].float();w=v*r;p=logits.sigmoid();axes=(1,2,3);weight=torch.where(g>0,2.,1.);bce=(F.binary_cross_entropy_with_logits(logits,g,reduction='none')*weight*w).sum(axes)/w.sum(axes).clamp_min(1);tp=(p*g*w).sum(axes);fp=(p*(1-g)*w).sum(axes);fn=((1-p)*g*w).sum(axes);tv=(tp+1)/(tp+.65*fn+.35*fp+1);pa=(p*w).sum(axes);ga=(g*w).sum(axes);area=F.smooth_l1_loss(torch.log((pa+1)/(ga+1)),torch.zeros_like(pa),reduction='none');pe=F.max_pool2d(p,3,1,1)+F.max_pool2d(-p,3,1,1);ge=F.max_pool2d(g,3,1,1)+F.max_pool2d(-g,3,1,1);edge_valid=(1-F.max_pool2d(1-v,3,1,1))*r;edge=(2*(pe*ge*edge_valid).sum(axes)+1)/((pe*edge_valid).sum(axes)+(ge*edge_valid).sum(axes)+1)
 with torch.no_grad():
  binary=p>.5;truth=g>0;inter=(binary&truth&(v>0)).sum(axes);union=((binary|truth)&(v>0)).sum(axes);iou=inter/union.clamp_min(1)
 qloss=(q-iou).square();loss=(bce+.75*(1-tv)+.3*area+.2*(1-edge)+.1*qloss).mean();return loss,dict(BCE=float(bce.mean().detach()),Tversky_loss=float((1-tv).mean().detach()),log_area_loss=float(area.mean().detach()),boundary_loss=float((1-edge).mean().detach()),quality_MSE=float(qloss.mean().detach()),enabled=int(enabled.sum()))
