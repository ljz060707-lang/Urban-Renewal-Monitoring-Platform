"""Two declared loss arms, measured on real native pixels after ROI unwarping."""
import torch
import torch.nn.functional as F
def native_logits(low,sizes,maximum):
 return torch.cat([F.pad(F.interpolate(low[i:i+1].float(),(int(s),int(s)),mode='bilinear',align_corners=False),(0,maximum-int(s),0,maximum-int(s))) for i,s in enumerate(sizes)],0)
def loss(logits,quality,target,valid,raw,enabled,recall_emphasis=False):
 if not enabled.any():return logits.sum()*0+quality.sum()*0,dict(enabled=0)
 z=logits[enabled];q=quality[enabled];g=target[enabled].float();v=valid[enabled].float();r=raw[enabled].float();p=z.sigmoid();axes=(1,2,3);pos=4. if recall_emphasis else 2.;fn_weight=.8 if recall_emphasis else .65
 bce=(F.binary_cross_entropy_with_logits(z,g,reduction='none')*torch.where(g>0,pos,1.)*v).sum(axes)/v.sum(axes).clamp_min(1);tp=(p*g*v).sum(axes);fp=(p*(1-g)*v).sum(axes);fn=((1-p)*g*v).sum(axes);tv=(tp+1)/(tp+fn_weight*fn+(1-fn_weight)*fp+1);area=F.smooth_l1_loss(torch.log(((p*v).sum(axes)+1)/((g*v).sum(axes)+1)),torch.zeros_like(tp),reduction='none');pe=F.max_pool2d(p,3,1,1)+F.max_pool2d(-p,3,1,1);ge=F.max_pool2d(g,3,1,1)+F.max_pool2d(-g,3,1,1);ev=1-F.max_pool2d(1-v,3,1,1);edge=(2*(pe*ge*ev).sum(axes)+1)/((pe*ev).sum(axes)+(ge*ev).sum(axes)+1)
 with torch.no_grad():
  binary=p>.5;truth=g>0;inter=(binary&truth&(v>0)).sum(axes);union=((binary|truth)&(v>0)).sum(axes);iou=inter/union.clamp_min(1)
 qloss=(q-iou).square();known_raw_TP=r*g*v;keep=(F.softplus(-z)*known_raw_TP).sum(axes)/known_raw_TP.sum(axes).clamp_min(1);total=(bce+.75*(1-tv)+.3*area+.2*(1-edge)+.1*qloss+(.35 if recall_emphasis else 0.)*keep).mean()
 return total,dict(loss=float(total.detach()),BCE=float(bce.mean().detach()),Tversky_loss=float((1-tv).mean().detach()),log_area_loss=float(area.mean().detach()),boundary_loss=float((1-edge).mean().detach()),quality_MSE=float(qloss.mean().detach()),known_raw_TP_keep_loss=float(keep.mean().detach()),enabled=int(enabled.sum()))
