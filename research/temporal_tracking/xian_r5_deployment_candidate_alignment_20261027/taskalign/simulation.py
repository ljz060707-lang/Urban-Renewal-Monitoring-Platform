"""Named geometry-error simulations, never real birth/death observations."""
from dataclasses import replace
from shapely.geometry import shape,mapping
from shapely.affinity import translate
from shapely.ops import unary_union
from .contracts import digest

def perturb(observation,mode,other=None,native_to_metric=None):
    native=shape(observation.native_geometry);metric=shape(observation.metric_geometry)
    if mode=='MASK_TRANSLATION':native=translate(native,2,-1);metric=translate(metric,2,-1)
    elif mode=='BOUNDARY_PERTURBATION':native=native.buffer(.5);metric=metric.buffer(.5)
    elif mode=='SEGMENTATION_SPLIT':
        from shapely.geometry import box
        x0,y0,x1,y1=native.bounds;native=native.intersection(box(x0,y0,(x0+x1)/2,y1))
        x0,y0,x1,y1=metric.bounds;metric=metric.intersection(box(x0,y0,(x0+x1)/2,y1))
    elif mode=='SEGMENTATION_MERGE':
        if other is None or other.epoch!=observation.epoch:raise ValueError('Merge needs another same-epoch observation')
        if observation.image_path and other.image_path!=observation.image_path:raise ValueError('Cannot merge observations from different native frames')
        native=unary_union([native,shape(other.native_geometry)]);metric=unary_union([metric,shape(other.metric_geometry)])
    else:raise ValueError('Unknown declared simulation type')
    # Simulated source pixels and metric geometry must remain in the same frame.
    # Identity-coordinate toy fixtures need no conversion; real frames do.
    if native_to_metric is not None:metric=native_to_metric(native)
    elif shape(observation.native_geometry).equals(shape(observation.metric_geometry)):metric=native
    else:raise ValueError('Explicit native-to-metric transform required for real-geometry simulation')
    out=replace(observation,uid='SIM_'+digest([observation.uid,mode,getattr(other,'uid',None)])[:24],native_geometry=mapping(native),metric_geometry=mapping(metric),proposal_kind='SYNTHETIC_GEOMETRY_ERROR_NOT_REAL_CHANGE')
    return out,dict(mode=mode,parent_observation=observation.uid,other_parent=getattr(other,'uid',None),image_pixels_transformed=False,
        real_temporal_change=False,visual_alignment_certified=False,change_loss_enabled=False,physical_identity_supervision='separate lineage lookup, merge remains ambiguous')
