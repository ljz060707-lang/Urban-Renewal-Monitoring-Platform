"""Transparent, source-clipped roof envelope visualization; never training GT."""
import math
import numpy as np
from shapely.geometry import mapping
from rasterio.features import rasterize
from ..label_transfer import Grid
from ..source_partition import SourceRegion
from .preview import _edge_pixels

ROOF_PREVIEW_QUALITY={0:'NO_SOURCE_AFTER_CANDIDATE_SHIFT',
    1:'UNVERIFIED_NUMERIC_CANDIDATE',2:'UNRESOLVED_NO_VECTOR',
    3:'PREVIEW_CELL_EDGE_AND_INSTANCE_SEAM_RISK',255:'OUTSIDE_SUPPORT'}


def render_roof_overlay_preview(observed_foreground,grid,regions,*,seam_width_px=1):
    """Render original/candidate outline RGBA and explicit quality on one grid.

    Foreground must be a preview-only positive envelope, NOT a hard label.
    Clip source support before moving and clip destination to the same declared
    preview cell; no foreign fragments or unknown-cell filling. Source-cell
    seams are hidden in both layers. No instance repair, acceptance, position
    precision, negative truth, or roof-to-ground interpretation is performed.
    """
    if np.ma.isMaskedArray(observed_foreground):raise ValueError('Explicit positive mask required')
    m=np.asarray(observed_foreground)
    if (not isinstance(grid,Grid) or grid.crs!='EPSG:32649' or
            m.shape!=(grid.height,grid.width) or m.dtype.kind!='b'):
        raise ValueError('Boolean positive envelope on the metric display grid required')
    if isinstance(seam_width_px,bool) or not isinstance(seam_width_px,int) or seam_width_px<0:
        raise ValueError('Nonnegative integer seam quarantine required')
    regions=list(regions)
    if any(not isinstance(r,SourceRegion) for r in regions):raise ValueError('Candidate SourceRegion required')
    ids=[r.region_id for r in regions]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate region ID')
    for i,r in enumerate(regions):
        if any(r.support.intersection(s.support).area>1e-6 for s in regions[i+1:]):
            raise ValueError('Ambiguous overlapping source supports')
    masks=[rasterize([(mapping(r.support),1)],out_shape=m.shape,transform=grid.affine,
                     fill=0,dtype='uint8').astype(bool) for r in regions]
    if any(not x.any() for x in masks):raise ValueError('Preview region has no grid pixels')
    if masks and (np.stack(masks).sum(axis=0)>1).any():raise ValueError('Ambiguous pixel supports')
    quality=np.full(m.shape,255,np.uint8)
    original=np.zeros((4,*m.shape),np.uint8);candidate=original.copy()
    moved_foreground=np.zeros(m.shape,bool);display=np.zeros(m.shape,bool)
    original_edge=_edge_pixels(m,1)
    records=[];h,w=m.shape
    for r,mask in zip(regions,masks):
        record={'region_id':r.region_id,'actual_source_date':r.actual_date,
                'candidate_correction_m':r.correction_m,'evidence_sha256':r.evidence_sha256,
                'candidate_rendered':False,'formal_correction_accepted':False}
        quality[mask]=2
        if r.correction_m is None:
            records.append(record);continue
        a,b,_,d,e,_=tuple(grid.affine)[:6];dx,dy=r.correction_m;det=a*e-b*d
        dc,dr=(e*dx-b*dy)/det,(-d*dx+a*dy)/det
        if any(not math.isclose(x,round(x),rel_tol=0,abs_tol=1e-8) for x in (dc,dr)):
            raise ValueError('No silent rounding or interpolation of subpixel candidate')
        dc,dr=int(round(dc)),int(round(dr))
        # Restrict the ORIGINAL data before movement, not just the destination.
        source=m & mask;warped=np.zeros(m.shape,bool);coverage=warped.copy()
        y0,y1=max(0,-dr),min(h,h-dr);x0,x1=max(0,-dc),min(w,w-dc)
        if y0<y1 and x0<x1:
            warped[y0+dr:y1+dr,x0+dc:x1+dc]=source[y0:y1,x0:x1]
            coverage[y0+dr:y1+dr,x0+dc:x1+dc]=mask[y0:y1,x0:x1]
        warped &= mask;coverage &= mask
        edge=_edge_pixels(mask,seam_width_px)
        quality[mask]=1;quality[mask & ~coverage]=0;quality[edge]=3
        show=mask & coverage & ~edge
        display |= show;moved_foreground |= warped
        raw_show=original_edge & mask & ~edge
        original[:3,raw_show]=np.array([0,220,245],np.uint8)[:,None]
        original[3,raw_show]=255
        record.update(candidate_rendered=True,pixel_shift_col_row=[dc,dr],
            source_positive_pixels=int(source.sum()),candidate_positive_pixels=int(warped.sum()),
            source_missing_after_shift_pixels=int((mask&~coverage).sum()),
            seam_risk_pixels=int(edge.sum()),display_support_pixels=int(show.sum()),
            preview_mode='PIXEL_ENVELOPE_NOT_COMPLETE_BUILDING_INSTANCE_TRANSPORT')
        records.append(record)
    shifted_edge=_edge_pixels(moved_foreground,1) & display
    candidate[:3,shifted_edge]=np.array([255,0,160],np.uint8)[:,None]
    candidate[3,shifted_edge]=255
    return {'original_rgba':original,'candidate_rgba':candidate,'quality':quality,
        'candidate_foreground':moved_foreground,'regions':records,
        'formal_correction_accepted':False,'training_use_allowed':False,
        'absolute_position_accuracy':'UNKNOWN','roof_to_ground_correction':'NOT_ESTABLISHED',
        'zero_semantics':'NO_POSITIVE_ENVELOPE_NOT_KNOWN_BACKGROUND'}
