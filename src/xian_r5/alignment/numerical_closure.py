"""Convention and same-support design audits, never an alignment acceptance adapter."""
import math
from affine import Affine
from shapely.geometry import Point
from .ground_trial import independent_groups


def content_correction(dx_col,dy_row,affine):
    """Inverse of content displacement, not a second geotransform correction."""
    if not isinstance(affine,Affine):
        raise ValueError('Explicit Affine required; use Affine.from_gdal for GDAL six tuple')
    if any(isinstance(v,bool) or not math.isfinite(v) for v in (dx_col,dy_row,*tuple(affine)[:6])):
        raise ValueError('Finite displacement and grid required')
    return (-affine.a*dx_col-affine.b*dy_row,-affine.d*dx_col-affine.e*dy_row)


def numerical_gate(errors_m,*,convention_error_m):
    errors=list(errors_m)
    return bool(errors and math.isfinite(convention_error_m) and abs(convention_error_m)<=1e-8
        and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=2 for v in errors))


def validate_same_region_design(rows,region_id,support):
    """Audit patch groups only. Precision ties require the existing evidence contract.

    No confidence, numeric peak, anchor or year label is upgraded to a tie.
    This interface does not authorize displacement application.
    """
    rows=list(rows)
    if support.is_empty or not support.is_valid:raise ValueError('Explicit valid region support required')
    for row in rows:
        xy=row.get('anchor_xy')
        if row.get('region_id')!=region_id:raise ValueError('FIT and CHECK must share one region')
        if xy is None or len(xy)!=2 or not all(math.isfinite(v) for v in xy) or not support.covers(Point(*xy)):
            raise ValueError('Candidate anchor outside exact region support')
    groups=independent_groups(rows)
    return dict(region_id=region_id,candidate_fit_groups=len(groups['FIT']),candidate_check_groups=len(groups['CHECK']),
        same_region_candidate_design=bool(groups['FIT'] and groups['CHECK']),
        quantity_requirement_fit=3,quantity_requirement_check=3,
        candidate_quantity_requirement_met=len(groups['FIT'])>=3 and len(groups['CHECK'])>=3,
        precision_eligible_ties=0,formal_fit_check_eligible=False,
        semantics='PATCH_GROUP_DESIGN_ONLY_NOT_FORMAL_PRECISION_TIE_ADMISSION',
        formal_admission_route='EXISTING_FROZEN_PRECISION_EVIDENCE_CONTRACT_NOT_THIS_DESIGN_AUDIT')
