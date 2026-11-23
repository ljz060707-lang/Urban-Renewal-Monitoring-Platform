"""Exact polygon-intersection shadow fractions; never supervision permission.

Source fragments may be translated for diagnostics, including seam-crossing
components. These are not complete-building transport: their affected pixels
are flagged. Unknown vectors remain unknown; source zeros are not background.
"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from .label_transfer import bounded_transport
from .source_partition import SourceRegion

NODATA = -9999.0
EPS = 1e-6
QUALITY_BANDS = ["SUMMARY", "SOURCE_SCOPE_SUPPORTED", "NUMERIC_VECTOR_AVAILABLE",
                 "DISPLACEMENT_VERIFIED", "DATE_COMPATIBLE", "BOUNDARY_RISK",
                 "SEAM_RISK", "OVERLAP_CONFLICT", "SOURCE_ZERO_AMBIGUOUS",
                 "TRAINING_ELIGIBLE"]
QUALITY_CODES = {0:"RELIABLE_UNUSED_THIS_SHADOW", 1:"UNVERIFIED_CANDIDATE",
                 2:"UNKNOWN_VECTOR_OR_SOURCE", 3:"BOUNDARY_OR_SEAM_RISK",
                 4:"OVERLAP_CONFLICT"}


def _polygon_ok(g):
    return (isinstance(g,(Polygon,MultiPolygon)) and not g.is_empty and
            g.is_valid and math.isfinite(g.area) and g.area > 0)


def _sum_intersections(geometries, cells):
    """Sum per-object positive-area intersections and component counts."""
    values=np.zeros(len(cells)); counts=np.zeros(len(cells),dtype=np.uint32)
    if not geometries:return values,counts
    geoms=np.array(geometries,dtype=object);tree=shapely.STRtree(geoms)
    for start in range(0,len(cells),1024):
        block=cells[start:start+1024]
        pairs=tree.query(block,predicate="intersects")
        if pairs.size==0:continue
        areas=shapely.area(shapely.intersection(block[pairs[0]],geoms[pairs[1]]))
        positive=areas>EPS
        np.add.at(values,start+pairs[0][positive],areas[positive])
        np.add.at(counts,start+pairs[0][positive],1)
    return values,counts


def _areas_and_counts(geometries,cells):
    """Union areas, count original components, flag duplicate-area conflicts."""
    geometries=list(geometries)
    summed,counts=_sum_intersections(geometries,cells)
    conflict=np.zeros(len(cells),dtype=bool)
    if not geometries:return summed,counts,conflict
    geoms=np.array(geometries,dtype=object);tree=shapely.STRtree(geoms)
    pairs=tree.query(geoms,predicate="intersects")
    selected=pairs[0]<pairs[1]
    pairs=pairs[:,selected]
    if pairs.size:
        overlap=shapely.intersection(geoms[pairs[0]],geoms[pairs[1]])
        overlap=overlap[shapely.area(overlap)>EPS]
        if len(overlap):
            union=shapely.union_all(geoms)
            union_area,_=_sum_intersections(list(shapely.get_parts(union)),cells)
            overlap_area,_=_sum_intersections(list(overlap),cells)
            conflict=overlap_area>EPS
            return union_area,counts,conflict
    return summed,counts,conflict


def build_fraction_shadow(objects, regions, grid):
    """Intersect immutable metric roof candidates with the whole pixel area.

    Returns raw/candidate fractions, transported geometric scope fractions,
    per-pixel quality and component counts. No valid-background inference,
    calibrated error envelope or temporal admission is performed here.
    """
    regions=list(regions)
    if grid.crs!="EPSG:32649":raise ValueError("Metric EPSG:32649 grid required")
    if not regions or any(not isinstance(r,SourceRegion) for r in regions):
        raise ValueError("Explicit source regions required")
    if len({r.region_id for r in regions})!=len(regions):
        raise ValueError("Duplicate source region")
    for i,left in enumerate(regions):
        for right in regions[i+1:]:
            if left.support.intersection(right.support).area>EPS:
                raise ValueError("Overlapping source scopes unresolved")
    if any(not isinstance(k,str) or not _polygon_ok(g) for k,g in objects.items()):
        raise ValueError("Explicit component keys and valid polygon geometries required")
    cells=np.array([grid.cell(c,r) for r in range(grid.height)
                    for c in range(grid.width)],dtype=object)
    footprint=Polygon([grid.pixel_to_map(c,r) for c,r in
                       ((0,0),(grid.width,0),(grid.width,grid.height),(0,grid.height))])
    source_scope=unary_union([r.support for r in regions]).intersection(footprint)
    raw={k:g.intersection(source_scope) for k,g in objects.items()}
    raw={k:g for k,g in raw.items() if g.area>EPS}
    candidates={};moved_scopes=[];numeric_scopes=[];seam_shapes=[];edge_shapes=[];ledger=[]
    for key,g in objects.items():
        hits=[r for r in regions if g.intersection(r.support).area>EPS]
        if not hits:continue
        seam=len(hits)>1
        partial=not source_scope.covers(g)
        if seam:seam_shapes.append(g.intersection(source_scope))
        if partial:edge_shapes.append(g.intersection(source_scope))
        fragments=[]
        for region in hits:
            if region.correction_m is None:continue
            moved,valid=bounded_transport({key:g},region.correction_m,
                                          region.support,region.support.intersection(footprint))
            fragment=moved[key]
            if fragment.area<=EPS:continue
            fragments.append(fragment)
            if seam:seam_shapes.append(fragment)
            if partial:edge_shapes.append(fragment)
        if fragments:candidates[key]=unary_union(fragments)
        ledger.append({"component_key":key,"source_regions":[r.region_id for r in hits],
            "source_dates":sorted({r.actual_date for r in hits}),
            "source_seam_component":seam,"partial_source_component":partial,
            "raw_area_in_scope_m2":raw.get(key,Polygon()).area,
            "candidate_area_in_scope_m2":candidates.get(key,Polygon()).area,
            "candidate_fragment_diagnostic_only":True,"physical_building_id":None})
    for region in regions:
        if region.correction_m is None:continue
        _,valid=bounded_transport({},region.correction_m,region.support,
                                  region.support.intersection(footprint))
        moved_scopes.append(valid);numeric_scopes.append(region.support.intersection(footprint))
        # 10 m diagnostic strip, not a measured location-error bound.
        strip=region.support.difference(region.support.buffer(-10)).intersection(footprint)
        edge_shapes.append(strip)
    raw_area,raw_count,raw_conflict=_areas_and_counts(raw.values(),cells)
    candidate_area,candidate_count,candidate_conflict=_areas_and_counts(candidates.values(),cells)
    scope_area,_=_sum_intersections([source_scope],cells)
    moved_union=unary_union(moved_scopes)
    moved_area,_=_sum_intersections([moved_union] if not moved_union.is_empty else [],cells)
    numeric_union=unary_union(numeric_scopes)
    numeric_area,_=_sum_intersections([numeric_union] if not numeric_union.is_empty else [],cells)
    seam_area,_=_sum_intersections([unary_union(seam_shapes)] if seam_shapes else [],cells)
    edge_area,_=_sum_intersections([unary_union(edge_shapes)] if edge_shapes else [],cells)
    pixel_area=abs(grid.affine.determinant)
    if (raw_area>scope_area+EPS).any() or (candidate_area>moved_area+EPS).any():
        raise ValueError("Roof evidence exceeds source scope")
    raw_fraction=raw_area/pixel_area;candidate_fraction=candidate_area/pixel_area
    available=moved_area>EPS
    candidate_fraction[~available]=NODATA
    raw_fraction[scope_area<=EPS]=NODATA
    difference=np.full(len(cells),NODATA)
    compare=available & (scope_area>EPS)
    difference[compare]=candidate_fraction[compare]-raw_fraction[compare]
    seam=seam_area>EPS
    boundary=(edge_area>EPS)|((moved_area>EPS)&(moved_area<pixel_area-EPS))
    conflict=raw_conflict|candidate_conflict
    quality=np.zeros((len(QUALITY_BANDS),len(cells)),dtype=np.uint16)
    quality[0]=2;quality[0,available]=1
    quality[0,available&(boundary|seam)]=3
    quality[0,conflict]=4
    quality[1]=scope_area>EPS;quality[2]=numeric_area>EPS
    # Verification/date/background completeness are NOT established.
    quality[5]=boundary;quality[6]=seam;quality[7]=conflict;quality[8]=1
    full_compare=compare&(scope_area>=pixel_area-EPS)&(moved_area>=pixel_area-EPS)
    core_compare=full_compare&~boundary&~seam&~conflict
    expected_raw=unary_union(list(raw.values())).intersection(footprint).area
    expected_candidate=unary_union(list(candidates.values())).intersection(footprint).area
    errors={"raw_area_m2":abs(raw_area.sum()-expected_raw),
            "candidate_area_m2":abs(candidate_area.sum()-expected_candidate)}
    if max(errors.values())>1e-3:raise ValueError("Area conservation failed")
    shape=(grid.height,grid.width)
    return {"raw_fraction":raw_fraction.reshape(shape),
            "candidate_fraction":candidate_fraction.reshape(shape),
            "difference":difference.reshape(shape),
            "quality":quality.reshape(len(QUALITY_BANDS),*shape),
            "component_count":np.stack([raw_count,candidate_count]).reshape(2,*shape),
            "scope_fraction":np.stack([scope_area/pixel_area,moved_area/pixel_area]).reshape(2,*shape),
            "full_compare":full_compare.reshape(shape),"core_compare":core_compare.reshape(shape),
            "component_ledger":ledger,"area_conservation_error_m2":errors,
            "input_component_count":len(objects),"training_eligible_pixels":0,
            "fraction_semantics":"RECORDED_POSITIVE_ROOF_AREA_PER_FULL_PIXEL_NOT_KNOWN_BACKGROUND",
            "candidate_mode":"SOURCE_FRAGMENT_DIAGNOSTIC_NOT_COMPLETE_INSTANCE_TRANSPORT"}


def compare_fraction_probes(result, old_binary=None, threshold=.43):
    """Frozen strict-threshold and difference probes, never new hard labels."""
    if threshold!=.43:raise ValueError("This diagnostic freezes 0.43, no search")
    raw=result["raw_fraction"];candidate=result["candidate_fraction"]
    full=result["full_compare"];core=result["core_compare"]
    flags=np.full(raw.shape,255,dtype=np.uint8)
    flags[full]=(raw[full]>.43).astype(np.uint8)+2*(candidate[full]>.43).astype(np.uint8)
    summaries={}
    for name,mask in (("full_source_scope",full),("excluding_boundary_seam_conflict",core)):
        delta=np.abs(candidate[mask]-raw[mask])
        summaries[name]={"denominator":int(mask.sum()),
            "numerically_unchanged_pixels":int((delta<=1e-6).sum()),
            "abs_difference_le_0_10_pixels":int((delta<=.10).sum()),
            "abs_difference_gt_0_10_pixels":int((delta>.10).sum()),
            "abs_difference_gt_0_25_pixels":int((delta>.25).sum()),
            "maximum_absolute_difference":float(delta.max()) if len(delta) else None,
            "mean_absolute_difference_diagnostic":float(delta.mean()) if len(delta) else None,
            "strict_0_43_flip_pixels":int(np.isin(flags[mask],[1,2]).sum()),
            "probe_0_to_1":int((flags[mask]==2).sum()),"probe_1_to_0":int((flags[mask]==1).sum()),
            "stability_is_validation_or_admission":False}
    old_diff=np.full((2,*raw.shape),255,dtype=np.uint8)
    if old_binary is not None:
        old_binary=np.asarray(old_binary)
        if old_binary.shape!=raw.shape or old_binary.dtype.kind not in "iu" or not np.isin(old_binary,[0,1,255]).all():
            raise ValueError("Old binary grid/codes mismatch")
        oldvalid=np.isin(old_binary,[0,1])
        rvalid=oldvalid&(raw!=NODATA)
        cvalid=oldvalid&full
        old_diff[0,rvalid]=((raw[rvalid]>.43)!=old_binary[rvalid]).astype(np.uint8)
        old_diff[1,cvalid]=((candidate[cvalid]>.43)!=old_binary[cvalid]).astype(np.uint8)
        summaries["old_binary_comparison"]={"raw_denominator":int(rvalid.sum()),
            "old_vs_raw_probe_mismatches":int((old_diff[0]==1).sum()),
            "candidate_full_scope_denominator":int(cvalid.sum()),
            "old_vs_candidate_probe_mismatches":int((old_diff[1]==1).sum()),
            "cause":"NOT_IDENTIFIABLE_FROM_OLD_BINARY; parent fraction missing, timing/geometry not independently established"}
    return {"probe_codes":flags,"old_mismatch":old_diff,"statistics":summaries}
