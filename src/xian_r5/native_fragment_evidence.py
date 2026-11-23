"""Native-grid serialization/strip-fragment evidence, never ground calibration."""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import unary_union


def native_fragment_evidence(native_objects, *, strip_rows=1024, serialization_tolerance_px=.001):
    """Recognize shared strip-core edges in a disposable native-grid view.

    The small tolerance checks round-trip coordinate serialization only. It is
    not an image localization bound and is never exported as metric precision.
    Original metric or source geometries must not be replaced by this view.
    """
    if type(strip_rows) is not int or strip_rows<1 or not math.isfinite(serialization_tolerance_px) or not 0<serialization_tolerance_px<.01:
        raise ValueError('Explicit strip/serialization contract required')
    if not native_objects:raise ValueError('Explicit native object geometries required')
    rounded={};errors={}
    for key,g in native_objects.items():
        if not isinstance(key,str) or not isinstance(g,(Polygon,MultiPolygon)) or not g.is_valid or g.is_empty:
            raise ValueError('Valid keyed native polygon views required')
        coordinates=shapely.get_coordinates(g)
        if not np.isfinite(coordinates).all():raise ValueError('Native coordinates nonfinite')
        error=float(np.abs(coordinates-np.rint(coordinates)).max());errors[key]=error
        if error>serialization_tolerance_px:
            return {'status':'NATIVE_INTEGER_GRID_ALIGNMENT_NOT_ESTABLISHED',
                'coordinate_roundtrip_fraction_px':errors,'fragment_groups':[],
                'source_geometry_modified':False,'formal_precision_bound_m':None}
        rounded[key]=shapely.set_precision(g,1.)
    keys=sorted(rounded);edges=[];neighbors={k:set() for k in keys}
    for i,key in enumerate(keys):
        for other in keys[i+1:]:
            boundary=rounded[key].boundary.intersection(rounded[other].boundary)
            for part in shapely.get_parts(boundary):
                if part.geom_type!='LineString':continue
                xy=np.asarray(part.coords)
                # Split polylines so a corner does not impersonate a straight
                # horizontal strip core edge.
                for a,b in zip(xy[:-1],xy[1:]):
                    if a[1]!=b[1] or a[0]==b[0] or int(a[1])%strip_rows!=0:continue
                    length=float(abs(a[0]-b[0]))
                    edges.append({'left_object':key,'right_object':other,'native_row':int(a[1]),
                        'shared_horizontal_edge_length_px':length,'strip_row_multiple':strip_rows})
                    neighbors[key].add(other);neighbors[other].add(key)
    groups=[];used=set()
    for key in keys:
        if key in used:continue
        members=[];todo=[key];used.add(key)
        while todo:
            current=todo.pop();members.append(current)
            for other in sorted(neighbors[current]):
                if other not in used:used.add(other);todo.append(other)
        if len(members)>1:
            union=unary_union([rounded[k] for k in members])
            groups.append({'historical_object_ids':sorted(members),'native_union_area_px2':union.area,
                'native_union_parts':len(list(shapely.get_parts(union))),
                'role':'STRIP_FRAGMENT_SET_NOT_PHYSICAL_BUILDING_ID'})
    return {'status':'SHARED_NATIVE_STRIP_EDGE_EVIDENCE' if groups else 'NO_SHARED_NATIVE_STRIP_EDGE',
        'coordinate_roundtrip_fraction_px':errors,'shared_edges':edges,'fragment_groups':groups,
        'source_geometry_modified':False,'formal_precision_bound_m':None,
        'native_view_may_replace_source_geometry':False,'new_physical_ids':0}
