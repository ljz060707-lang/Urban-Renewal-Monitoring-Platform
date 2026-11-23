"""Isolated Xi'an candidate V2 rebuild and full held-edge recovery."""
from __future__ import annotations
import argparse, ast, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from _tmp_xian_final_candidate_rebuild import DATES, SCENES, TRANSITIONS, normalize_edge, progress, sb, sf, si, text, write_csv, write_json, DSU

FIXED = ["B02-P2-24","B03-TEX04","B04-TQA07","B04-TQA10","B05-TQA03","B05-P1-10","B01-P2-26","OBJ_0001","OBJ_0008","B05-TEX01","B01-TEX02","B05-P1-06"]

def key(e):
    return (text(e.get("from_scene", e.get("source_epoch"))), text(e.get("to_scene", e.get("target_epoch"))), text(e.get("source_object_id")), text(e.get("target_object_id")))

def dct(v):
    try:
        x = ast.literal_eval(text(v))
        return x if isinstance(x, dict) else {}
    except (SyntaxError, ValueError):
        return {}

def src_flags(source, e):
    s = text(source)
    out = set()
    if s == "STRICT_CORE": out.add("STRICT_CORE")
    for flag in ("V4B_STRUCTURAL_CLOSURE","COMPOUND_SPLIT_MERGE","COMPOUND_SPLIT","COMPOUND_MERGE","GAP_RETURN","ABA_SAFE_CLOSURE"):
        if flag in s: out.add(flag)
    if "PARALLAX_SAFE" in s: out.add("PARALLAX_SAFE_RECOVERY")
    elif "SAFE_RECOVERY" in s: out.add("SAFE_RECOVERY")
    if "DIRECT_TEMPORAL_SUPPORT_ONLY" in s or (text(e.get("from_scene")) == "xa180429" and text(e.get("to_scene")) == "xa201231"):
        out.add("DIRECT_TEMPORAL_SUPPORT_ONLY")
    return out

def transition_ok(e):
    return (text(e.get("from_scene")), text(e.get("to_scene"))) in {("xa180429","xa190101"),("xa190101","xa201231")}

def source_priority(flags):
    for name in ("STRICT_CORE", "V4B_STRUCTURAL_CLOSURE", "GAP_RETURN", "COMPOUND_SPLIT_MERGE", "COMPOUND_SPLIT", "COMPOUND_MERGE", "ABA_SAFE_CLOSURE", "PARALLAX_SAFE_RECOVERY", "SAFE_RECOVERY", "DIRECT_TEMPORAL_SUPPORT_ONLY"):
        if name in flags:
            return name
    return "REVIEW_ONLY"

def classify(e):
    relation = text(e.get("candidate_relation")) or text(e.get("relation"))
    sm = text(e.get("split_merge_type")) or "NONE"
    if sb(e.get("identity_conflict")):
        return "KEEP_REVIEW", "identity_conflict"
    if relation != "1:1" or sm not in {"","NONE"}:
        return "STRUCTURAL_COMPOUND", "non_1_to_1_or_split_merge"
    if si(e.get("reasonable_candidate_count_from_source"),99) != 1 or si(e.get("reasonable_candidate_count_to_target"),99) != 1:
        return "KEEP_REVIEW", "strong_competitor"
    regular = sf(e.get("score")) >= .58 and sf(e.get("buffered_iou")) >= .55 and sf(e.get("area_ratio")) >= .55 and sf(e.get("local_neighborhood_consistency")) >= .60 and sf(e.get("alignment_aware_displacement_m")) <= 50
    if regular:
        return "SAFE_RECOVERY", "calibrated_regular_safe_region"
    parallax = sf(e.get("score")) >= .55 and sf(e.get("buffered_iou")) >= .40 and sf(e.get("area_ratio")) >= .45 and sf(e.get("local_neighborhood_consistency")) >= .50 and 45 < sf(e.get("alignment_aware_displacement_m")) <= 90
    if parallax:
        return "PARALLAX_SAFE", "calibrated_parallax_safe_region"
    if sf(e.get("score")) < .40 or sf(e.get("buffered_iou")) < .15 or sf(e.get("area_ratio")) < .20 or sf(e.get("local_neighborhood_consistency")) < .30:
        return "REJECT", "below_minimum_evidence_floor"
    return "KEEP_REVIEW", "outside_validated_safe_region"

def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-root", type=Path, required=True)
    ap.add_argument("--v4b-root", type=Path, required=True)
    ap.add_argument("--candidate-v1", type=Path, required=True)
    ap.add_argument("--repair-root", type=Path, required=True)
    ap.add_argument("--visual-queue", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise RuntimeError("refusing to overwrite existing output")
    out = a.output
    out.mkdir(parents=True)
    formal, v4b, v1, repair = a.formal_root, a.v4b_root, a.candidate_v1, a.repair_root
    progress(out, "INIT", iteration="V2", next_action="load frozen objects")

    obj = pd.read_csv(formal/"04_identity_graph"/"tracking_objects_unfiltered.csv", encoding="utf-8-sig", low_memory=False, usecols=["scene_key","formal_date","building_id","track_id","object_uid","source_asset_id","local_actual_date","area_m2"])
    obj["object_uid"] = obj["object_uid"].astype(str); obj["scene_key"] = obj["scene_key"].astype(str); obj["building_id"] = obj["building_id"].astype(str); obj["area_m2"] = pd.to_numeric(obj["area_m2"], errors="coerce").fillna(0.0)
    uids = obj.object_uid.tolist(); uid_idx = {u:i for i,u in enumerate(uids)}
    scene_arr = obj.scene_key.to_numpy(); building_arr = obj.building_id.to_numpy(); area_arr = obj.area_m2.to_numpy(dtype=float)

    v1obj = pd.read_csv(v1/"FINAL_CANDIDATE_TRACKING_OBJECTS.csv", encoding="utf-8-sig", low_memory=False)
    v1obj.object_uid = v1obj.object_uid.astype(str); v1obj.physical_track_id = v1obj.physical_track_id.astype(str)
    v1obj.identity_sources = v1obj.identity_sources.fillna("").astype(str)
    base_pid = dict(zip(v1obj.object_uid, v1obj.physical_track_id)); base_src = dict(zip(v1obj.object_uid, v1obj.identity_sources))
    if set(base_pid) != set(uids):
        raise RuntimeError("candidate-v1 object inventory mismatch")

    dsu = DSU(len(uids)); groups = defaultdict(list)
    for u,p in base_pid.items(): groups[p].append(u)
    for members in groups.values():
        for u in members[1:]: dsu.union(uid_idx[members[0]], uid_idx[u])

    edge_records = {}; edge_flags = defaultdict(set); edge_reasons = defaultdict(list)
    old = pd.read_csv(v1/"FINAL_IDENTITY_SOURCE_LEDGER.csv", encoding="utf-8-sig", low_memory=False)
    for r in old.to_dict("records"):
        e = {"source_object_id":text(r.get("source_object_id")),"target_object_id":text(r.get("target_object_id")),"from_scene":text(r.get("source_epoch")),"to_scene":text(r.get("target_epoch")),"score":sf(r.get("score")),"buffered_iou":sf(r.get("buffered_iou")),"area_ratio":sf(r.get("area_ratio")),"alignment_aware_displacement_m":sf(r.get("displacement")),"local_neighborhood_consistency":sf(r.get("neighborhood_score")),"score_gap_source":sf(r.get("source_margin")),"score_gap_target":sf(r.get("target_margin")),"candidate_relation":text(r.get("candidate_relation")),"split_merge_type":text(r.get("split_merge_type")),"identity_conflict":text(r.get("identity_conflict")).lower() in {"true","yes","1"}}
        if e["source_object_id"] not in uid_idx or e["target_object_id"] not in uid_idx: continue
        k = key(e); edge_records.setdefault(k,e); edge_flags[k].update(src_flags(r.get("candidate_source"),e))
        if text(r.get("reason")): edge_reasons[k].append(text(r.get("reason")))

    edge_cols = ["source_object_id","target_object_id","from_scene","to_scene","source_index","target_index","distance_m","polygon_iou","buffered_iou","bbox_iou","area_ratio","shape_similarity","local_neighborhood_consistency","score","identity_confidence","alignment_aware_displacement_m","relation","candidate_relation","split_merge_type","identity_conflict","accepted_for_identity","candidate_count_from_source","candidate_count_to_target","reasonable_candidate_count_from_source","reasonable_candidate_count_to_target","score_gap_source","score_gap_target","mutual_best"]
    strict_pre=[]; held=[]
    for batch in pq.ParquetFile(formal/"04_identity_graph"/"identity_edges.parquet").iter_batches(batch_size=250000, columns=edge_cols):
        data=batch.to_pydict()
        for i in range(batch.num_rows):
            r={k:data[k][i] for k in edge_cols}
            if not sb(r.get("accepted_for_identity")): continue
            e=normalize_edge(r,"FORMAL_ACCEPTED")
            is_strict=(not e["identity_conflict"] and e["candidate_relation"]=="1:1" and e["split_merge_type"] in {"","NONE"} and e["score"]>=.60 and e["buffered_iou"]>=.55 and e["area_ratio"]>=.55 and e["local_neighborhood_consistency"]>=.60 and e["alignment_aware_displacement_m"]<=50)
            (strict_pre if is_strict else held).append(e)
    src_bucket=defaultdict(list); dst_bucket=defaultdict(list)
    for e in strict_pre:
        pair=(e["from_scene"],e["to_scene"]); src_bucket[(*pair,e["source_object_id"])].append(e); dst_bucket[(*pair,e["target_object_id"])].append(e)
    keep=[]
    for e in strict_pre:
        pair=(e["from_scene"],e["to_scene"]); unique=len(src_bucket[(*pair,e["source_object_id"])])==1 and len(dst_bucket[(*pair,e["target_object_id"])])==1; margin=min(e["score_gap_source"],e["score_gap_target"])>=.15
        if unique or (margin and e["mutual_best"]): keep.append(e)
    sc=Counter((e["from_scene"],e["to_scene"],e["source_object_id"]) for e in keep); dc=Counter((e["from_scene"],e["to_scene"],e["target_object_id"]) for e in keep)
    strict_rows=[e for e in keep if sc[(e["from_scene"],e["to_scene"],e["source_object_id"])]==1 and dc[(e["from_scene"],e["to_scene"],e["target_object_id"])]==1]
    strict_by_key={}; strict_occ=Counter()
    for e in strict_rows:
        strict_by_key.setdefault(key(e),e); strict_occ[key(e)]+=1; edge_records.setdefault(key(e),e); edge_flags[key(e)].add("STRICT_CORE")
    progress(out,"LOAD_EDGES",strict_input_rows=len(strict_rows),strict_unique_keys=len(strict_by_key),held_rows=len(held),next_action="promote V4B and held recovery")

    comp_cols=["component_id","canonical_compound_track_id","component_status","scene_object_count_2018","scene_object_count_2019","scene_object_count_2020","scene_union_area_m2","structural_types","candidate_relations","presence_values_2018","presence_values_2019","presence_values_2020","object_uids","resolution_reasons","identity_confidence"]
    comp_df=pd.read_csv(v4b/"COMPOUND_TRACKS.csv",encoding="utf-8-sig",low_memory=False,usecols=lambda c:c in comp_cols)
    comp_meta={}; uid_comp={}
    for r in comp_df.to_dict("records"):
        cid=text(r.get("component_id")); nodes=[u for u in text(r.get("object_uids")).split(";") if u in uid_idx]; status=text(r.get("component_status")); comp_meta[cid]={"row":r,"nodes":nodes,"status":status}
        if status in {"RESOLVED_CONTINUOUS_IDENTITY_COMPONENT","RESOLVED_GAP_RETURN"}:
            for u in nodes: uid_comp[u]=cid
    resolved={cid for cid,m in comp_meta.items() if m["status"].startswith("RESOLVED")}
    comp_state={}
    for name in ("RESOLVED_SPLIT_MERGE.csv","RESOLVED_MULTIPLE_CANDIDATES.csv","RESOLVED_GAP_RETURN.csv"):
        p=v4b/name
        if not p.exists(): continue
        d=pd.read_csv(p,encoding="utf-8-sig",low_memory=False,usecols=lambda c:c in {"component_id","transition","resolved_formal_state"})
        for r in d.to_dict("records"):
            if text(r.get("component_id")) and text(r.get("transition")) and text(r.get("resolved_formal_state")):
                comp_state[(text(r["component_id"]),text(r["transition"]))]=text(r["resolved_formal_state"])
    for cid in resolved:
        nodes=comp_meta[cid]["nodes"]
        if not nodes: continue
        raw=comp_meta[cid]["row"]; structural=text(raw.get("structural_types")).upper(); flags={"V4B_STRUCTURAL_CLOSURE"}
        if "SPLIT_MERGE" in structural: flags.add("COMPOUND_SPLIT_MERGE")
        elif "SPLIT" in structural: flags.add("COMPOUND_SPLIT")
        elif "MERGE" in structural: flags.add("COMPOUND_MERGE")
        if comp_meta[cid]["status"]=="RESOLVED_GAP_RETURN": flags.add("GAP_RETURN")
        for u in nodes[1:]:
            dsu.union(uid_idx[nodes[0]],uid_idx[u])
            e={"source_object_id":nodes[0],"target_object_id":u,"from_scene":scene_arr[uid_idx[nodes[0]]],"to_scene":scene_arr[uid_idx[u]],"score":sf(raw.get("identity_confidence")),"buffered_iou":sf(raw.get("aligned_union_iou_mean")),"area_ratio":sf(raw.get("union_area_m2")),"alignment_aware_displacement_m":sf(raw.get("max_alignment_aware_displacement_m")),"local_neighborhood_consistency":0.0,"candidate_relation":"COMPOUND_PROMOTION","split_merge_type":structural,"identity_conflict":False}
            edge_records.setdefault(key(e),e); edge_flags[key(e)].update(flags); edge_reasons[key(e)].append("promoted V4B component "+cid)

    calibration={}
    review_path=repair/"REPAIRED_AFFECTED_VISUAL_REVIEW.csv"; queue_path=a.visual_queue
    if review_path.exists() and queue_path.exists():
        review=pd.read_csv(review_path,encoding="utf-8-sig",low_memory=False); queue=pd.read_csv(queue_path,encoding="utf-8-sig",low_memory=False).drop_duplicates("sample_id")
        joined=review.merge(queue,on="sample_id",how="inner",suffixes=("_review","_queue"))
        for r in joined.to_dict("records"):
            dec=text(r.get("recovery_decision")); lab=text(r.get("visual_label_review")) or text(r.get("visual_label"))
            if dec not in {"SAFE_RECOVERY","PARALLAX_SAFE_RECOVERY"} or lab not in {"CORRECT","PLAUSIBLE"}: continue
            e={"source_object_id":text(r.get("source_object_id")),"target_object_id":text(r.get("target_object_id")),"from_scene":text(r.get("from_scene")),"to_scene":text(r.get("to_scene"))}
            if e["source_object_id"] and e["target_object_id"]: calibration[key(e)]=("PARALLAX_SAFE" if dec=="PARALLAX_SAFE_RECOVERY" else "SAFE_RECOVERY","inherited visual calibration edge")

    held_out=[]; held_dec={}
    for e in held:
        k=key(e); dec,reason=calibration.get(k,classify(e)); held_dec[k]=dec
        held_out.append({"source_epoch":e["from_scene"],"target_epoch":e["to_scene"],"source_object_id":e["source_object_id"],"target_object_id":e["target_object_id"],"pair":e["from_scene"]+"__"+e["to_scene"],"score":e["score"],"buffered_iou":e["buffered_iou"],"area_ratio":e["area_ratio"],"displacement":e["alignment_aware_displacement_m"],"neighborhood":e["local_neighborhood_consistency"],"source_margin":e["score_gap_source"],"target_margin":e["score_gap_target"],"competitor_count":max(e["reasonable_candidate_count_from_source"],e["reasonable_candidate_count_to_target"]),"candidate_relation":e["candidate_relation"],"split_merge_type":e["split_merge_type"],"identity_conflict":"YES" if e["identity_conflict"] else "NO","hold_reason":"FORMAL_ACCEPTED_NON_STRICT","decision":dec,"decision_reason":reason,"identity_support_eligible":"YES","formal_transition_eligible":"YES" if transition_ok(e) else "NO"})
        if dec in {"SAFE_RECOVERY","PARALLAX_SAFE"}:
            dsu.union(uid_idx[e["source_object_id"]],uid_idx[e["target_object_id"]]); edge_records.setdefault(k,e); edge_flags[k].add("SAFE_RECOVERY" if dec=="SAFE_RECOVERY" else "PARALLAX_SAFE_RECOVERY"); edge_reasons[k].append(reason)
    write_csv(out/"HELD_FULL_RECOVERY_DECISION.csv",held_out,list(held_out[0]) if held_out else ["source_object_id"])

    aba_used=0
    aba_path=repair/"ABA_SAFE_CLOSURE_CANDIDATES.csv"
    if aba_path.exists():
        aba=pd.read_csv(aba_path,encoding="utf-8-sig",low_memory=False); aba=aba[aba["decision"].astype(str).eq("SAFE_ABA_IDENTITY_CLOSURE_CANDIDATE")]
        for r in aba.to_dict("records"):
            src,mid,dst=text(r.get("source_2018")),text(r.get("middle_2019")),text(r.get("target_2020"))
            if src not in uid_idx or dst not in uid_idx: continue
            pairs=[(src,mid,"xa180429","xa190101","bridge_ab_score","bridge_ab_buffered_iou","bridge_ab_area_ratio","bridge_ab_displacement_m"),(mid,dst,"xa190101","xa201231","bridge_bc_score","bridge_bc_buffered_iou","bridge_bc_area_ratio","bridge_bc_displacement_m")] if mid in uid_idx else [(src,dst,"xa180429","xa201231","direct_score","direct_buffered_iou","direct_area_ratio","direct_displacement")]
            for aa,bb,fs,ts,cs,ci,ca,cd in pairs:
                e={"source_object_id":aa,"target_object_id":bb,"from_scene":fs,"to_scene":ts,"score":sf(r.get(cs)),"buffered_iou":sf(r.get(ci)),"area_ratio":sf(r.get(ca)),"alignment_aware_displacement_m":sf(r.get(cd)),"local_neighborhood_consistency":sf(r.get("direct_neighborhood_consistency")),"candidate_relation":"1:1","identity_conflict":False}
                dsu.union(uid_idx[aa],uid_idx[bb]); edge_records.setdefault(key(e),e); edge_flags[key(e)].add("ABA_SAFE_CLOSURE")
                if fs=="xa180429" and ts=="xa201231": edge_flags[key(e)].add("DIRECT_TEMPORAL_SUPPORT_ONLY")
            aba_used+=1

    roots=defaultdict(list)
    for i in range(len(uids)): roots[dsu.find(i)].append(i)
    ordered=sorted(roots,key=lambda r:min(uids[i] for i in roots[r]))
    pid_root={r:f"PTRK_R4V2_{n:08d}" for n,r in enumerate(ordered,1)}
    pid_idx={i:pid_root[dsu.find(i)] for i in range(len(uids))}
    pid_uid={u:pid_idx[i] for u,i in uid_idx.items()}
    track_comp=defaultdict(set)
    for u,cid in uid_comp.items(): track_comp[pid_uid[u]].add(cid)
    track_flags=defaultdict(set)
    for u,sources in base_src.items():
        for source in sources.split(";"):
            if source: track_flags[pid_uid[u]].update(src_flags(source,{"source_object_id":u}))
    for k,flags in edge_flags.items():
        for u in k[2:4]:
            if u in pid_uid: track_flags[pid_uid[u]].update(flags)
    for pid,cids in track_comp.items():
        for cid in cids:
            track_flags[pid].add("V4B_STRUCTURAL_CLOSURE")
            st=text(comp_meta[cid]["row"].get("structural_types")).upper()
            if "SPLIT_MERGE" in st: track_flags[pid].add("COMPOUND_SPLIT_MERGE")
            elif "SPLIT" in st: track_flags[pid].add("COMPOUND_SPLIT")
            elif "MERGE" in st: track_flags[pid].add("COMPOUND_MERGE")
            if comp_meta[cid]["status"]=="RESOLVED_GAP_RETURN": track_flags[pid].add("GAP_RETURN")

    evidence=defaultdict(dict)
    for r in read_rows(formal/"05_presence_state"/"PRESENCE_1_0_U.csv"):
        if text(r.get("building_id")) and text(r.get("scene_key")): evidence[text(r["building_id"])][text(r["scene_key"])]=r
    comp_nodes={}; comp_area={}
    for cid,m in comp_meta.items():
        for scene in SCENES:
            comp_nodes[(cid,scene)]={u for u in m["nodes"] if scene_arr[uid_idx[u]]==scene}; comp_area[(cid,scene)]=sf(dct(m["row"].get("scene_union_area_m2")).get(scene))

    presence=[]; tracks=[]; pinfo={}
    for root in ordered:
        idxs=roots[root]; pid=pid_root[root]; by_scene={s:[i for i in idxs if scene_arr[i]==s] for s in SCENES}; cids=sorted(track_comp.get(pid,set())); original=sorted(set(building_arr[i] for i in idxs)); info={}
        for scene in SCENES:
            ids=by_scene[scene]; covered=set().union(*(comp_nodes.get((cid,scene),set()) for cid in cids)) if cids else set()
            area=sum(area_arr[i] for i in ids if uids[i] not in covered)+sum(comp_area.get((cid,scene),0.0) for cid in cids if comp_nodes.get((cid,scene)))
            if ids: value,reason,uid="1","POSITIVE_CANONICAL_OBJECT_SUPPORT",uids[ids[0]]
            elif any(comp_meta[cid]["status"]=="RESOLVED_GAP_RETURN" for cid in cids) and scene=="xa190101": value,reason,uid="U","GAP_RETURN_MIDDLE_OBSERVABILITY",""
            else:
                evs=[evidence.get(b,{}).get(scene) for b in original if evidence.get(b,{}).get(scene)]
                zero=[e for e in evs if text(e.get("presence"))=="0" and text(e.get("presence_reason"))=="RELIABLE_CLEAR_ABSENCE"]; unk=[e for e in evs if text(e.get("presence")) in {"U",""} or text(e.get("presence_reason"))=="UNOBSERVED_OR_UNCERTAIN"]
                value,reason,uid=("0","RELIABLE_CLEAR_ABSENCE_FROM_OBSERVABILITY_EVIDENCE","") if zero and not unk else ("U","UNOBSERVED_OR_UNCERTAIN_FROM_OBSERVABILITY_EVIDENCE","")
            info[scene]={"presence":value,"reason":reason,"object_uid":uid,"area_m2":area,"object_count":len(ids)}
            presence.append({"physical_track_id":pid,"canonical_object_id":uid,"scene_key":scene,"formal_date":DATES[scene],"presence":value,"presence_reason":reason,"candidate_object_uid":uid,"area_m2":area,"object_count":len(ids),"compound_track_id":";".join(cids),"identity_sources":";".join(sorted(track_flags.get(pid,set()))),"scientific_status":"CANDIDATE_NOT_GROUND_TRUTH"})
        pinfo[pid]=info; tracks.append({"physical_track_id":pid,"canonical_object_count":len(idxs),"scene_count":sum(bool(by_scene[s]) for s in SCENES),"object_count_2018":len(by_scene["xa180429"]),"object_count_2019":len(by_scene["xa190101"]),"object_count_2020":len(by_scene["xa201231"]),"presence_sequence":"-".join(info[s]["presence"] for s in SCENES),"compound_track_id":";".join(cids),"identity_sources":";".join(sorted(track_flags.get(pid,set()))),"scientific_status":"CANDIDATE_NOT_GROUND_TRUTH"})
    write_csv(out/"PRESENCE_V2_FROM_SCRATCH.csv",presence,list(presence[0])); write_csv(out/"FINAL_CANDIDATE_PHYSICAL_TRACKS_V2.csv",tracks,list(tracks[0]))

    track_vstate=defaultdict(lambda:defaultdict(set))
    for pid,cids in track_comp.items():
        for cid in cids:
            for interval in ("2018_2019","2019_2020"):
                if (cid,interval) in comp_state: track_vstate[pid][interval].add(comp_state[(cid,interval)])
    states=[]; counts={t:Counter() for t,_,_ in TRANSITIONS}; areas={t:Counter() for t,_,_ in TRANSITIONS}; den={t:0.0 for t,_,_ in TRANSITIONS}; add_audit=[]; demol_audit=[]
    for tr in tracks:
        pid=tr["physical_track_id"]; info=pinfo[pid]; seq=tr["presence_sequence"]; sources=set(text(tr["identity_sources"]).split(";"))-{""}
        for interval,ss,tt in TRANSITIONS:
            a1,b1=info[ss]["presence"],info[tt]["presence"]; sa,ta=info[ss]["area_m2"],info[tt]["area_m2"]; union=max(sa,ta); den[interval]+=union; vs=track_vstate.get(pid,{}).get(interval,set()); vstate=sorted(vs)[0] if len(vs)==1 else ""
            if seq=="1-U-1": state,event,eligible,reason="STABLE","GAP_RETURN","NO","ENDPOINT_CONTINUITY_1_U_1"
            elif a1=="1" and b1=="1": state,event,eligible,reason="STABLE","STABLE","YES","RELIABLE_PRESENT_BOTH_EPOCHS"
            elif a1=="0" and b1=="1": state,event,eligible,reason="ADDITION","ADDITION","YES","RELIABLE_0_TO_1"
            elif a1=="1" and b1=="0": state,event,eligible,reason="DEMOLITION","DEMOLITION","YES","RELIABLE_1_TO_0"
            else: state,event,eligible,reason="UNKNOWN","UNKNOWN","NO","UNRESOLVED_IDENTITY_OBSERVABILITY_OR_PARALLAX"
            if seq in {"1-0-1","0-1-0"} and state in {"ADDITION","DEMOLITION"}:
                if vstate in {"STABLE","ADDITION","DEMOLITION"}: reason="V4B_RESOLVED_COMPONENT_INTERVAL_STATE"
                else: state,event,eligible,reason="UNKNOWN","REBUILD_OR_FLICKER_REVIEW","NO","UNRESOLVED_REBUILD_OR_FLICKER_WITHOUT_INTERVAL_EVIDENCE"
            if seq=="1-U-1" and vstate=="STABLE": state,event,eligible,reason="STABLE","GAP_RETURN","NO","V4B_GAP_RETURN_COMPONENT_CONTINUITY"
            counts[interval][state]+=1; value=ta if state in {"STABLE","ADDITION"} else sa if state=="DEMOLITION" else union; areas[interval][state]+=value
            states.append({"physical_track_id":pid,"interval":interval,"presence_sequence":seq,"source_presence":a1,"target_presence":b1,"state":state,"event":event,"strict_change_eligible":eligible,"state_reason":reason,"source_area_m2":sa,"target_area_m2":ta,"union_area_m2":union,"area_m2":value,"gap_return":"YES" if event=="GAP_RETURN" else "NO","v4b_interval_state":vstate,"identity_sources":";".join(sorted(sources)),"scientific_status":"CANDIDATE_NOT_GROUND_TRUTH"})
            if interval=="2019_2020" and state=="ADDITION": add_audit.append({"physical_track_id":pid,"interval":interval,"area_m2":value,"presence_sequence":seq,"reason_class":"A_TRACK_BREAK" if seq in {"1-0-1","0-1-0"} else ("A_PARALLAX_RELOCATION" if "PARALLAX_SAFE_RECOVERY" in sources else ("A_SPLIT_FRAGMENT" if any("COMPOUND" in x or "V4B" in x for x in sources) else "A_TRUE_RELIABLE_0_TO_1")),"identity_sources":";".join(sorted(sources))})
            if interval=="2018_2019" and state=="DEMOLITION": demol_audit.append({"physical_track_id":pid,"interval":interval,"area_m2":value,"presence_sequence":seq,"reason_class":"D_TRACK_BREAK" if seq in {"1-0-1","0-1-0"} else ("D_PARALLAX" if "PARALLAX_SAFE_RECOVERY" in sources else ("D_SPLIT_MERGE" if any("COMPOUND" in x or "V4B" in x for x in sources) else "D_TRUE_1_TO_0")),"identity_sources":";".join(sorted(sources))})
    write_csv(out/"STATE_V2_FROM_SCRATCH.csv",states,list(states[0])); write_csv(out/"ADDITION_2019_2020_REASON_AUDIT.csv",add_audit,list(add_audit[0]) if add_audit else ["physical_track_id"]); write_csv(out/"DEMOLITION_2018_2019_REASON_AUDIT.csv",demol_audit,list(demol_audit[0]) if demol_audit else ["physical_track_id"])

    residual=[]
    for r in states:
        if r["state"]!="UNKNOWN": continue
        ss=set(text(r["identity_sources"]).split(";"))-{""}
        cause="U03_PARALLAX" if "PARALLAX_SAFE_RECOVERY" in ss else ("U06_GAP" if r["presence_sequence"]=="1-U-1" else ("U07_COMPOUND_UNRESOLVED" if any("COMPOUND" in x or "V4B" in x for x in ss) else ("U05_UNOBSERVABLE" if r["source_presence"]=="U" or r["target_presence"]=="U" else ("U01_IDENTITY_MULTIPLE_OR_NO_MATCH" if not ss else "U08_INSUFFICIENT_EVIDENCE"))))
        residual.append({**r,"unknown_cause":cause,"phase":"RESIDUAL_UNKNOWN_PHASE_B","resolution_status":"RETAINED_UNKNOWN"})
    write_csv(out/"RESIDUAL_UNKNOWN_PHASE_B_V2.csv",residual,list(residual[0]) if residual else ["physical_track_id"])

    ledger=[]
    for n,k in enumerate(sorted(set(edge_records)|set(strict_by_key)|set(held_dec)),1):
        e=dict(edge_records.get(k) or strict_by_key.get(k) or next((x for x in held if key(x)==k),{})); flags=set(edge_flags.get(k,set())); flags.update({"STRICT_CORE"} if k in strict_by_key else set())
        if held_dec.get(k)=="SAFE_RECOVERY": flags.add("SAFE_RECOVERY")
        if held_dec.get(k)=="PARALLAX_SAFE": flags.add("PARALLAX_SAFE_RECOVERY")
        ledger.append({"edge_id":f"E_V2_{n:09d}","source_epoch":text(e.get("from_scene")),"target_epoch":text(e.get("to_scene")),"source_object_id":text(e.get("source_object_id")),"target_object_id":text(e.get("target_object_id")),"relation_type":text(e.get("candidate_relation")) or text(e.get("relation")),"primary_source":source_priority(flags),"source_flags":";".join(sorted(flags)),"is_strict_core":"YES" if "STRICT_CORE" in flags else "NO","is_v4b":"YES" if "V4B_STRUCTURAL_CLOSURE" in flags else "NO","is_aba":"YES" if "ABA_SAFE_CLOSURE" in flags else "NO","is_compound":"YES" if any(x in flags for x in {"COMPOUND_SPLIT","COMPOUND_MERGE","COMPOUND_SPLIT_MERGE"}) else "NO","is_gap_return":"YES" if "GAP_RETURN" in flags else "NO","is_safe_recovery":"YES" if "SAFE_RECOVERY" in flags else "NO","is_parallax_safe_recovery":"YES" if "PARALLAX_SAFE_RECOVERY" in flags else "NO","is_direct_support":"YES" if text(e.get("from_scene"))=="xa180429" and text(e.get("to_scene"))=="xa201231" else "NO","score":sf(e.get("score")),"buffered_iou":sf(e.get("buffered_iou")),"area_ratio":sf(e.get("area_ratio")),"displacement":sf(e.get("alignment_aware_displacement_m")),"neighborhood_score":sf(e.get("local_neighborhood_consistency")),"source_margin":sf(e.get("score_gap_source")),"target_margin":sf(e.get("score_gap_target")),"candidate_relation":text(e.get("candidate_relation")),"split_merge_type":text(e.get("split_merge_type")),"identity_conflict":"YES" if sb(e.get("identity_conflict")) else "NO","identity_support_eligible":"YES","formal_transition_eligible":"YES" if transition_ok(e) else "NO","reason":";".join(edge_reasons.get(k,[]))})
    write_csv(out/"FINAL_IDENTITY_SOURCE_LEDGER_V2.csv",ledger,list(ledger[0]))

    strict_ledger={key(r) for r in ledger if r["is_strict_core"]=="YES"}; missing=sorted(set(strict_by_key)-strict_ledger)
    recon=[{"input_strict_edges":len(strict_rows),"input_strict_unique_keys":len(strict_by_key),"ledger_strict_unique_keys":len(strict_ledger),"strict_occurrence_rows":sum(strict_occ.values()),"relabelled_rows":0,"missing_rows":len(missing),"missing_explained":"YES" if not missing else "NO"}]
    write_csv(out/"STRICT_CORE_LEDGER_RECONCILIATION_V2.csv",recon,list(recon[0])); write_csv(out/"STRICT_CORE_LEDGER_MISSING_V2.csv",[{"from_scene":x[0],"to_scene":x[1],"source_object_id":x[2],"target_object_id":x[3]} for x in missing],["from_scene","to_scene","source_object_id","target_object_id"])

    graph=[]
    for i,r in obj.iterrows():
        pid=pid_idx[i]; graph.append({"record_type":"NODE","edge_stage":"CANONICAL_OBJECT","object_uid":r["object_uid"],"scene":r["scene_key"],"formal_date":r["formal_date"],"current_building_id":r["building_id"],"canonical_object_id":r["object_uid"],"physical_track_id":pid,"compound_track_id":";".join(sorted(track_comp.get(pid,set()))),"area_m2":r["area_m2"],"candidate_source":";".join(sorted(track_flags.get(pid,set()))),"primary_source":source_priority(track_flags.get(pid,set()))})
    for r in ledger:
        graph.append({"record_type":"EDGE","edge_id":r["edge_id"],"edge_stage":"V2_SOURCE_LEDGER","source_object_id":r["source_object_id"],"target_object_id":r["target_object_id"],"from_scene":r["source_epoch"],"to_scene":r["target_epoch"],"physical_track_id":pid_uid.get(r["source_object_id"],""),"candidate_source":r["source_flags"],"primary_source":r["primary_source"],"transition_eligible":r["formal_transition_eligible"],"identity_support_eligible":r["identity_support_eligible"],"score":r["score"],"buffered_iou":r["buffered_iou"],"area_ratio":r["area_ratio"],"alignment_aware_displacement_m":r["displacement"],"local_neighborhood_consistency":r["neighborhood_score"],"candidate_relation":r["candidate_relation"],"split_merge_type":r["split_merge_type"],"identity_conflict":r["identity_conflict"]})
    gdf=pd.DataFrame(graph); pq.write_table(pa.Table.from_pandas(gdf,preserve_index=False),out/"FINAL_CANDIDATE_IDENTITY_GRAPH_V2.parquet",compression="zstd"); gdf.to_csv(out/"FINAL_CANDIDATE_IDENTITY_GRAPH_V2.csv",index=False,encoding="utf-8-sig")

    out_obj=obj.copy(); out_obj["canonical_object_id"]=out_obj.object_uid; out_obj["physical_track_id"]=[pid_idx[i] for i in range(len(obj))]; out_obj["compound_track_id"]=[";".join(sorted(track_comp.get(pid_idx[i],set()))) for i in range(len(obj))]; out_obj["identity_sources"]=[";".join(sorted(track_flags.get(pid_idx[i],set()))) for i in range(len(obj))]; out_obj["scientific_status"]="CANDIDATE_NOT_GROUND_TRUTH"; out_obj.to_csv(out/"FINAL_CANDIDATE_TRACKING_OBJECTS_V2.csv",index=False,encoding="utf-8-sig")

    state_df=pd.DataFrame(states); qa=[]
    for st in ("STABLE","ADDITION","DEMOLITION","UNKNOWN"):
        for n,r in enumerate(state_df[state_df.state.eq(st)].sort_values(["physical_track_id","interval"]).head(100).to_dict("records"),1):
            qa.append({"sample_id":f"{st}_QA_{n:03d}","category":st,"physical_track_id":r["physical_track_id"],"interval":r["interval"],"presence_sequence":r["presence_sequence"],"visual_label":"NOT_REVIEWED","visual_review_status":"PENDING_ASSET_GENERATION","raw_asset_status":"INDEX_ONLY","scientific_status":"CANDIDATE_NOT_GROUND_TRUTH"})
    write_csv(out/"VISUAL_QA_SAMPLE_INDEX_V2.csv",qa,list(qa[0]) if qa else ["sample_id"])

    v4b_reg=[]
    state_map={(r["physical_track_id"],r["interval"]):r["state"] for r in states}
    for cid in sorted(resolved):
        pids=sorted({pid_uid[u] for u in comp_meta[cid]["nodes"] if u in pid_uid})
        for interval in ("2018_2019","2019_2020"):
            expected=comp_state.get((cid,interval))
            if not expected: continue
            observed=sorted({state_map.get((pid,interval),"MISSING") for pid in pids})
            if len(pids)!=1 or not observed or "UNKNOWN" in observed or "MISSING" in observed:
                v4b_reg.append({"component_id":cid,"interval":interval,"expected_state":expected,"observed_states":";".join(observed),"physical_track_count":len(pids),"reason":"R03_COMPOUND_TRACK_NOT_CONSUMED_BY_PRESENCE" if len(pids)==1 else "R02_COMPONENT_NOT_PROMOTED"})
    write_csv(out/"V4B_REGRESSION_V2.csv",v4b_reg,list(v4b_reg[0]) if v4b_reg else ["component_id"])

    area_pct={}; unknown_pct={}
    for interval,_,_ in TRANSITIONS:
        total=den[interval]; area_pct[interval]={s:100*areas[interval][s]/total if total else 0 for s in ("STABLE","ADDITION","DEMOLITION","UNKNOWN")}; unknown_pct[interval]=area_pct[interval]["UNKNOWN"]
    held_counts=Counter(r["decision"] for r in held_out); stable_ok=all(area_pct[i]["STABLE"]>=area_pct[i]["ADDITION"]+area_pct[i]["DEMOLITION"] for i,_,_ in TRANSITIONS)
    receipt={"task":"XIAN-V4B-REGRESSION-HELD-RECOVERY-AND-FINAL-STATE-CLOSURE-001","canonical_objects":len(obj),"physical_tracks":len(tracks),"strict_core_input_rows":len(strict_rows),"strict_core_unique_keys":len(strict_by_key),"v4b_promoted_components":len(resolved),"aba_rows_used":aba_used,"held_edges":len(held),"held_full_recovery_decisions":dict(held_counts),"safe_recovery_unique_edges":sum(x=="SAFE_RECOVERY" for x in held_dec.values()),"parallax_safe_unique_edges":sum(x=="PARALLAX_SAFE" for x in held_dec.values()),"ledger_reconciliation_gate":"PASS" if not missing else "FAIL","v4b_regression_gate":"PASS" if not v4b_reg else "FAIL","v4b_regression_rows":len(v4b_reg),"unknown_area_pct":unknown_pct,"unknown_lt5_gate":"PASS" if all(x<5 for x in unknown_pct.values()) else "FAIL","state_area_pct":area_pct,"state_sanity_gate":"PASS" if stable_ok else "FAIL","tracking_accuracy_gate":"REVIEW_REQUIRED","fixed_case_regression":"REVIEW_REQUIRED","false_change_visual_gate":"NOT_RUN","full_city_grid_gate":"NOT_RUN","final_product_gate":"NOT_RUN","ready_for_final_production":"NO","residual_unknown_rows":len(residual),"formal_outputs_modified":False,"geometry_owner_formal_unchanged":True,"scientific_status":"CANDIDATE_NOT_GROUND_TRUTH","next_action":"GENERATE_AND_REVIEW_REAL_VISUAL_QA_ASSETS"}
    write_json(out/"FINAL_CANDIDATE_RECEIPT_V2.json",receipt)
    (out/"FINAL_CANDIDATE_REPORT_V2.md").write_text("# Xi'an Candidate IdentityGraph V2\n\n状态：REVIEW_REQUIRED\n\n本轮从 candidate-v1 identity baseline 加入 full held recovery，按 V4B 已解决 interval 兼容 State decoder，并从新 physical track 重算 Presence/State。正式 Geometry、Tracking、Presence、State、TIF 未修改。\n\n详细 Gate、来源账本和逐对象明细见 FINAL_CANDIDATE_RECEIPT_V2.json、FINAL_IDENTITY_SOURCE_LEDGER_V2.csv、HELD_FULL_RECOVERY_DECISION.csv 与 V4B_REGRESSION_V2.csv。\n",encoding="utf-8")
    progress(out,"COMPLETE",**receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
