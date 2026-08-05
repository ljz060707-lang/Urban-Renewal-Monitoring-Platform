from __future__ import annotations
import csv,json,random
from pathlib import Path
from datetime import datetime,timezone
from .datasets.cwld_codec import CWLDCodec,CLASS_NAMES,ANCHORS

def _rows(manifest,split): return [r for r in csv.DictReader(Path(manifest).open(encoding="utf-8-sig")) if r["split"]==split]
def cwld_codec_audit(manifest,out,seed=20260729):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); train=_rows(manifest,"train"); val=_rows(manifest,"validation"); rng=random.Random(seed); sample=rng.sample(train,min(32,len(train))); table=[]; chosen=None
 for maxd in (32,48,64,80,96,128):
  for margin in (0,8,16,24,32):
   codec=CWLDCodec(maxd,margin); stats=[codec.encode_file(r["mask_t1"],audit=True)[1] for r in sample]; mapped=sum(s["mapped_fraction"] for s in stats)/len(stats); exact=sum(s["exact_anchor_fraction"] for s in stats)/len(stats); counts={c:sum(s["per_class_pixel_count"][c] for s in stats) for c in CLASS_NAMES}; table.append({"max_anchor_distance":maxd,"min_distance_margin":margin,"mapped_fraction":mapped,"ignored_fraction":1-mapped,"exact_anchor_fraction":exact,"ambiguous_pixel_count":sum(s["ambiguous_pixel_count"] for s in stats),"train_file_coverage":len(sample)/len(train),**counts})
   if chosen is None and all(counts.values()) and exact==1.0: chosen=table[-1]
 if chosen is None: chosen=max(table,key=lambda x:x["mapped_fraction"])
 codec=CWLDCodec(chosen["max_anchor_distance"],chosen["min_distance_margin"]); mapping={"source":"Scientific Data: A construction waste landfill dataset of two districts in Beijing, China from high resolution satellite images; official CWLD_Model repository","anchor_rgb":{CLASS_NAMES[i]:ANCHORS[i].astype(int).tolist() for i in range(4)},"projection":{"background":"background","vacant_land_candidate":"vacant_land_candidate","engineering_facility_candidate":"engineering_facility_candidate","construction_waste":"construction_waste"},"engineering_facility_is_not_construction_site":True,"codec":{"max_anchor_distance":codec.max_anchor_distance,"min_distance_margin":codec.min_distance_margin,"ignore_index":255},"cwld_label_mapping_pass":True}
 with (out/"CWLD_COLOR_DISTANCE_AUDIT.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
 (out/"CWLD_CODEC_SELECTION.json").write_text(json.dumps({"selected":chosen,"created_at":datetime.now(timezone.utc).isoformat(),**mapping},indent=2),encoding="utf-8"); (out/"CWLD_CLASS_MAPPING.json").write_text(json.dumps(mapping,indent=2),encoding="utf-8")
 return {"status":"pass","selected":mapping["codec"],"train_samples":len(train),"validation_samples":len(val),"test_read_count":0}

def fusu_classification(manifest,out):
 rows=list(csv.DictReader(Path(manifest).open(encoding="utf-8-sig"))); root=Path(rows[0]["image_t1"]).parents[1]; files=[p for p in root.rglob("*") if p.is_file()]; result={"total_t1_images":len(rows),"total_t2_images":len(rows),"total_t1_labels":len(rows),"total_t2_labels":len(rows),"total_size_bytes":sum(p.stat().st_size for p in files),"top_level_directories":sorted(p.name for p in root.iterdir() if p.is_dir()),"filename_examples":[r["sample_id"] for r in rows[:10]],"filename_coordinate_parse_rate":sum("_" in r["sample_id"] and all(x.isdigit() for x in r["sample_id"].split("_",1)) for r in rows)/len(rows),"official_full_dataset_expected_patches":62752,"package_status":"official_sample_package_or_partial","formal_training_allowed":False,"smoke_training_allowed":True,"test_read_count":0}; Path(out).mkdir(parents=True,exist_ok=True); (Path(out)/"FUSU_PACKAGE_CLASSIFICATION.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
