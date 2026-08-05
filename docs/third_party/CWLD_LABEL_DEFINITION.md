# CWLD label definition

Source: *A construction waste landfill dataset of two districts in Beijing, China from high resolution satellite images* (Scientific Data), together with the official CWLD_Model repository.

| RGB anchor | Official class | Candidate ontology output |
|---|---|---|
| 0,0,0 | background | background |
| 255,255,255 | vacant landfillable area | vacant_land_candidate |
| 0,0,255 | engineering facility area | engineering_facility_candidate |
| 255,0,0 | waste dumping area | construction_waste |

Engineering facility is explicitly **not** construction-site ground truth. RGBA edge pixels are audited by anchor distance and may be ignored rather than forced into background.
