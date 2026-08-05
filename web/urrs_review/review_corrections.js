document.getElementById('constructionCorrections').onclick=async()=>{
  const response=await fetch('/api/construction/corrections',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  const result=await response.json();
  document.getElementById('constructionMessage').textContent=result.error||`已导出 ${result.feature_count} 个候选到 QGIS`;
};
