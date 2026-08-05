(() => {
  const $ = id => document.getElementById(id);
  const requestedProject = new URLSearchParams(window.location.search).get('project');
  const state = {project: requestedProject || 'nanjing_building', module: '', page: 1, items: [], index: -1, sample: null, drawing: false, pendingAction: null, points: []};
  const drawActions = new Set(['add', 'correct', 'split', 'merge', 'ignore']);

  async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok && data.error) throw new Error(data.error);
    return data;
  }

  function query() {
    const params = new URLSearchParams({project: state.project, page: state.page, page_size: 60});
    if (state.module) params.set('module', state.module);
    if ($('split').value) params.set('split', $('split').value);
    if ($('status').value) params.set('status', $('status').value);
    if ($('errorType').value) params.set('error_type', $('errorType').value);
    return params;
  }

  async function loadQueue() {
    const data = await api('/api/queue?' + query());
    state.items = data.items;
    $('queueCount').textContent = data.total;
    $('pageLabel').textContent = state.page;
    const modules = [...new Set(data.items.map(item => item.module))];
    const moduleSelect = $('module');
    const oldModule = moduleSelect.value;
    moduleSelect.innerHTML = '<option value="">全部模块</option>' + modules.map(m => `<option>${m}</option>`).join('');
    moduleSelect.value = modules.includes(oldModule) ? oldModule : '';
    $('queueList').innerHTML = data.items.map((item, index) => `<li class="queue-item ${index === state.index ? 'active' : ''}" data-index="${index}"><span class="tag">${item.split}</span><strong>${item.sample_id}</strong><small>${item.module} · ${item.scene_date} · ${item.status}</small></li>`).join('');
    $('queueList').querySelectorAll('.queue-item').forEach(item => item.onclick = () => selectSample(Number(item.dataset.index)));
    if (state.items.length) await selectSample(Math.max(0, Math.min(state.index, state.items.length - 1)));
    else clearSample();
  }

  async function selectSample(index) {
    state.index = index;
    const item = state.items[index];
    if (!item) return;
    state.drawing = false; state.pendingAction = null; state.points = [];
    state.sample = await api('/api/sample/' + encodeURIComponent(item.sample_id));
    render();
    preload(index + 1);
  }

  function preload(index) {
    const item = state.items[index];
    if (item && item.chip_path) { const image = new Image(); image.src = '/api/chip/' + encodeURIComponent(item.sample_id); }
  }

  function clearSample() { state.sample = null; $('sampleTitle').textContent = '没有匹配样本'; $('emptyState').style.display = 'block'; }

  function render() {
    const sample = state.sample;
    $('sampleTitle').textContent = sample.sample_id;
    $('saveState').textContent = sample.status === 'reviewed' ? '已审核' : '待审核';
    $('splitBadge').textContent = sample.split + (sample.blind_first ? ' · blind-first' : '');
    $('decision').value = sample.decision || '';
    $('error').value = sample.error_type || '';
    $('negative').value = sample.negative_type || '';
    $('notes').value = sample.notes || '';
    $('backgroundConfirm').checked = Boolean(sample.metadata && sample.metadata.complete_background_confirmed);
    $('metadata').innerHTML = `<div><b>模块</b> ${sample.module}</div><div><b>空间块</b> ${sample.spatial_block_id}</div><div><b>日期</b> ${sample.scene_date}</div><div><b>参考状态</b> <code>candidate_not_ground_truth</code></div><pre>${JSON.stringify(sample.reference || {}, null, 2)}</pre>`;
    const dateChips = (sample.metadata && sample.metadata.date_chips) || {};
    $('dateGrid').innerHTML = Object.keys(dateChips).map(date => `<figure><img src="/api/chip/${encodeURIComponent(sample.sample_id)}?date=${encodeURIComponent(date)}"><figcaption>${date}</figcaption></figure>`).join('');
    const image = $('chip'); image.src = '/api/chip/' + encodeURIComponent(sample.sample_id); image.style.display = sample.chip_path ? 'block' : 'none';
    $('emptyState').style.display = sample.chip_path ? 'none' : 'block';
    image.onload = resizeCanvas;
    draw();
  }

  function extent(geometry) {
    const points = [];
    (function walk(value) {
      if (Array.isArray(value) && typeof value[0] === 'number') points.push(value);
      else if (Array.isArray(value)) value.forEach(walk);
    })(geometry && geometry.coordinates);
    if (!points.length) return [0, 0, 1, 1];
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }

  function geometryFromPoints() {
    if (!state.sample || state.points.length < 3 || !$('overlay').width) return null;
    const base = state.sample.geometry || state.sample.source_geometry;
    const raw = extent(base), spanX = raw[2] - raw[0] || 0.0002, spanY = raw[3] - raw[1] || 0.0002;
    const b = [raw[0] - (raw[2] === raw[0] ? spanX : 0), raw[1] - (raw[3] === raw[1] ? spanY : 0), raw[2] + (raw[2] === raw[0] ? spanX : 0), raw[3] + (raw[3] === raw[1] ? spanY : 0)], canvas = $('overlay');
    const coords = state.points.map(point => [b[0] + point[0] / canvas.width * (b[2] - b[0]), b[3] - point[1] / canvas.height * (b[3] - b[1])]);
    coords.push(coords[0]);
    return {type: 'Polygon', coordinates: [coords]};
  }

  function draw() {
    const canvas = $('overlay'), context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    const sample = state.sample;
    if (!sample) return;
    const geometry = sample.geometry || sample.source_geometry;
    if (geometry && geometry.coordinates) {
      const rings = geometry.type === 'Polygon' ? geometry.coordinates : [geometry.coordinates];
      const b = extent(geometry), points = rings[0].map(point => [(point[0] - b[0]) / (b[2] - b[0] || 1) * canvas.width, (b[3] - point[1]) / (b[3] - b[1] || 1) * canvas.height]);
      context.strokeStyle = $('showHuman').checked ? '#79dc9a' : '#49d6d1';
      context.globalAlpha = Number($('opacity').value) / 100; context.lineWidth = 2; context.beginPath();
      points.forEach((point, index) => index ? context.lineTo(...point) : context.moveTo(...point)); context.closePath(); context.stroke(); context.globalAlpha = 1;
    }
    if (state.points.length) { context.strokeStyle = '#6ca8ff'; context.lineWidth = 2; context.beginPath(); state.points.forEach((point, index) => index ? context.lineTo(...point) : context.moveTo(...point)); context.stroke(); }
  }

  function resizeCanvas() { const canvas = $('overlay'), image = $('chip'); canvas.width = image.clientWidth; canvas.height = image.clientHeight; draw(); }

  function beginDrawing(action) {
    state.drawing = true; state.pendingAction = action; state.points = [];
    $('saveState').textContent = '请在影像上点击绘制，Ctrl+S 保存'; draw();
  }

  async function save(action, extra = {}) {
    if (!state.sample) return;
    const actualAction = action === 'save' && state.pendingAction ? state.pendingAction : action;
    const geometry = geometryFromPoints();
    const payload = {sample_id: state.sample.sample_id, action: actualAction, decision: $('decision').value, error_type: $('error').value || null, negative_type: $('negative').value || null, notes: $('notes').value || null, complete_background_confirmed: $('backgroundConfirm').checked, reviewer: 'local_reviewer', ...(geometry ? {geometry} : {}), ...extra};
    try {
      const result = await api('/api/review/save', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      state.drawing = false; state.pendingAction = null; state.points = [];
      $('saveState').textContent = '已保存 ' + new Date(result.saved_at).toLocaleTimeString();
      await loadQueue();
    } catch (error) { $('saveState').textContent = '保存失败: ' + error.message; }
  }

  function next(delta) { const index = state.index + delta; if (state.items.length && index >= 0 && index < state.items.length) selectSample(index); }

  async function refreshGate() {
    try {
      const data = await api('/api/training-gate?project=' + encodeURIComponent(state.project));
      const blocked = data.gates.filter(gate => gate.status !== 'pass').map(gate => `${gate.gate_id}: ${gate.status}`).join('<br>');
      $('gateBox').innerHTML = '<b>Training Gate</b><br>' + (blocked || '全部通过，可生成训练包') + `<br><small>Holdout: ${data.holdout_sealed ? 'sealed' : 'open'}</small>`;
    } catch (_) { /* the review page remains usable while the gate endpoint restarts */ }
  }

  $('project').onchange = () => { state.project = $('project').value; state.page = 1; state.index = -1; loadQueue(); refreshGate(); };
  ['module', 'split', 'status', 'errorType'].forEach(id => $(id).onchange = () => { state[id] = $(id).value; state.page = 1; state.index = -1; loadQueue(); });
  $('prevPage').onclick = () => { if (state.page > 1) { state.page--; state.index = -1; loadQueue(); } };
  $('nextPage').onclick = () => { state.page++; state.index = -1; loadQueue(); };
  $('prevBtn').onclick = () => next(-1); $('nextBtn').onclick = () => next(1);
  $('saveNext').onclick = () => save('save').then(() => next(1));
  document.querySelectorAll('[data-action]').forEach(button => button.onclick = () => drawActions.has(button.dataset.action) ? beginDrawing(button.dataset.action) : save(button.dataset.action));
  $('showModel').onchange = draw; $('showHuman').onchange = draw; $('opacity').oninput = draw; $('chip').onload = resizeCanvas; window.onresize = resizeCanvas;
  $('backupBtn').onclick = async () => { const result = await api('/api/backup', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'}); $('saveState').textContent = '备份完成 ' + result.backup_path; };
  $('exportBtn').onclick = async () => { const result = await api('/api/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({project: state.project})}); $('saveState').textContent = '导出完成 ' + result.export_path; };
  $('overlay').onclick = event => { if (!state.drawing) return; const rect = $('overlay').getBoundingClientRect(); state.points.push([event.clientX - rect.left, event.clientY - rect.top]); draw(); };
  document.addEventListener('keydown', event => {
    if (event.target.matches('input,textarea,select')) return;
    if (event.key === 'j' || event.key === 'J') next(-1); if (event.key === 'k' || event.key === 'K') next(1);
    if (event.ctrlKey && event.key.toLowerCase() === 's') { event.preventDefault(); save('save'); }
    if (event.key === ' ') { event.preventDefault(); $('showModel').checked = !$('showModel').checked; draw(); }
    if (event.key === 'v' || event.key === 'V') { $('showHuman').checked = !$('showHuman').checked; draw(); }
    if (event.key === 'i' || event.key === 'I') beginDrawing('ignore'); if (event.key === 'Delete') save('delete');
    if (event.key === '1') save('accept'); if (event.key === '2') beginDrawing('correct'); if (event.key === '3') beginDrawing('ignore'); if (event.key === '4') save('later');
    if (event.key === 'm' || event.key === 'M') beginDrawing('merge'); if (event.key === 'x' || event.key === 'X') beginDrawing('split'); if (event.key === 'c' || event.key === 'C') save('propagate'); if (event.key === 'y' || event.key === 'Y') save('copy_adjacent');
  });
  loadQueue(); refreshGate();
})();
