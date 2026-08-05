(() => {
  const $ = (id) => document.getElementById(id);
  const seed = { inventory: { records: [] }, state: { decisions: {} }, rows: [], index: 0 };
  const states = [
    'stable_building', 'possible_demolition', 'cleared_or_exposed_ground',
    'possible_active_construction', 'possible_structure_rising',
    'possible_completed_building', 'possible_repair_or_refurbishment',
    'ordinary_bare_land', 'waste_or_material_storage', 'uncertain'
  ];
  const transitions = [
    'NO_VISIBLE_CHANGE', 'DEMOLITION_OR_CLEARING', 'CONSTRUCTION_START',
    'CONTINUING_CONSTRUCTION', 'STRUCTURE_APPEARANCE', 'COMPLETION',
    'REPAIR_OR_REFURBISHMENT_CANDIDATE', 'LIKELY_FALSE_CHANGE', 'UNCERTAIN'
  ];
  const decisions = ['ACCEPT', 'CORRECT', 'UNCERTAIN', 'REJECT'];
  const esc = (value) => String(value ?? '').replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[character]));
  const seedAsset = (path) => path ? `/construction-preview/${encodeURIComponent(String(path).replaceAll('\\', '/'))}` : '';
  const get = async (url) => {
    const response = await fetch(url, { cache: 'no-store' });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `GET ${url} failed`);
    return body;
  };
  const post = async (url, payload) => {
    const response = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `POST ${url} failed`);
    return body;
  };
  const isTrue = (value) => value === true || value === 'true' || value === 1 || value === '1';
  const options = (values, current) => `<option value="">Choose</option>${values.map((value) => `<option value="${value}" ${value === current ? 'selected' : ''}>${value}</option>`).join('')}`;
  const linkFor = (path) => `<a href="${seedAsset(path)}" target="_blank" rel="noopener">${esc(String(path).replaceAll('\\', '/').split('/').pop())}</a>`;
  const assetItems = (paths) => (Array.isArray(paths) ? paths : []).map((path) => {
    const lower = String(path).toLowerCase();
    if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
      return `<figure class="seed-evidence-figure"><img src="${seedAsset(path)}" alt="existing audited evidence"><figcaption>${linkFor(path)}</figcaption></figure>`;
    }
    return `<div class="source seed-asset-link">${linkFor(path)}</div>`;
  }).join('');
  const setSelect = (id, values, current) => { $(id).innerHTML = options(values, current); };
  const setFlag = (id, value) => { $(id).checked = isTrue(value); };

  function rowsForDisplay() {
    const all = seed.inventory.records || [];
    if (!$('seedPendingOnly').checked) return all;
    return all.filter((row) => !seed.state.decisions[row.sample_id]);
  }

  function renderEvidence(row) {
    const evidence = row.evidence || {};
    const geometry = evidence.building_geometry || {};
    const full20 = evidence.full20 || {};
    let html = '';
    if (geometry.status === 'available') {
      html += `<div class="evidence-card"><h3>RSBuilding evidence: before / after probabilities, masks, boundaries, objects</h3><div class="seed-evidence-assets">${assetItems(geometry.paths)}</div></div>`;
    } else {
      html += '<div class="evidence-card"><h3>RSBuilding evidence</h3><p class="help">unavailable in this audited sample</p></div>';
    }
    if (full20.status === 'available' && Array.isArray(full20.paths) && full20.paths.length) {
      html += `<div class="evidence-card"><h3>Full20 (if available)</h3><div class="seed-evidence-assets">${assetItems(full20.paths)}</div></div>`;
    }
    $('seedEvidence').innerHTML = html;
  }

  function renderSources(row) {
    const before = row.source_metadata?.before || {};
    const after = row.source_metadata?.after || {};
    const line = (name, value) => `<div class="source"><b>${esc(name)}</b><br>${esc(value)}</div>`;
    $('seedSources').innerHTML = [
      line('dates', `${row.before_date || ''} -> ${row.after_date || ''}`),
      line('before source', before.source_path || row.raw_before_path || ''),
      line('after source', after.source_path || row.raw_after_path || ''),
      line('before CRS', before.source_crs || 'unavailable'),
      line('after CRS', after.source_crs || 'unavailable'),
      line('before window bounds', JSON.stringify(before.source_window || null)),
      line('after window bounds', JSON.stringify(after.source_window || null)),
      line('label source', seed.state.decisions[row.sample_id]?.label_source || row.label_source || 'human_seed_label'),
      line('formal ground truth', 'false'),
      line('model independent', 'true'),
      line('construction model display', 'suppressed')
    ].join('');
  }

  function renderForm(row) {
    const decision = seed.state.decisions[row.sample_id] || {};
    setSelect('seedBeforeSurfaceState', states, decision.before_surface_state || '');
    setSelect('seedAfterSurfaceState', states, decision.after_surface_state || '');
    setSelect('seedTransitionType', transitions, decision.transition_type || '');
    setSelect('seedDecision', decisions, decision.decision || '');
    setFlag('seedExistingBuildingBefore', decision.existing_building_before);
    setFlag('seedExistingBuildingAfter', decision.existing_building_after);
    setFlag('seedConstructionActivityVisible', decision.construction_activity_visible);
    setFlag('seedDemolitionVisible', decision.demolition_visible);
    setFlag('seedClearingVisible', decision.clearing_visible);
    setFlag('seedNewStructureVisible', decision.new_structure_visible);
    setFlag('seedRepairVisible', decision.repair_or_refurbishment_visible);
    $('seedConfidence').value = decision.confidence ?? 0.5;
    $('seedConfidenceValue').textContent = Number($('seedConfidence').value).toFixed(2);
    $('seedNotes').value = decision.notes || '';
    $('seedReviewer').value = decision.reviewer || '';
    $('seedHumanConfirmed').checked = isTrue(decision.human_confirmed);
    const disabled = !row;
    ['seedBeforeSurfaceState', 'seedAfterSurfaceState', 'seedTransitionType', 'seedDecision', 'seedConfidence', 'seedNotes', 'seedReviewer', 'seedHumanConfirmed', 'seedSave'].forEach((id) => { $(id).disabled = disabled; });
  }

  function render() {
    seed.rows = rowsForDisplay();
    if (!seed.rows.length) {
      $('seedSampleId').textContent = 'No seed records match the current filter';
      $('seedMeta').textContent = '';
      ['seedPairPreview', 'seedBeforePreview', 'seedAfterPreview'].forEach((id) => $(id).removeAttribute('src'));
      $('seedEvidence').innerHTML = '';
      $('seedSources').innerHTML = '';
      renderForm(null);
      $('seedStats').innerHTML = `total: ${seed.inventory.seed_count || 0}<br>labeled: ${Object.keys(seed.state.decisions || {}).length}<br>remaining: ${Math.max(0, (seed.inventory.seed_count || 0) - Object.keys(seed.state.decisions || {}).length)}<br>status: ${esc(seed.stateSummary?.coverage_status || seed.inventory.coverage_status || 'PENDING HUMAN SEED LABELS')}`;
      return;
    }
    seed.index = Math.min(seed.index, seed.rows.length - 1);
    const row = seed.rows[seed.index];
    $('seedSampleId').textContent = row.sample_id;
    $('seedMeta').textContent = `${row.track_seed_id} | ${row.before_date} -> ${row.after_date} | preview: ${row.preview_status || 'unknown'}`;
    $('seedPairPreview').src = seedAsset(row.pair_preview_path);
    $('seedBeforePreview').src = seedAsset(row.raw_before_path);
    $('seedAfterPreview').src = seedAsset(row.raw_after_path);
    renderEvidence(row);
    renderSources(row);
    renderForm(row);
    $('seedStats').innerHTML = `displayed: ${seed.rows.length}<br>total: ${seed.inventory.seed_count || 0}<br>labeled: ${Object.keys(seed.state.decisions || {}).length}<br>verified seeds: ${seed.stateSummary?.verified_seed_count || 0}<br>status: ${esc(seed.stateSummary?.coverage_status || seed.inventory.coverage_status || 'PENDING HUMAN SEED LABELS')}`;
  }

  async function load() {
    try {
      seed.inventory = await get('/api/seed-temporal/inventory');
      const stateResponse = await get('/api/seed-temporal/state');
      seed.state = stateResponse.state || { decisions: {} };
      seed.stateSummary = stateResponse.summary || {};
      seed.index = 0;
      render();
    } catch (error) {
      $('seedMessage').textContent = `Seed Temporal unavailable: ${error.message}`;
      $('seedSave').disabled = true;
    }
  }

  function payload() {
    const row = seed.rows[seed.index];
    return {
      sample_id: row.sample_id,
      inventory_hash: seed.inventory.inventory_hash,
      before_surface_state: $('seedBeforeSurfaceState').value,
      after_surface_state: $('seedAfterSurfaceState').value,
      transition_type: $('seedTransitionType').value,
      existing_building_before: $('seedExistingBuildingBefore').checked,
      existing_building_after: $('seedExistingBuildingAfter').checked,
      construction_activity_visible: $('seedConstructionActivityVisible').checked,
      demolition_visible: $('seedDemolitionVisible').checked,
      clearing_visible: $('seedClearingVisible').checked,
      new_structure_visible: $('seedNewStructureVisible').checked,
      repair_or_refurbishment_visible: $('seedRepairVisible').checked,
      confidence: Number($('seedConfidence').value),
      notes: $('seedNotes').value,
      reviewer: $('seedReviewer').value,
      decision: $('seedDecision').value,
      human_confirmed: $('seedHumanConfirmed').checked
    };
  }

  async function save() {
    if (!seed.rows.length) return;
    const body = payload();
    if (!body.before_surface_state || !body.after_surface_state || !body.transition_type || !body.decision || !body.reviewer.trim()) {
      $('seedMessage').textContent = 'Choose both surface states, a transition, a decision, and a reviewer.';
      return;
    }
    try {
      const result = await post('/api/seed-temporal/review', body);
      $('seedMessage').textContent = `Saved ${result.decision.label_source}; formal_ground_truth=false`;
      await load();
    } catch (error) {
      $('seedMessage').textContent = `Save failed: ${error.message}`;
    }
  }

  $('seedSave').onclick = save;
  $('seedExport').onclick = async () => {
    try {
      const result = await post('/api/seed-temporal/export', {});
      $('seedMessage').textContent = `Export receipt ready for ${result.rows} seed labels.`;
    } catch (error) {
      $('seedMessage').textContent = `Export failed: ${error.message}`;
    }
  };
  $('seedPrev').onclick = () => { seed.index = Math.max(0, seed.index - 1); render(); };
  $('seedNext').onclick = () => { seed.index = Math.min(seed.rows.length - 1, seed.index + 1); render(); };
  $('seedPendingOnly').onchange = () => { seed.index = 0; render(); };
  $('seedConfidence').oninput = () => { $('seedConfidenceValue').textContent = Number($('seedConfidence').value).toFixed(2); };
  load();
})();
