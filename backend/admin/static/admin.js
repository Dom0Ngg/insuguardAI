const state = { overview:null, claims:[], runs:[], tgSessions:[], selectedChatId:null, decisionThreshold:null };
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const fmtPct = (v) => v == null ? '—' : `${(Number(v)*100).toFixed(1)}%`;
const fmtDate = (v) => v ? new Date(v).toLocaleString('es-ES') : '—';
const pretty = (v) => JSON.stringify(v, null, 2);

function toast(text){ const el=$('#toast'); el.textContent=text; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),2600); }
async function api(url, options={}){ const r=await fetch(url,{headers:{'Content-Type':'application/json',...(options.headers||{})},...options}); if(!r.ok){ let msg=`HTTP ${r.status}`; try{const d=await r.json();msg=d.detail||msg}catch{} throw new Error(msg)} return r.json(); }
function closeEvidenceModal(){ const m=$('#evidence-modal'); if(!m)return; m.classList.add('hidden'); document.body.classList.remove('modal-open'); $('#evidence-modal-body').innerHTML=''; $('#evidence-modal-actions').innerHTML=''; }
function showEvidenceModal({eyebrow='EVIDENCIA',title='Detalle',meta='',body='',actions=''}){
  $('#evidence-modal-eyebrow').textContent=eyebrow; $('#evidence-modal-title').textContent=title;
  $('#evidence-modal-meta').innerHTML=meta; $('#evidence-modal-body').innerHTML=body; $('#evidence-modal-actions').innerHTML=actions;
  $('#evidence-modal').classList.remove('hidden'); document.body.classList.add('modal-open');
}
function viewerUrl(url,page){ return page ? `${url}#page=${Number(page)}` : url; }
function openDocumentViewer(title,url,page=null,meta=''){
  const target=viewerUrl(url,page);
  showEvidenceModal({eyebrow:'DOCUMENTO ORIGINAL',title,meta,body:`<iframe class="document-frame" src="${esc(target)}" title="${esc(title)}"></iframe>`,actions:`<a class="button secondary" href="${esc(target)}" target="_blank" rel="noopener">Abrir en pestaña nueva ↗</a><button class="button" data-close-modal>Cerrar</button>`});
  bindModalClose();
}
function bindModalClose(){ $$('[data-close-modal]').forEach(x=>x.onclick=closeEvidenceModal); }
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeEvidenceModal()});
function statusPill(text, kind='ok'){ return `<span class="status-pill ${kind}">${esc(text)}</span>`; }
function boolPill(value, yes='Sí', no='No'){ return value ? statusPill(yes,'error') : statusPill(no,'ok'); }
function switchView(name){
  $$('.nav').forEach(n=>n.classList.toggle('active',n.dataset.view===name));
  $$('.view').forEach(v=>v.classList.toggle('active',v.id===`view-${name}`));
  const titles={overview:'Centro de supervisión',claims:'Bandeja de expedientes',telegram:'Canal de entrada · Telegram',runs:'Auditoría técnica',mlops:'Monitorización MLOps',rag:'Asistente de pólizas · supervisor'};
  $('#page-title').textContent=titles[name]||name; loadView(name);
}
$$('.nav').forEach(n=>n.addEventListener('click',()=>switchView(n.dataset.view)));
$$('[data-go]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.go)));
$('#refresh').addEventListener('click',()=>loadView($('.nav.active').dataset.view,true));
setInterval(()=>$('#clock').textContent=new Date().toLocaleTimeString('es-ES'),1000);

function priorityForClaim(c){
  if(!c.analysis_status) return ['Pendiente','warn'];
  if(c.manual_review_required) return ['Alta','error'];
  if(c.risk_level==='Medio') return ['Media','warn'];
  return ['Normal','ok'];
}

function supervisorStatusPill(status){
  const map={
    pending_analysis:['Pendiente de análisis','warn'],
    pending_ground_truth:['Ground truth pendiente','warn'],
    ground_truth_confirmed:['Ground truth confirmado','ok'],
    reference_data:['Datos de referencia','neutral']
  };
  const [label,kind]=map[status]||[status||'—','neutral'];
  return statusPill(label,kind);
}
function groundTruthLabel(c){
  if(c.source==='kaggle_demo'){
    if(c.dataset_ground_truth===1) return statusPill('Referencia · fraude','error');
    if(c.dataset_ground_truth===0) return statusPill('Referencia · no fraude','neutral');
    return statusPill('Referencia','neutral');
  }
  if(!c.monitoring_recorded) return statusPill('Analizar primero','neutral');
  if(c.production_ground_truth===1) return statusPill('Fraude confirmado','error');
  if(c.production_ground_truth===0) return statusPill('No fraude confirmado','ok');
  return statusPill('Pendiente','warn');
}

async function loadOverview(){
  const [o,c,r]=await Promise.all([api('/admin/api/overview'),api('/admin/api/claims'),api('/admin/api/runs?limit=6')]);
  state.overview=o; state.claims=c.claims; state.runs=r.runs; state.decisionThreshold=c.decision_threshold;
  const monitored=state.claims.filter(x=>x.monitoring_recorded).length;
  const manual=state.claims.filter(x=>x.manual_review_required).length;
  const pendingGt=state.claims.filter(x=>x.source!=='kaggle_demo' && x.monitoring_recorded && x.production_ground_truth==null).length;
  const pendingAnalysis=state.claims.filter(x=>!x.analysis_status).length;
  $('#kpis').innerHTML=[
    ['Expedientes',o.claims,'Disponibles en la bandeja'],
    ['Revisión manual',manual,'Marcados por el modelo'],
    ['Ground truth pendiente',pendingGt,'Casos monitorizados sin resultado'],
    ['Pendientes de análisis',pendingAnalysis,'Sin ejecución registrada']
  ].map(([k,v,s])=>`<div class="kpi"><p class="eyebrow">${esc(k)}</p><div class="value">${esc(v)}</div><small>${esc(s)}</small></div>`).join('');
  const ragOk=o.rag?.status!=='unavailable'; const tgConf=o.telegram?.configuration?.configured;
  $('#system-status').innerHTML=[
    ['Modelo',`${o.model.name||'Logistic L1'} · v${o.model.version}`,'ok'],
    ['Umbral operativo',fmtPct(o.model.decision_threshold),'ok'],
    ['RAG',ragOk?'PostgreSQL + pgvector':'No disponible',ragOk?'ok':'warn'],
    ['Telegram',tgConf?'Bot API · long polling':'Modo demo local',tgConf?'ok':'warn'],
    ['MLOps',`${monitored} claims monitorizados`,'ok']
  ].map(([a,b,k])=>`<div class="status-row"><span>${esc(a)}</span>${statusPill(b,k)}</div>`).join('');
  $('#overview-claims').innerHTML=claimsTable(state.claims.slice().reverse().slice(0,6),false);
  $('#overview-runs').innerHTML=runsCards(state.runs);
}

function claimsTable(rows, clickable=true){
  if(!rows.length)return '<div class="empty">No hay claims.</div>';
  return `<table class="table"><thead><tr><th>Claim</th><th>Estado</th><th>Prioridad</th><th>Riesgo / score</th><th>Revisión</th><th>Ground truth</th><th>Docs</th><th>Último análisis</th></tr></thead><tbody>${rows.map(c=>{
    const [priority,kind]=priorityForClaim(c);
    return `<tr class="${clickable?'clickable':''}" ${clickable?`data-claim="${esc(c.claim_id)}"`:''}>
      <td><strong>${esc(c.claim_id)}</strong><small class="cell-sub">${esc(c.source||'—')}</small></td>
      <td>${supervisorStatusPill(c.supervisor_status)}</td>
      <td>${statusPill(priority,kind)}</td>
      <td><span class="risk ${esc(c.risk_level||'')}">${esc(c.risk_level||'—')}</span><small class="cell-sub">${fmtPct(c.fraud_score)}</small></td>
      <td>${c.analysis_status ? boolPill(Boolean(c.manual_review_required),'Manual','No requerida') : statusPill('Sin análisis','neutral')}</td>
      <td>${groundTruthLabel(c)}</td>
      <td>${(c.documents||[]).length}</td>
      <td>${fmtDate(c.last_analysis_at)}</td>
    </tr>`;
  }).join('')}</tbody></table>`;
}

async function loadClaims(){ const d=await api('/admin/api/claims'); state.claims=d.claims; state.decisionThreshold=d.decision_threshold; renderClaims(); }
function renderClaims(){
  const q=$('#claims-filter').value.toLowerCase().trim();
  const status=$('#claims-status-filter').value;
  const rows=state.claims.filter(c=>{
    const matchesText=!q || JSON.stringify(c).toLowerCase().includes(q);
    const matchesStatus=!status || c.supervisor_status===status;
    return matchesText && matchesStatus;
  });
  $('#claims-table').innerHTML=claimsTable(rows,true);
  $$('[data-claim]').forEach(r=>r.addEventListener('click',()=>showClaim(r.dataset.claim)));
}
$('#claims-filter').addEventListener('input',renderClaims);
$('#claims-status-filter').addEventListener('change',renderClaims);

function findAgent(trace, token){ return (trace||[]).find(t=>(t.agent_name||'').toLowerCase().includes(token.toLowerCase()))?.output || {}; }
function bytesLabel(v){ if(v==null)return '—'; const n=Number(v); if(n<1024)return `${n} B`; if(n<1024*1024)return `${(n/1024).toFixed(1)} KB`; return `${(n/1024/1024).toFixed(1)} MB`; }
function documentEvidence(docOut,claimId){
  const docs=docOut.document_files||[];
  if(!docs.length) return `<div class="evidence-empty"><strong>Sin documentación adjunta</strong><span>Los adjuntos del cliente son opcionales. El modelo de fraude funciona con los datos estructurados del claim.</span></div>`;
  return `<div class="document-list">${docs.map(d=>{
    const name=d.document_name||'';
    const openUrl=`/admin/api/claims/${encodeURIComponent(claimId)}/documents/${encodeURIComponent(name)}`;
    return `<div class="document-row">
      <div><strong>${esc(name)}</strong><span>${esc(d.mime_type||'Documento aportado por el cliente')}</span>
        <div class="evidence-actions"><button class="mini-button primary" data-open-claim-doc="${esc(name)}" data-doc-url="${esc(openUrl)}">↗ Abrir documento</button></div>
      </div>
      <div class="doc-meta">${statusPill(d.status==='available'?'Disponible':'No encontrado',d.status==='available'?'ok':'error')} ${d.size_bytes!=null?statusPill(bytesLabel(d.size_bytes),'neutral'):''}</div>
      <p class="doc-preview">Archivo original disponible para revisión directa por el supervisor.</p>
    </div>`;
  }).join('')}</div>`;
}
function relevanceBadge(level){
  const map={alta:'ok',media:'neutral',baja:'warn',insuficiente:'error'};
  return statusPill(`Relevancia ${level||'—'}`,map[level]||'neutral');
}
function evidenceItems(title,items,kind){
  if(!items?.length)return '';
  const cls=kind==='exclusion'?'evidence-exclusion':kind==='coverage'?'evidence-coverage':'evidence-condition';
  return `<div class="policy-evidence-group ${cls}"><h4>${esc(title)}</h4>${items.map(x=>{
    const pages=x.pages?.length?` · p. ${x.pages.join(', ')}`:'';
    return `<div class="policy-evidence-item"><div><strong>${esc(x.chunk_id||'Fuente')}</strong><small>${esc(x.source||'')}${esc(pages)}</small></div><p>${esc(x.text||'')}</p></div>`;
  }).join('')}</div>`;
}
function ragEvidence(rag){
  const sources=rag.sources||[];
  const policyId=rag.policy_id||'';
  const policyUrl=policyId?`/admin/api/policies/${encodeURIComponent(policyId)}/document`:'';
  const quality=rag.retrieval_quality||'—'; const best=rag.top_semantic_score;
  return `<div class="policy-summary policy-summary-extended">
    <div class="info-box"><span>Resultado orientativo</span><b>${esc(rag.coverage_result||'Sin consulta')}</b></div>
    <div class="info-box"><span>Póliza</span><b>${esc(policyId||'No seleccionada')}</b>${policyUrl?`<div class="evidence-actions"><button class="mini-button" data-open-policy data-policy-url="${esc(policyUrl)}">↗ Abrir póliza completa</button></div>`:''}</div>
    <div class="info-box"><span>Calidad del retrieval</span><b>${esc(quality)}</b></div>
    <div class="info-box"><span>Mejor similitud coseno</span><b>${best==null?'—':Number(best).toFixed(3)}</b></div>
  </div>
  ${rag.query?`<div class="query-readback"><span>Pregunta del supervisor</span><strong>${esc(rag.query)}</strong></div>`:''}
  <p class="muted">${esc(rag.rag_answer||'Formula una pregunta para recuperar evidencia de la póliza.')}</p>
  ${rag.score_notice?`<div class="score-notice">ℹ️ ${esc(rag.score_notice)}</div>`:''}
  ${rag.recommended_action?`<div class="recommended-action"><span>Acción recomendada al especialista</span><strong>${esc(rag.recommended_action)}</strong></div>`:''}
  <div class="policy-evidence-grid">
    ${evidenceItems('Evidencia de cobertura',rag.coverage_evidence,'coverage')}
    ${evidenceItems('Posibles exclusiones',rag.exclusion_evidence,'exclusion')}
    ${evidenceItems('Condiciones detectadas',rag.condition_evidence,'condition')}
  </div>
  ${sources.length?`<div class="source-list source-list-review">${sources.map((src,i)=>`<div class="source-card">
    <div class="source-head"><strong>Fuente ${i+1} · ${esc(src.chunk_id||'chunk')}</strong>${relevanceBadge(src.relevance_level)}</div>
    <div class="source-score">Similitud coseno <b>${src.semantic_score==null?'—':Number(src.semantic_score).toFixed(3)}</b> · señal de recuperación, no probabilidad</div>
    ${src.pages?.length?`<div class="source-pages">${src.pages.map(pg=>statusPill(`Página ${pg}`,'neutral')).join('')}</div>`:''}
    <p>${esc(src.snippet||'')}</p>
    <div class="evidence-actions">
      <button class="mini-button primary" data-rag-source="${i}">⌕ Ver fragmento exacto</button>
      ${policyUrl?`<button class="mini-button" data-open-policy-page="${i}">↗ Abrir en documento</button>`:''}
    </div>
  </div>`).join('')}</div>`:'<div class="evidence-empty compact"><span>No hay fragmentos recuperados todavía.</span></div>'}`;
}
function policyReviewWorkbench(d,rag){
  const policies=d.available_policies||[];
  const selected=rag.policy_id||policies[0]?.policy_id||'';
  const query=rag.query||'';
  const options=policies.map(p=>`<option value="${esc(p.policy_id)}" ${p.policy_id===selected?'selected':''}>${esc(p.policy_id)}${p.name?` · ${esc(p.name)}`:''}</option>`).join('');
  return `<div class="policy-review-workbench">
    <div class="policy-review-fields">
      <label><span>Póliza a consultar</span><select id="policy-review-policy">${options}</select></label>
      <label class="policy-question"><span>Pregunta del supervisor</span><textarea id="policy-review-query" rows="3" placeholder="Ej.: ¿Qué exclusiones pueden afectar a este siniestro?">${esc(query)}</textarea></label>
    </div>
    <div class="policy-review-presets">
      <button type="button" class="chip-button" data-policy-preset="¿Qué coberturas de la póliza pueden aplicar a este siniestro?">Coberturas aplicables</button>
      <button type="button" class="chip-button" data-policy-preset="¿Existe alguna exclusión o limitación aplicable a este siniestro?">Buscar exclusiones</button>
      <button type="button" class="chip-button" data-policy-preset="¿Qué condiciones, franquicias o límites deben comprobarse?">Condiciones y límites</button>
      <button type="button" class="chip-button" data-policy-preset="¿Qué documentación exige la póliza para tramitar este tipo de siniestro?">Documentación requerida</button>
    </div>
    <div class="policy-review-actions"><button id="run-policy-review" class="button" type="button">🔎 Preguntar a la póliza</button><small>Solo visible para el supervisor. No vuelve a ejecutar el modelo de fraude ni genera un evento MLOps.</small></div>
  </div>`;
}
function shapEvidence(fraud){
  const items=(fraud.local_shap||[]).filter(x=>Math.abs(Number(x.shap_value||0))>0).slice(0,6);
  if(!items.length)return '<div class="evidence-empty compact"><span>No hay contribuciones SHAP distintas de cero para mostrar.</span></div>';
  return `<div class="factor-list">${items.map(x=>`<div class="factor-row"><div><strong>${esc(x.feature)}</strong><span>${x.direction==='fraud'?'Aumenta señal de fraude':'Reduce señal de fraude'}</span></div><b class="${x.direction==='fraud'?'factor-up':'factor-down'}">${Number(x.shap_value).toFixed(3)}</b></div>`).join('')}</div>`;
}
function verdictText(score, threshold){ if(score==null)return 'Sin predicción'; return Number(score)>=Number(threshold) ? 'Fraude probable' : 'No fraude probable'; }
function comparisonHtml(score, threshold, gt){ if(gt==null || score==null)return statusPill('Pendiente de resultado confirmado','warn'); const pred=Number(score)>=Number(threshold)?1:0; return pred===Number(gt)?statusPill('Modelo y ground truth coinciden','ok'):statusPill('Discrepancia modelo / ground truth','error'); }
function groundTruthPanel(d,res){
  if(d.source==='kaggle_demo'){ const gt=d.dataset_ground_truth; return `<div class="ground-truth-panel reference"><div><p class="eyebrow">ETIQUETA DE REFERENCIA</p><h3>Ground truth del dataset</h3><p class="muted">Se muestra únicamente para evaluación demo. No se usa como entrada del modelo.</p></div><div class="gt-current">${gt===1?statusPill('Fraude','error'):gt===0?statusPill('No fraude','ok'):statusPill('No disponible','neutral')}</div></div>`; }
  const mon=d.monitoring; const gt=mon?.FraudFound_P; const canSet=Boolean(mon);
  return `<div class="ground-truth-panel"><div class="gt-copy"><p class="eyebrow">RESOLUCIÓN DEL SUPERVISOR</p><h3>Ground truth confirmado</h3><p class="muted">Registra el resultado real cuando la investigación esté cerrada. Alimenta las métricas MLOps y no modifica la inferencia histórica.</p>${comparisonHtml(res.fraud_score,d.decision_threshold,gt)}</div><div class="gt-actions"><div class="gt-current">${gt===1?statusPill('Actual: FRAUDE','error'):gt===0?statusPill('Actual: NO FRAUDE','ok'):statusPill(canSet?'Sin confirmar':'Analiza primero','warn')}</div><button class="button success" data-ground-truth="0" ${canSet?'':'disabled'}>✓ Confirmar NO fraude</button><button class="button danger" data-ground-truth="1" ${canSet?'':'disabled'}>⚠ Confirmar FRAUDE</button>${gt!=null?`<small>Actualizado: ${fmtDate(mon.ground_truth_updated_at)}</small>`:''}</div></div>`;
}
function bindPolicyEvidence(rag){
  const policyUrl=rag.policy_id?`/admin/api/policies/${encodeURIComponent(rag.policy_id)}/document`:null;
  $$('[data-open-policy]').forEach(btn=>btn.addEventListener('click',()=>openDocumentViewer(`Póliza ${rag.policy_id}`,btn.dataset.policyUrl)));
  $$('[data-rag-source]').forEach(btn=>btn.addEventListener('click',()=>{
    const src=(rag.sources||[])[Number(btn.dataset.ragSource)]||{}; const page=src.page||src.pages?.[0]||null;
    const meta=[src.semantic_score!=null?statusPill(`${(Number(src.semantic_score)*100).toFixed(1)}% similitud`,'neutral'):'',src.chunk_id?statusPill(src.chunk_id,'neutral'):'',src.pages?.length?statusPill(`Página(s): ${src.pages.join(', ')}`,'neutral'):''].join('');
    const actions=policyUrl?`<button class="button secondary" id="modal-open-policy-source">Abrir póliza${page?` · p. ${page}`:''} ↗</button><button class="button" data-close-modal>Cerrar</button>`:`<button class="button" data-close-modal>Cerrar</button>`;
    showEvidenceModal({eyebrow:'EVIDENCIA RAG · FRAGMENTO EXACTO',title:`${rag.policy_id||'Póliza'} · ${src.chunk_id||'chunk'}`,meta,body:`<pre>${esc(src.exact_text||src.snippet||'')}</pre><p class="exact-evidence-note">Fragmento exacto recuperado desde PostgreSQL + pgvector. Contrástalo con el PDF original antes de resolver el expediente.</p>`,actions});
    bindModalClose(); const open=$('#modal-open-policy-source'); if(open)open.onclick=()=>openDocumentViewer(`Póliza ${rag.policy_id}`,policyUrl,page,meta);
  }));
  $$('[data-open-policy-page]').forEach(btn=>btn.addEventListener('click',()=>{ const src=(rag.sources||[])[Number(btn.dataset.openPolicyPage)]||{}; openDocumentViewer(`Póliza ${rag.policy_id}`,policyUrl,src.page||src.pages?.[0]||null); }));
}
async function showClaim(id){
  const d=await api(`/admin/api/claims/${encodeURIComponent(id)}`);
  const latest=d.runs?.[0]; const res=latest?.result||{}; const trace=res.agents_trace||[];
  const docs=findAgent(trace,'documental'); const fraud=findAgent(trace,'fraude'); const validation=findAgent(trace,'validación');
  const policyReview=d.policy_review?.output||{}; const threshold=d.decision_threshold;
  const [priority,priorityKind]=priorityForClaim({analysis_status:latest?.status,manual_review_required:res.manual_review_required,risk_level:res.risk_level});
  const el=$('#claim-detail'); el.classList.remove('hidden');
  el.innerHTML=`
    <div class="case-header"><div><p class="eyebrow">EXPEDIENTE · ${esc(d.source)}</p><h2>${esc(id)}</h2><div class="case-tags">${statusPill(`Prioridad ${priority}`,priorityKind)} ${latest?statusPill('Analizado','ok'):statusPill('Pendiente de análisis','warn')}</div></div><button id="analyze-claim" class="button">↻ Ejecutar / actualizar fraude</button></div>
    <div class="supervisor-grid">
      <div class="decision-card primary"><span>Fraud score</span><strong>${fmtPct(res.fraud_score)}</strong><small>Umbral ${fmtPct(threshold)}</small></div>
      <div class="decision-card"><span>Decisión del modelo</span><strong>${esc(verdictText(res.fraud_score,threshold))}</strong><small>${esc(res.risk_level||'Sin nivel de riesgo')}</small></div>
      <div class="decision-card"><span>Revisión manual</span><strong>${res.fraud_score==null?'—':res.manual_review_required?'REQUERIDA':'NO REQUERIDA'}</strong><small>Señal de priorización</small></div>
      <div class="decision-card"><span>Documentos del cliente</span><strong>${(d.claim.documents||[]).length}</strong><small>${docs.documents_missing?.length?`${docs.documents_missing.length} no localizados`:'Evidencia original disponible'}</small></div>
    </div>
    ${groundTruthPanel(d,res)}
    <div class="supervisor-sections">
      <section class="work-card"><div class="work-head"><div><p class="eyebrow">EVIDENCIA DOCUMENTAL</p><h3>Documentos aportados por el cliente</h3></div>${statusPill('Revisión documental','neutral')}</div><p class="muted">Los adjuntos se conservan como evidencia original y pueden abrirse directamente desde el expediente.</p>${documentEvidence(docs,id)}</section>
      <section class="work-card"><div class="work-head"><div><p class="eyebrow">EXPLICABILIDAD</p><h3>Factores locales del modelo</h3></div>${fraud.model_used?statusPill(fraud.model_used,'neutral'):''}</div>${shapEvidence(fraud)}</section>
      <section class="work-card wide policy-review-card"><div class="work-head"><div><p class="eyebrow">SOLO SUPERVISOR</p><h3>Asistente RAG de pólizas</h3></div>${policyReview.coverage_result?statusPill(policyReview.coverage_result,policyReview.coverage_result.includes('exclusión')||policyReview.coverage_result.includes('Revisión')?'warn':policyReview.coverage_result.includes('cubierto')?'ok':'neutral'):statusPill('Sin consulta','neutral')}</div>${policyReviewWorkbench(d,policyReview)}${ragEvidence(policyReview)}</section>
      <section class="work-card"><div class="work-head"><div><p class="eyebrow">CALIDAD DEL EXPEDIENTE</p><h3>Validación estructurada</h3></div>${validation.validation_status==='valid'?statusPill('Válido','ok'):statusPill(validation.validation_status||'Sin validar','warn')}</div><p class="muted">${(validation.issues||[]).length?esc((validation.issues||[]).join(' · ')):'Las variables suministradas cumplen el contrato esperado por el modelo.'}</p></section>
      <section class="work-card"><div class="work-head"><div><p class="eyebrow">EXPLICACIÓN CONSOLIDADA</p><h3>Resumen antifraude</h3></div></div><p class="explanation-text">${esc(res.explanation||'Ejecuta el análisis para generar una explicación.')}</p></section>
    </div>
    <details class="technical-details"><summary>Ver traza técnica multiagente completa</summary><div class="details-body">${traceHtml(trace)}</div></details>
    <details class="technical-details"><summary>Ver datos estructurados del claim</summary><div class="details-body"><pre>${esc(pretty(d.claim))}</pre></div></details>`;

  $$('[data-open-claim-doc]').forEach(btn=>btn.addEventListener('click',()=>openDocumentViewer(btn.dataset.openClaimDoc,btn.dataset.docUrl)));
  bindPolicyEvidence(policyReview);
  $$('[data-policy-preset]').forEach(btn=>btn.addEventListener('click',()=>{ const q=$('#policy-review-query'); q.value=btn.dataset.policyPreset||''; q.focus(); }));
  $('#run-policy-review')?.addEventListener('click',async()=>{
    const policyId=$('#policy-review-policy')?.value; const query=$('#policy-review-query')?.value.trim();
    if(!policyId){toast('Selecciona una póliza');return;} if(!query || query.length<3){toast('Escribe una pregunta para la póliza');return;}
    const button=$('#run-policy-review'); const oldText=button.textContent; button.disabled=true; button.textContent='Consultando pgvector…';
    try{ await api(`/admin/api/claims/${encodeURIComponent(id)}/policy-review`,{method:'POST',body:JSON.stringify({policy_id:policyId,query})}); toast('Consulta de póliza actualizada'); await showClaim(id); }
    catch(e){toast(e.message)} finally { if(button?.isConnected){button.disabled=false;button.textContent=oldText;} }
  });
  $('#analyze-claim').addEventListener('click',async()=>{ try{await api(`/admin/api/claims/${encodeURIComponent(id)}/analyze`,{method:'POST',body:'{}'});toast('Análisis antifraude completado');await loadClaims();await showClaim(id);}catch(e){toast(e.message)} });
  $$('[data-ground-truth]').forEach(btn=>btn.addEventListener('click',async()=>{ const value=Number(btn.dataset.groundTruth); const label=value===1?'FRAUDE':'NO FRAUDE'; if(!window.confirm(`¿Confirmar ground truth como ${label} para ${id}?`))return; try{await api(`/admin/api/claims/${encodeURIComponent(id)}/ground-truth`,{method:'PUT',body:JSON.stringify({FraudFound_P:value})});toast(`Ground truth registrado: ${label}`);await loadClaims();await showClaim(id);}catch(e){toast(e.message)} }));
  el.scrollIntoView({behavior:'smooth',block:'start'});
}

function traceHtml(trace){ if(!trace.length)return '<div class="empty">Aún no hay ejecución registrada.</div>'; return `<div class="trace">${trace.map(t=>`<div class="trace-item"><div class="trace-head"><strong>${esc(t.agent_name)}</strong><small>${esc(t.status)} · ${Number(t.duration_ms||0).toFixed(2)} ms</small></div><pre>${esc(pretty(t.output))}</pre></div>`).join('')}</div>`; }
function runsCards(rows){ if(!rows.length)return '<div class="empty">No hay ejecuciones.</div>'; return rows.map(r=>`<div class="run-card" data-run="${esc(r.run_id)}"><strong>${esc(r.claim_id)}</strong><small>${esc(r.source_channel)} · ${fmtDate(r.completed_at)} · ${Number(r.duration_ms||0).toFixed(0)} ms</small></div>`).join(''); }
async function loadRuns(){ const d=await api('/admin/api/runs?limit=100'); state.runs=d.runs; $('#runs-list').innerHTML=runsCards(state.runs); $$('[data-run]').forEach(x=>x.addEventListener('click',()=>showRun(x.dataset.run))); }
async function showRun(id){ const r=await api(`/admin/api/runs/${id}`); const result=r.result||{}; $('#run-detail').innerHTML=`<div class="panel-head"><div><p class="eyebrow">RUN ${esc(r.run_id.slice(0,8))}</p><h2>${esc(r.claim_id)}</h2></div>${statusPill(r.status,r.status==='ok'?'ok':'error')}</div><div class="json-grid"><div class="info-box"><span>Canal</span><b>${esc(r.source_channel)}</b></div><div class="info-box"><span>Duración</span><b>${Number(r.duration_ms||0).toFixed(2)} ms</b></div><div class="info-box"><span>Fraud score</span><b>${fmtPct(result.fraud_score)}</b></div><div class="info-box"><span>Riesgo</span><b class="risk ${esc(result.risk_level||'')}">${esc(result.risk_level||'—')}</b></div></div><h3 style="margin-top:18px">Agentes ejecutados</h3>${traceHtml(result.agents_trace||[])}${r.error?`<pre>${esc(r.error)}</pre>`:''}`; }

async function loadTelegram(){ const d=await api('/admin/api/telegram/sessions'); state.tgSessions=d.sessions; $('#tg-mode').textContent=d.configuration.mode==='telegram_long_polling'?'Bot API · long polling':'Modo demo local'; $('#tg-sessions').innerHTML=d.sessions.length?d.sessions.map(s=>`<div class="session-card" data-chat-id="${esc(s.chat_id)}"><strong>${esc(s.claim_id||'Sin claim')}</strong><small>${esc(s.chat_id_masked)} · ${esc(s.status)} · paso ${s.current_step}</small></div>`).join(''):'<div class="empty">Sin sesiones. Usa el simulador con NUEVO.</div>'; $$('[data-chat-id]').forEach(x=>x.addEventListener('click',()=>loadChat(x.dataset.chatId))); if(state.selectedChatId) await loadChat(state.selectedChatId); }
async function loadChat(chatId){ state.selectedChatId=chatId; $('#tg-chat-id').value=chatId; const d=await api(`/admin/api/telegram/sessions/${encodeURIComponent(chatId)}/messages`); $('#chat-title').textContent=`Conversación · ${chatId.slice(-4).padStart(chatId.length,'*')}`; $('#chat-messages').innerHTML=d.messages.length?d.messages.map(m=>`<div class="message ${m.direction}">${esc(m.text||m.file_name||`[${m.message_type}]`)}<div class="message-meta">${esc(m.direction)} · ${fmtDate(m.created_at)} · ${esc(m.delivery_status||'')}</div></div>`).join(''):'<div class="empty">Sin mensajes.</div>'; $('#chat-messages').scrollTop=$('#chat-messages').scrollHeight; }
$('#tg-simulator').addEventListener('submit',async(e)=>{ e.preventDefault(); const chatId=$('#tg-chat-id').value.trim(); const text=$('#tg-text').value.trim(); if(!text)return; try{await api('/admin/api/telegram/simulate',{method:'POST',body:JSON.stringify({chat_id:chatId,text})}); $('#tg-text').value=''; state.selectedChatId=chatId; await loadTelegram(); toast('Mensaje procesado por Telegram Intake Agent'); }catch(err){toast(err.message)} });

async function loadMlops(){ const d=await api('/admin/api/mlops'); const cards=[['Monitorización',`${d.monitoring.unique_claims||0} claims`,d.monitoring.status||'activo'],['Data drift',d.data_drift.overall_status||d.data_drift.status,`${d.data_drift.production_claims||d.data_drift.production_claims_total||0}/500`],['Prediction drift',d.prediction_drift.drift_status||d.prediction_drift.status,`PSI ${d.prediction_drift.psi??'—'}`],['Performance',d.performance.status,`${d.performance.labeled_claims||0} etiquetados`],['Reentrenamiento',d.retraining.retraining_recommended?'Recomendado':'No',d.retraining.status],['Ground truth',`${d.monitoring.labeled_claims||0}`, 'resultados confirmados']]; $('#mlops-cards').innerHTML=cards.map(([a,b,c])=>`<div class="panel"><p class="eyebrow">${esc(a)}</p><h2>${esc(b)}</h2><p class="muted">${esc(c)}</p></div>`).join(''); $('#mlops-json').textContent=pretty(d); }
function policyExtractionLabel(method){ const labels={pdf_hybrid_text_ocr:'PDF híbrido · OCR selectivo',pdf_ocr_tesseract:'PDF escaneado · OCR',pdf_text_extraction:'PDF digital · texto nativo',pdf_text_extraction_pypdf:'PDF digital · pypdf'}; return labels[method]||method||'No disponible'; }
async function loadRag(){
  const [d,p]=await Promise.all([api('/admin/api/rag'),api('/admin/api/policies')]);
  const cards=[['Backend',d.backend||'PostgreSQL + pgvector'],['Pólizas',d.total_policies??'—'],['Chunks',d.total_chunks??'—'],['Embedding',d.embedding_dimension?`${d.embedding_dimension}D`:'—']];
  $('#rag-cards').innerHTML=cards.map(([a,b])=>`<div class="kpi"><p class="eyebrow">${esc(a)}</p><div class="value" style="font-size:20px">${esc(b)}</div></div>`).join('');
  $('#rag-json').textContent=pretty(d);
  const policies=p.policies||[];
  $('#rag-policy-select').innerHTML=policies.map(x=>`<option value="${esc(x.policy_id)}">${esc(x.policy_id)}${x.name?` · ${esc(x.name)}`:''}</option>`).join('');
  $('#rag-documents').innerHTML=(d.documents||[]).length?`<div class="document-list">${d.documents.map(x=>`<div class="document-row"><div><strong>${esc(x.policy_id)}</strong><span>${esc(x.source)}</span><div class="evidence-actions"><button class="mini-button" data-open-indexed-policy="${esc(x.policy_id)}">↗ Abrir póliza</button></div></div><div class="doc-meta">${statusPill(policyExtractionLabel(x.extraction_method),x.ocr_pages?.length?'warn':'ok')}${x.native_pages?.length?statusPill(`Texto nativo: ${x.native_pages.length} pág.`,'ok'):''}${x.ocr_pages?.length?statusPill(`OCR: p. ${x.ocr_pages.join(', ')}`,'warn'):''}${x.unreadable_pages?.length?statusPill(`Sin leer: p. ${x.unreadable_pages.join(', ')}`,'error'):''}${statusPill(`${x.chunks} chunks`,'neutral')}</div><p class="doc-preview">OCR se aplica únicamente a páginas de póliza sin texto nativo suficiente.</p></div>`).join('')}</div>`:'<div class="empty">No hay pólizas indexadas.</div>';
  $$('[data-open-indexed-policy]').forEach(btn=>btn.addEventListener('click',()=>openDocumentViewer(`Póliza ${btn.dataset.openIndexedPolicy}`,`/admin/api/policies/${encodeURIComponent(btn.dataset.openIndexedPolicy)}/document`)));
  $$('[data-rag-preset]').forEach(btn=>btn.addEventListener('click',()=>{ $('#rag-policy-query').value=btn.dataset.ragPreset||''; $('#rag-policy-query').focus(); }));
  $('#run-rag-policy-query').onclick=async()=>{
    const policyId=$('#rag-policy-select').value; const query=$('#rag-policy-query').value.trim();
    if(!policyId){toast('Selecciona una póliza');return;} if(query.length<3){toast('Escribe una pregunta');return;}
    const b=$('#run-rag-policy-query'); const old=b.textContent; b.disabled=true; b.textContent='Consultando pgvector…';
    try{ const r=await api('/admin/api/policies/query',{method:'POST',body:JSON.stringify({policy_id:policyId,query})}); const out=r.output||{}; $('#rag-query-result').innerHTML=ragEvidence(out); bindPolicyEvidence(out); }
    catch(e){toast(e.message)} finally{b.disabled=false;b.textContent=old;}
  };
}
async function loadView(name){ try{ if(name==='overview')await loadOverview(); if(name==='claims')await loadClaims(); if(name==='telegram')await loadTelegram(); if(name==='runs')await loadRuns(); if(name==='mlops')await loadMlops(); if(name==='rag')await loadRag(); }catch(e){ console.error(e); toast(e.message); } }
loadOverview();
