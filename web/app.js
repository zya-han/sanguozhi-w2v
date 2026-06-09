/* 《三國志》 Word2Vec 탐색기 — 순수 클라이언트 코사인 엔진.
   벡터는 L2 정규화돼 있으므로 코사인 = 내적(dot). 백엔드 없음. */
'use strict';

let DIM = 0, COUNT = 0;
let V = null;                 // Float32Array [COUNT*DIM], 정규화됨
let TOKENS = [];              // index -> token
let FREQ = [];
const tokenToIndex = new Map();
const readingToTokens = new Map();   // 한글음 -> [token...] (freq 내림차순)
const tokenToReading = new Map();

/* ---------- 데이터 로드 ---------- */
async function load() {
  let vocab, readings, buf;
  if (window.__W2V_EMBED__) {                 // 단일 HTML(내장) 모드
    const e = window.__W2V_EMBED__;
    vocab = e.vocab; readings = e.readings;
    const bin = atob(e.vectorsB64), n = bin.length, bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) bytes[i] = bin.charCodeAt(i);
    buf = bytes.buffer;
  } else {                                    // 정적 파일(fetch) 모드
    [vocab, readings, buf] = await Promise.all([
      fetch('data/vocab.json').then(r => r.json()),
      fetch('data/readings.json').then(r => r.json()),
      fetch('data/vectors.bin').then(r => r.arrayBuffer()),
    ]);
  }
  DIM = vocab.dim; COUNT = vocab.count;
  TOKENS = vocab.tokens; FREQ = vocab.freq;
  V = new Float32Array(buf);
  if (V.length !== COUNT * DIM) throw new Error('벡터 크기 불일치');

  TOKENS.forEach((t, i) => tokenToIndex.set(t, i));
  for (const [t, r] of Object.entries(readings)) {
    tokenToReading.set(t, r);
    if (!readingToTokens.has(r)) readingToTokens.set(r, []);
    readingToTokens.get(r).push(t);
  }
  for (const arr of readingToTokens.values())
    arr.sort((a, b) => (FREQ[tokenToIndex.get(b)] || 0) - (FREQ[tokenToIndex.get(a)] || 0));

  document.getElementById('vocabCount').textContent = COUNT.toLocaleString();
  document.getElementById('loadStatus').remove();
  document.getElementById('app').hidden = false;
}

/* ---------- 핵심 연산 ---------- */
function dot(i, j) {
  let s = 0; const a = i * DIM, b = j * DIM;
  for (let k = 0; k < DIM; k++) s += V[a + k] * V[b + k];
  return s;
}
function mostSimilar(i, topn) {
  const base = i * DIM, out = [];
  for (let j = 0; j < COUNT; j++) {
    if (j === i) continue;
    let s = 0; const off = j * DIM;
    for (let k = 0; k < DIM; k++) s += V[off + k] * V[base + k];
    out.push([j, s]);
  }
  out.sort((a, b) => b[1] - a[1]);
  return out.slice(0, topn).map(([j, s]) => [TOKENS[j], s]);
}
// gensim doesnt_match: 평균 벡터(정규화)와의 코사인이 최소인 토큰.
function doesntMatch(indices) {
  const mean = new Float32Array(DIM);
  for (const i of indices) { const off = i * DIM; for (let k = 0; k < DIM; k++) mean[k] += V[off + k]; }
  let nrm = 0; for (let k = 0; k < DIM; k++) nrm += mean[k] * mean[k];
  nrm = Math.sqrt(nrm) || 1; for (let k = 0; k < DIM; k++) mean[k] /= nrm;
  let worst = indices[0], worstSim = Infinity;
  const scored = indices.map(i => {
    let s = 0; const off = i * DIM; for (let k = 0; k < DIM; k++) s += V[off + k] * mean[k];
    if (s < worstSim) { worstSim = s; worst = i; }
    return [i, s];
  });
  return { worst, scored };
}

/* ---------- 입력 해석 (한자 또는 한글 음) ---------- */
// 반환: {token} | {candidates:[...]} | {missing:str}
function resolve(raw) {
  const s = (raw || '').trim();
  if (!s) return { missing: '' };
  if (tokenToIndex.has(s)) return { token: s };
  if (readingToTokens.has(s)) {
    const c = readingToTokens.get(s);
    return c.length === 1 ? { token: c[0] } : { candidates: c };
  }
  return { missing: s };
}

/* ---------- 렌더 헬퍼 ---------- */
const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function rdSpan(tok) { const r = tokenToReading.get(tok); return r ? ` <span class="rd">${esc(r)}</span>` : ''; }
function commonKey(tok) { const b = tok.replace(/[《》]/g, ''); return b.length >= 2 ? b.slice(-2) : b; }

// 막대 행들 (정규화: 최댓값 = 100%)
function bars(list, commonSet) {
  const max = list.length ? list[0][1] : 1;
  return list.map(([t, s]) => {
    const pct = Math.max(5, Math.round((s / max) * 100));
    const cm = commonSet && commonSet.has(commonKey(t)) ? ' common' : '';
    return `<div class="bar-row${cm}" data-token="${esc(t)}">
      <span class="bar-label"><span class="bw han">${esc(t)}</span>${rdSpan(t)}</span>
      <span class="bar-track"><span class="bar-cover" style="width:${100 - pct}%"></span></span>
      <span class="bar-score">${s.toFixed(3)}</span></div>`;
  }).join('');
}
function notFound(box, term) {
  box.innerHTML = `<p class="warn">‘${esc(term)}’ 을(를) 어휘에서 찾지 못했습니다. 한자 또는 한글 음으로 다시 입력해 주세요.</p>`;
}
function pickFirst(r) { return r.token || (r.candidates && r.candidates[0]); }

/* ---------- 1. 두 단어 비교 ---------- */
function runCmp() {
  const box = document.getElementById('cmpResult');
  const ra = resolve(document.getElementById('cmpA').value);
  const rb = resolve(document.getElementById('cmpB').value);
  const ta = pickFirst(ra), tb = pickFirst(rb);
  if (!ta) return notFound(box, document.getElementById('cmpA').value);
  if (!tb) return notFound(box, document.getElementById('cmpB').value);
  const ia = tokenToIndex.get(ta), ib = tokenToIndex.get(tb);
  const sim = dot(ia, ib);
  const la = mostSimilar(ia, 14), lb = mostSimilar(ib, 14);
  const ka = new Set(la.map(([t]) => commonKey(t)));
  const kb = new Set(lb.map(([t]) => commonKey(t)));
  const common = new Set([...ka].filter(k => kb.has(k)));
  box.innerHTML = `
    <div class="bigscore"><div class="num">${sim.toFixed(3)}</div>
      <div class="cap"><span class="han">${esc(ta)}</span>${rdSpan(ta)} ↔
        <span class="han">${esc(tb)}</span>${rdSpan(tb)} 코사인 유사도</div></div>
    <label class="common-toggle"><input type="checkbox" id="cmpCommon">
      양쪽 공통(끝 두 글자가 같으면 공통)</label>
    <div class="cols" id="cmpCols">
      <div><div class="col-head"><span class="han">${esc(ta)}</span>${rdSpan(ta)} 과 비슷한 단어</div>
        <div class="bars">${bars(la, common)}</div></div>
      <div><div class="col-head"><span class="han">${esc(tb)}</span>${rdSpan(tb)} 과 비슷한 단어</div>
        <div class="bars">${bars(lb, common)}</div></div>
    </div>`;
  document.getElementById('cmpCommon').addEventListener('change', e =>
    document.getElementById('cmpCols').classList.toggle('show-common', e.target.checked));
}

/* ---------- 2. 스파이 찾기 ---------- */
let spyTokens = [];
function renderSpyChips() {
  document.getElementById('spyChips').innerHTML = spyTokens.map((t, i) =>
    `<span class="chip"><span class="han">${esc(t)}</span>${rdSpan(t)}<span class="x" data-i="${i}">✕</span></span>`
  ).join('');
  document.querySelectorAll('#spyChips .x').forEach(x =>
    x.addEventListener('click', () => { spyTokens.splice(+x.dataset.i, 1); renderSpyChips(); }));
}
function addSpyToken(raw) {
  const t = pickFirst(resolve(raw));
  if (t && !spyTokens.includes(t)) spyTokens.push(t);
  renderSpyChips();
}
function runSpy() {
  const box = document.getElementById('spyResult');
  if (spyTokens.length < 3) return box.innerHTML = `<p class="warn">단어를 3개 이상 모아 주세요.</p>`;
  const idx = spyTokens.map(t => tokenToIndex.get(t));
  const { worst, scored } = doesntMatch(idx);
  scored.sort((a, b) => a[1] - b[1]);
  const list = scored.map(([i, s]) =>
    `<span class="${i === worst ? 'lo' : ''}">${esc(TOKENS[i])}(${esc(tokenToReading.get(TOKENS[i]) || '')}) ${s.toFixed(3)}</span>`
  ).join(' · ');
  box.innerHTML = `<div class="spybox">
    <div class="quote">우리 중에 스파이가 있는 것 같아…</div>
    <div class="who"><span class="han">${esc(TOKENS[worst])}</span><span class="rd">${esc(tokenToReading.get(TOKENS[worst]) || '')}</span></div>
    <div class="scores">무리 평균과의 유사도: ${list}</div></div>`;
}

/* ---------- 3. 비슷한 단어 찾기 ---------- */
function runSim(term) {
  const box = document.getElementById('simResult');
  const topn = +document.getElementById('simTopn').value;
  const r = resolve(term);
  const t = pickFirst(r);
  if (!t) return notFound(box, term);
  const sims = mostSimilar(tokenToIndex.get(t), topn);
  box.innerHTML = `<div class="qword"><span class="han">${esc(t)}</span>${rdSpan(t)} 와 비슷한 단어 ${topn}</div>
    <div class="bars">${bars(sims)}</div>`;
}

/* ---------- 자동완성 (빈도순) ---------- */
const SUGGEST_N = 10;
function suggestList(q) {
  // TOKENS는 빈도 내림차순(gensim index_to_key) → 빈 입력은 상위 빈도 그대로.
  if (!q) return TOKENS.slice(0, SUGGEST_N);
  // 한자·한글 독음에서 '포함' 일치 모두 후보. 시작(prefix) 일치를 앞에, 그 다음 빈도순.
  const rank = new Map();                       // token -> 0(prefix) | 1(contains)
  const mark = (t, r) => { const c = rank.get(t); if (c === undefined || r < c) rank.set(t, r); };
  for (const t of TOKENS) { const i = t.indexOf(q); if (i === 0) mark(t, 0); else if (i > 0) mark(t, 1); }
  for (const [r, toks] of readingToTokens) {
    const i = r.indexOf(q); if (i < 0) continue;
    const rk = i === 0 ? 0 : 1; for (const t of toks) mark(t, rk);
  }
  return [...rank.keys()].sort((a, b) => {
    const d = rank.get(a) - rank.get(b);
    return d || (FREQ[tokenToIndex.get(b)] || 0) - (FREQ[tokenToIndex.get(a)] || 0);
  }).slice(0, SUGGEST_N);
}
function attachSuggest(inputId, boxId, onSelect) {
  const input = document.getElementById(inputId), box = document.getElementById(boxId);
  let act = -1, items = [];
  const hide = () => { box.hidden = true; box.innerHTML = ''; act = -1; items = []; };
  const draw = () => {
    box.innerHTML = items.map((t, i) => {
      const f = FREQ[tokenToIndex.get(t)] || 0;
      return `<div class="sg${i === act ? ' act' : ''}" data-token="${esc(t)}"><span class="bw han">${esc(t)}</span>${rdSpan(t)}<span class="fr">${f.toLocaleString()}</span></div>`;
    }).join('');
    box.hidden = !items.length;
    box.querySelectorAll('.sg').forEach(d => d.addEventListener('mousedown', e => {
      e.preventDefault(); onSelect(d.dataset.token); hide();
    }));
  };
  const refresh = () => { items = suggestList(input.value.trim()); act = -1; draw(); };
  input.addEventListener('input', refresh);
  input.addEventListener('focus', refresh);
  input.addEventListener('keydown', e => {
    if (box.hidden || !items.length) return;
    if (e.key === 'ArrowDown') { act = (act + 1) % items.length; e.preventDefault(); draw(); }
    else if (e.key === 'ArrowUp') { act = (act - 1 + items.length) % items.length; e.preventDefault(); draw(); }
    else if (e.key === 'Enter' && act >= 0) { e.preventDefault(); onSelect(items[act]); hide(); }
    else if (e.key === 'Escape') hide();
  });
  input.addEventListener('blur', () => setTimeout(hide, 150));
}

/* ---------- 탭 ---------- */
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.panel').forEach(p => p.hidden = p.dataset.panel !== name);
}

/* ---------- 막대/칩 클릭 → 비슷한 단어 탭으로 이동 ---------- */
function delegateBars(boxId) {
  document.getElementById(boxId).addEventListener('click', e => {
    const row = e.target.closest('.bar-row[data-token]');
    if (!row) return;
    showTab('similar');
    document.getElementById('simInput').value = row.dataset.token;
    runSim(row.dataset.token);
    document.getElementById('simInput').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
}

/* ---------- 와이어링 ---------- */
function wire() {
  document.getElementById('tabs').addEventListener('click', e => {
    const b = e.target.closest('.tab'); if (b) showTab(b.dataset.tab);
  });

  // 예시 링크
  document.querySelectorAll('.ex').forEach(a => a.addEventListener('click', () => {
    const k = a.dataset.ex;
    if (k === 'compare') {
      showTab('compare');
      document.getElementById('cmpA').value = a.dataset.a;
      document.getElementById('cmpB').value = a.dataset.b; runCmp();
    } else if (k === 'spy') {
      showTab('spy'); spyTokens = a.dataset.list.split('·'); renderSpyChips(); runSpy();
    } else if (k === 'similar') {
      showTab('similar'); document.getElementById('simInput').value = a.dataset.a; runSim(a.dataset.a);
    }
  }));

  // 비교
  document.getElementById('cmpBtn').addEventListener('click', runCmp);
  document.getElementById('cmpA').addEventListener('keydown', e => { if (e.key === 'Enter') runCmp(); });
  document.getElementById('cmpB').addEventListener('keydown', e => { if (e.key === 'Enter') runCmp(); });
  attachSuggest('cmpA', 'cmpASuggest', t => { document.getElementById('cmpA').value = t; });
  attachSuggest('cmpB', 'cmpBSuggest', t => { document.getElementById('cmpB').value = t; });

  // 스파이
  document.getElementById('spyBtn').addEventListener('click', runSpy);
  document.getElementById('spyClear').addEventListener('click', () => {
    spyTokens = []; renderSpyChips(); document.getElementById('spyResult').innerHTML = '';
  });
  attachSuggest('spyInput', 'spyInputSuggest', t => { addSpyToken(t); document.getElementById('spyInput').value = ''; });

  // 비슷한 단어
  document.getElementById('simBtn').addEventListener('click', () => runSim(document.getElementById('simInput').value));
  document.getElementById('simInput').addEventListener('keydown', e => { if (e.key === 'Enter') runSim(e.target.value); });
  const tn = document.getElementById('simTopn');
  tn.addEventListener('input', () => document.getElementById('simTopnVal').textContent = tn.value);
  attachSuggest('simInput', 'simInputSuggest', t => { document.getElementById('simInput').value = t; runSim(t); });

  ['cmpResult', 'simResult'].forEach(delegateBars);
}

load().then(wire).catch(err => {
  const s = document.getElementById('loadStatus');
  if (s) s.textContent = '데이터 로드 실패: ' + err.message;
});
