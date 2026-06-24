'use strict';

let state = null, ws = null, reconnT = null;
let selectedCards = new Set();
let isPlay = false;

const PHASE_LABELS = {
  blind_select:'Blind Select', selecting_hand:'Playing', round_eval:'Round Complete',
  shop:'Shop', pack_opening:'Pack Opening', game_over:'Game Over'
};
const TYPE_ICONS = { joker:'🃏', tarot:'🌙', planet:'🪐', spectral:'👻', voucher:'🎟', booster:'📦' };

function $(id){ return document.getElementById(id); }
function mk(tag,cls,html){ const e=document.createElement(tag); if(cls)e.className=cls; if(html!==undefined)e.innerHTML=html; return e; }
function esc(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmt(n){ return n==null?'—':Number(n).toLocaleString(); }

/* ── WebSocket ───────────────────────────────────────────── */
function connect(){
  // WS_PORT is injected into index.html by the Python server at request time
  const wsPort = window.WS_PORT || (parseInt(location.port||'8765') + 1);
  ws = new WebSocket(`ws://${location.hostname}:${wsPort}`);
  ws.onopen = ()=>{ $('conn').className='ok'; $('connecting').style.display='none'; clearTimeout(reconnT); };
  ws.onmessage = e=>{ state=JSON.parse(e.data); selectedCards.clear(); render(state); };
  ws.onclose = ()=>{ $('conn').className='err'; reconnT=setTimeout(connect,1500); };
  ws.onerror = ()=>{ $('conn').className='err'; };
}
function send(data){ ws&&ws.readyState===1&&ws.send(JSON.stringify(data)); }

/* ── Card builders ───────────────────────────────────────── */
function playingCard(c, idx, sel){
  const d = mk('div', 'card ' + (c.red ? 'red' : 'black'));
  if(c.debuff) d.classList.add('debuffed');
  if(c.seal)   d.classList.add('seal-' + c.seal.toLowerCase());
  if(c.edition){
    const entry = Object.entries(c.edition).find(([,v])=>v);
    if(entry) d.classList.add(entry[0]);
  }
  d.innerHTML = `<span class="cr">${esc(c.rank_short)}</span><span class="cs">${esc(c.suit_symbol)}</span><span class="cb">${esc(c.rank_short)}</span>${c.enhancement ? `<span class="cenh">${esc(c.enhancement)}</span>` : ''}`;
  if(sel){
    d.classList.add('selectable');
    if(selectedCards.has(idx)) d.classList.add('selected');
    d.onclick = ()=>{
      selectedCards.has(idx) ? selectedCards.delete(idx) : selectedCards.add(idx);
      renderActions(state);
      document.querySelectorAll('#hand-body .card').forEach((c2,i2)=>c2.classList.toggle('selected', selectedCards.has(i2)));
    };
  }
  return d;
}

function jokerCard(c){
  const d = mk('div', 'jcard');
  const desc = (c.desc || '');
  const shortDesc = desc.length > 72 ? desc.slice(0, 69) + '…' : desc;
  d.innerHTML = `
    <span class="ji">${TYPE_ICONS.joker}</span>
    <span class="jn">${esc(c.name || c.key || '?')}</span>
    <span class="jd">${esc(shortDesc)}</span>
    <span class="jc">$${c.cost || 0}</span>
    ${c.eternal    ? '<span class="be" title="Eternal" style="color:#88ccff">∞</span>' : ''}
    ${c.perishable ? '<span class="be" title="Perishable" style="color:#ff8866">🍂</span>' : ''}
    ${c.rental     ? '<span class="bl" title="Rental">$</span>' : ''}
  `;
  return d;
}

function itemCard(c, type){
  const t = (type || c.type || c.set || '').toLowerCase();
  const icon = TYPE_ICONS[t] || '🂠';
  const d = mk('div', 'icard ' + (t || ''));
  d.innerHTML = `
    <span class="ii">${icon}</span>
    <span class="in">${esc((c.name || c.key || '?').slice(0, 16))}</span>
    <span class="it">${esc(t)}</span>
    ${c.cost != null ? `<span class="ic">$${c.cost}</span>` : ''}
  `;
  return d;
}

/* ── Main render ─────────────────────────────────────────── */
function render(s){
  renderHeader(s);
  renderScore(s);
  renderSidebar(s);
  renderMain(s);
  renderActions(s);
  renderLog(s);
}

function renderHeader(s){
  $('h-phase').textContent  = PHASE_LABELS[s.phase] || s.phase || '—';
  $('h-ante').textContent   = `Ante ${s.ante ?? '—'} · ${s.blind_on_deck || '—'}`;
  $('h-money').textContent  = `$${fmt(s.dollars)}`;
  $('h-hands').textContent  = `${s.hands_left ?? '—'} ♠`;
  $('h-disc').textContent   = `${s.discards_left ?? '—'} ↩`;
  $('h-won').style.display  = s.won ? '' : 'none';
}

function renderScore(s){
  const pct = Math.min(s.chips_progress || 0, 1);
  $('bar').style.width    = (pct * 100).toFixed(1) + '%';
  $('s-cur').textContent  = fmt(s.chips);
  $('s-max').textContent  = fmt(s.blind_chips);
  $('s-pct').textContent  = (pct * 100).toFixed(0) + '%';
  const h = s.last_hand, heval = $('heval');
  if(h && h.name){
    heval.classList.add('visible');
    $('he-name').textContent  = h.name;
    $('he-chips').textContent = `${fmt(h.chips)} chips`;
    $('he-mult').textContent  = `× ${h.mult} mult`;
    $('he-score').textContent = s.last_score ? `= ${fmt(s.last_score)}` : '';
  } else {
    heval.classList.remove('visible');
  }
}

function renderSidebar(s){
  const jb = $('joker-body'); jb.innerHTML = '';
  $('joker-count').textContent = s.jokers?.length ? `${s.jokers.length}` : '';
  if(s.jokers?.length) s.jokers.forEach(j => jb.appendChild(jokerCard(j)));
  else jb.appendChild(mk('span', 'empty', 'No jokers yet'));

  const cb = $('cons-body'); cb.innerHTML = '';
  if(s.consumables?.length) s.consumables.forEach(c => cb.appendChild(itemCard(c)));
  else cb.appendChild(mk('span', 'empty', 'None'));
}

function renderMain(s){
  const m = $('main'); m.innerHTML = '';
  switch(s.phase){
    case 'selecting_hand': renderHand(s, m);       break;
    case 'blind_select':   renderBlindSelect(s, m); break;
    case 'shop':           renderShop(s, m);        break;
    case 'pack_opening':   renderPack(s, m);        break;
    case 'round_eval':     renderRoundEval(s, m);   break;
    case 'game_over':      renderGameOver(s, m);    break;
    default: if(s.hand?.length) renderHand(s, m);
  }
}

function renderHand(s, c){
  const p = mk('div', 'panel'); p.id = 'hand-panel';
  p.innerHTML = `<div class="ph">Your Hand <span class="ph-right">Deck: ${s.deck_remaining ?? '—'} · Discard pile: ${s.discard_pile ?? '—'}</span></div>`;
  const b = mk('div', 'pb'); b.id = 'hand-body'; b.style.minHeight = '100px';
  const canSel = isPlay && (s.legal_actions || []).some(a => a.type === 'PlayHand' || a.type === 'Discard');
  if(s.hand?.length) s.hand.forEach((card, i) => b.appendChild(playingCard(card, i, canSel)));
  else b.appendChild(mk('span', 'empty', 'No cards in hand'));
  p.appendChild(b);

  // Hands played this round
  const played = s.hands_played_round || [];
  const hp = mk('div', 'hands-played');
  hp.innerHTML = '<span class="hp-label">This round:</span> ';
  if(played.length === 0){
    hp.innerHTML += '<span class="hp-none">No hands played yet</span>';
  } else {
    played.forEach(h => {
      const badge = mk('span', 'hp-badge');
      badge.textContent = h.count > 1 ? `${h.name} ×${h.count}` : h.name;
      hp.appendChild(badge);
    });
  }
  p.appendChild(hp);
  c.appendChild(p);
}

function renderBlindSelect(s, c){
  const p = mk('div', 'panel');
  const blind = s.blind_on_deck || 'Small';
  const chips = s.blind_chips ? `${fmt(s.blind_chips)} chips` : '—';
  p.innerHTML = `<div class="ph">Blind Select</div>
    <div class="blind-screen">
      <div class="bs-blind">${esc(blind)} Blind</div>
      <div class="bs-target">🎯 ${chips}</div>
      <div class="bs-sub">Ante ${s.ante}</div>
    </div>`;
  c.appendChild(p);
}

function renderShop(s, c){
  const p = mk('div', 'panel');
  p.innerHTML = `<div class="ph">Shop <span class="ph-right">$${s.dollars} available</span></div>`;
  const b = mk('div', 'pb col'); b.style.gap = '12px';

  function shopSection(label, cards, actionType, btnLabel){
    if(!cards?.length) return;
    const sec = mk('div', 'shop-sec');
    sec.appendChild(mk('div', 'shop-label', label));
    const row = mk('div', 'shop-row');
    cards.forEach((card, i) => {
      const w = mk('div', 'shop-item');
      w.appendChild(card.type === 'playing' ? playingCard(card, i, false) : itemCard(card));
      const legal = (s.legal_actions || []).find(a => a.type === actionType && (a.shop_index === i || a.card_index === i));
      if(isPlay && legal){ const btn = mk('button', 'btn success', btnLabel(card)); btn.onclick = ()=>send(legal); w.appendChild(btn); }
      row.appendChild(w);
    });
    sec.appendChild(row); b.appendChild(sec);
  }

  shopSection('For Sale',      s.shop_cards,    'BuyCard',       c => `Buy $${c.cost}`);
  shopSection('Vouchers',      s.shop_vouchers, 'RedeemVoucher', c => `Redeem $${c.cost}`);
  shopSection('Booster Packs', s.shop_boosters, 'OpenBooster',   c => `Open $${c.cost}`);
  if(!s.shop_cards?.length && !s.shop_vouchers?.length && !s.shop_boosters?.length)
    b.appendChild(mk('span', 'empty', 'Shop is empty'));
  p.appendChild(b); c.appendChild(p);
}

function renderPack(s, c){
  const remaining = s.pack_choices_remaining || 0;
  const p = mk('div', 'panel');
  p.innerHTML = `<div class="ph">${esc(s.pack_type || 'Pack')} <span class="ph-right">${remaining} pick${remaining !== 1 ? 's' : ''} remaining</span></div>`;
  const b = mk('div', 'pb');
  (s.pack_cards || []).forEach((card, i) => {
    const w = mk('div', 'shop-item');
    w.appendChild(card.type === 'playing' ? playingCard(card, i, false) : itemCard(card));
    const legal = (s.legal_actions || []).find(a => a.type === 'PickPackCard' && a.card_index === i);
    if(isPlay && legal){ const btn = mk('button', 'btn primary', 'Pick'); btn.onclick = ()=>send(legal); w.appendChild(btn); }
    b.appendChild(w);
  });
  if(!s.pack_cards?.length) b.appendChild(mk('span', 'empty', 'No cards'));
  p.appendChild(b); c.appendChild(p);
}

function renderRoundEval(s, c){
  const p = mk('div', 'panel');
  p.innerHTML = `<div class="ph">Round Complete</div>
    <div class="phase-screen">
      <div class="ps-big" style="color:#3ddc84">✓ Blind Defeated!</div>
      <div class="ps-sub">Collect earnings and head to the shop</div>
    </div>`;
  c.appendChild(p);
}

function renderGameOver(s, c){
  const p = mk('div', 'panel');
  if(s.won)
    p.innerHTML = `<div class="ph">Run Complete</div><div class="phase-screen"><div class="ps-big" style="color:#f0c040">🏆 You Won!</div><div class="ps-sub">Ante ${s.ante} · ${s.round} rounds</div></div>`;
  else
    p.innerHTML = `<div class="ph">Game Over</div><div class="phase-screen"><div class="ps-big" style="color:#ff5a5a">💀 Game Over</div><div class="ps-sub">Reached Ante ${s.ante} · Round ${s.round}</div></div>`;
  c.appendChild(p);
}

/* ── Actions ─────────────────────────────────────────────── */
function renderActions(s){
  const row = $('acts-row'), note = $('watch-note'), label = $('acts-label');
  row.innerHTML = '';
  if(!isPlay){ note.classList.add('show'); label.textContent = 'Watch Mode'; return; }
  note.classList.remove('show'); label.textContent = 'Your Turn';

  const legal = s.legal_actions || [];
  if(!legal.length){ row.appendChild(mk('span', 'empty', 'Waiting…')); return; }

  if(s.phase === 'selecting_hand')     buildHandActions(s, row, legal);
  else if(s.phase === 'shop')          buildShopActions(s, row, legal);
  else {
    legal.forEach(a => {
      if(a.type.startsWith('Swap')) return;
      const btn = mk('button', 'btn ' + btnCls(a.type), esc(a.label || a.type));
      btn.onclick = ()=>send(a); row.appendChild(btn);
    });
  }
}

function buildHandActions(s, row, legal){
  const canPlay = legal.find(a=>a.type==='PlayHand'), canDiscard = legal.find(a=>a.type==='Discard');
  const sel = [...selectedCards].sort((a,b)=>a-b);
  if(canPlay){
    const btn = mk('button', 'btn primary' + (sel.length ? '' : ' off'), `▶ Play${sel.length ? ` (${sel.length})` : ''}`);
    if(sel.length) btn.onclick = ()=>send({type:'PlayHand', card_indices:sel});
    row.appendChild(btn);
  }
  if(canDiscard){
    const btn = mk('button', 'btn danger' + (sel.length ? '' : ' off'), `↩ Discard${sel.length ? ` (${sel.length})` : ''}`);
    if(sel.length) btn.onclick = ()=>send({type:'Discard', card_indices:sel});
    row.appendChild(btn);
  }
  if(canPlay || canDiscard) row.appendChild(mk('div', 'btn-div'));
  ['rank','suit'].forEach(m=>{
    const a = legal.find(x=>x.type==='SortHand'&&x.mode===m);
    if(a){ const b=mk('button','btn','↕ '+m[0].toUpperCase()+m.slice(1)); b.onclick=()=>send(a); row.appendChild(b); }
  });
  legal.filter(a=>a.type==='UseConsumable').forEach(a=>{
    const con = (s.consumables||[])[a.card_index];
    const b = mk('button','btn gold','Use '+(con?con.name:'?'));
    b.onclick=()=>send(a); row.appendChild(b);
  });
}

function buildShopActions(s, row, legal){
  const next   = legal.find(a=>a.type==='NextRound');
  const reroll = legal.find(a=>a.type==='Reroll');
  const sells  = legal.filter(a=>a.type==='SellCard');
  if(next)  { const b=mk('button','btn success','▶ Next Round'); b.onclick=()=>send(next); row.appendChild(b); }
  if(reroll){ const free=s.free_rerolls||0; const b=mk('button','btn',free?`🔄 Reroll (Free ×${free})`:`🔄 Reroll $${s.reroll_cost}`); b.onclick=()=>send(reroll); row.appendChild(b); }
  if(sells.length){ row.appendChild(mk('div','btn-div')); }
  sells.forEach(a=>{
    const area = a.area==='jokers' ? s.jokers : s.consumables;
    const card = (area||[])[a.card_index];
    const b = mk('button','btn danger',`Sell ${card?card.name:'?'}`);
    b.onclick=()=>send(a); row.appendChild(b);
  });
}

function btnCls(t){
  return ({SelectBlind:'gold',CashOut:'success',NextRound:'success',SellCard:'danger'})[t]||'';
}

/* ── Log ─────────────────────────────────────────────────── */
function renderLog(s){
  const list = $('log-list'); list.innerHTML = '';
  const hist = s.history || [], last = s.last_action;
  const all = last && (hist.length===0 || hist[hist.length-1]!==last) ? [...hist, last] : hist;
  if(!all.length){ list.appendChild(mk('div','le empty','No actions yet')); return; }
  [...all].reverse().forEach((t, i) => list.appendChild(mk('div','le'+(i===0?' new':''),`→ ${esc(t)}`)));
}

/* ── Init ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', ()=>{
  isPlay = new URLSearchParams(location.search).get('mode') === 'play';
  $('h-mode').textContent = isPlay ? '▶ Play' : '👁 Watch';
  document.title = isPlay ? 'Jackdaw — Play' : 'Jackdaw — Watch';
  connect();
});
