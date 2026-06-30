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
  ws.onmessage = e=>{
    if(isReplay){
      const msg = JSON.parse(e.data);
      if(msg.type === 'episodes'){
        replayEpisodesRaw = msg.episodes || [];
        recomputeReplayView();
      } else if(msg.type === 'episode'){
        onReplayEpisodeLoaded(msg);
      } else if(msg.type === 'cursor'){
        onReplayCursor(msg);
      }
    } else {
      state = JSON.parse(e.data); selectedCards.clear(); render(state);
    }
  };
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
  renderHandHistory(s.hand_history);
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

/* ── Hand History ────────────────────────────────────────── */
const HH_MAX_RENDER = 60; // cap rendered rows for long replay episodes

function renderHandHistory(entries){
  entries = entries || [];
  const sec = $('hist'); sec.innerHTML = '';
  if(!entries.length) return;

  const hdr = mk('div', 'hh-header');
  const capped = entries.length > HH_MAX_RENDER;
  hdr.innerHTML = `Hand History <span style="color:#4a6a4a;font-size:10px;font-weight:400;letter-spacing:0;text-transform:none;">(${entries.length} played${capped ? `, showing last ${HH_MAX_RENDER}` : ''})</span>`;
  sec.appendChild(hdr);

  const scroll = mk('div', 'hh-scroll');
  // Show newest first, capped for performance on long episodes
  const shown = capped ? entries.slice(entries.length - HH_MAX_RENDER) : entries;
  [...shown].reverse().forEach(e => {
    const card = mk('div', 'hh-entry' + (e.debuffed ? ' debuffed' : ''));

    // Meta: A1·R1·H2
    const meta = mk('div', 'hh-meta');
    meta.innerHTML = `<span class="hh-tag">A${e.ante}</span><span class="hh-tag">R${e.round}</span><span class="hh-tag">H${e.hand_num}</span>`;
    card.appendChild(meta);

    // Hand type
    const ht = mk('div', 'hh-type');
    ht.textContent = e.debuffed ? `${e.hand_type} ✗` : e.hand_type;
    card.appendChild(ht);

    // Score
    const sc = mk('div', 'hh-score');
    sc.innerHTML = `<span class="hh-chips">${fmt(e.chips)}</span><span class="hh-dim">chips ×</span><span class="hh-mult">${fmt(e.mult)}</span><span class="hh-dim">mult =</span><span class="hh-total">${fmt(e.total)}</span>`;
    card.appendChild(sc);

    // Cards played
    if(e.cards?.length){
      const row = mk('div', 'hh-cards');
      e.cards.forEach(c => {
        const chip = mk('span', 'hh-card' + (c.red ? ' red' : '') + (c.scored ? ' scored' : '') + (c.destroyed ? ' destroyed' : ''));
        let label = c.label || c.name || c.key || '?';
        if(c.enhancement) label += `[${c.enhancement}]`;
        if(c.edition && c.edition !== 'base') label += `{${c.edition}}`;
        if(c.seal) label += `·${c.seal}`;
        chip.textContent = label;
        row.appendChild(chip);
      });
      card.appendChild(row);
    }

    // Active jokers
    if(e.jokers?.length){
      const jrow = mk('div', 'hh-jokers');
      e.jokers.forEach(j => {
        const jc = mk('span', 'hh-joker');
        jc.textContent = j.name || j.key || '?';
        jrow.appendChild(jc);
      });
      card.appendChild(jrow);
    }

    // Dollars earned
    if(e.dollars_earned > 0){
      const d = mk('div', 'hh-dollars');
      d.textContent = `+$${e.dollars_earned}`;
      card.appendChild(d);
    }

    if(e.debuffed){
      const db = mk('div', 'hh-debuffed');
      db.textContent = 'Debuffed by boss blind';
      card.appendChild(db);
    }

    scroll.appendChild(card);
  });
  sec.appendChild(scroll);
}

/* ── Log ─────────────────────────────────────────────────── */
function renderLog(s){
  const list = $('log-list'); list.innerHTML = '';
  const hist = s.history || [], last = s.last_action;
  const all = last && (hist.length===0 || hist[hist.length-1]!==last) ? [...hist, last] : hist;
  if(!all.length){ list.appendChild(mk('div','le empty','No actions yet')); return; }
  [...all].reverse().forEach((t, i) => list.appendChild(mk('div','le'+(i===0?' new':''),`→ ${esc(t)}`)));
}

/* ── Replay mode ─────────────────────────────────────────── */
let isReplay = false;
let replayEpisodesRaw = [];   // raw summaries from server, server order
let replayView = [];          // filtered + sorted summaries currently shown
let replayActiveEp = -1;      // "episode" number (not array index) of loaded episode
let replayTotalSteps = 0;
let replaySort = 'index';
let replayFilter = 'all';

// Cached data for the currently-loaded episode (sent once via 'episode' msg)
let replayCur = null;   // { action_log, hand_history, total_steps, episode_index, seed, won, ante_reached, rounds_beaten }
let replayRowEls = [];  // tl-row elements indexed by action_log step
let replayStepIdx = 0;

// Convert card label like "As", "10H", "Kd" → "A♠", "10♥", "K♦"
function prettyCard(label){
  if(!label) return '?';
  const sym = {S:'♠',H:'♥',D:'♦',C:'♣',s:'♠',h:'♥',d:'♦',c:'♣'};
  const last = label.slice(-1);
  return sym[last] ? label.slice(0,-1) + sym[last] : label;
}

function isRedCard(label){
  const last = (label||'').slice(-1);
  return last === 'H' || last === 'h' || last === 'D' || last === 'd' ||
         label.includes('♥') || label.includes('♦');
}

function tlChip(label){
  const d = mk('span', 'tl-chip' + (isRedCard(label) ? ' red' : ''));
  d.textContent = prettyCard(label);
  return d;
}

const EP_ROW_H = 54; // approx px height of one .ep-row incl. gap — used for virtualization

function sortEpisodes(eps, mode){
  const arr = eps.slice();
  switch(mode){
    case 'reward_desc': arr.sort((a,b)=>(b.total_reward||0)-(a.total_reward||0)); break;
    case 'reward_asc':  arr.sort((a,b)=>(a.total_reward||0)-(b.total_reward||0)); break;
    case 'ante_desc':   arr.sort((a,b)=>(b.ante_reached||0)-(a.ante_reached||0)); break;
    case 'rounds_desc': arr.sort((a,b)=>(b.rounds_beaten||0)-(a.rounds_beaten||0)); break;
    case 'steps_desc':  arr.sort((a,b)=>(b.steps||0)-(a.steps||0)); break;
    case 'steps_asc':   arr.sort((a,b)=>(a.steps||0)-(b.steps||0)); break;
    case 'index_desc':  arr.sort((a,b)=>b.index-a.index); break;
    default:             arr.sort((a,b)=>a.index-b.index); // 'index'
  }
  return arr;
}

function filterEpisodes(eps, mode){
  if(mode === 'won')  return eps.filter(e => e.won);
  if(mode === 'lost') return eps.filter(e => !e.won);
  return eps;
}

function recomputeReplayView(){
  replayView = sortEpisodes(filterEpisodes(replayEpisodesRaw, replayFilter), replaySort);
  renderReplaySidebar();
}

function buildEpRow(ep){
  const rew = ep.total_reward || 0;
  const row = mk('div', 'ep-row' + (ep.won ? ' won' : '') + (ep.episode === replayActiveEp ? ' active' : ''));
  row.title = `Seed: ${ep.seed||'?'} · ${ep.hands_played||0} hands played`;
  row.innerHTML = `
    <div class="ep-top">
      <span class="ep-num">#${ep.episode}</span>
      <span class="ep-badge">A${ep.ante_reached}·R${ep.rounds_beaten}</span>
      <span class="ep-status ${ep.won ? 'win' : ''}">${ep.won ? 'WIN' : '—'}</span>
    </div>
    <div class="ep-bot">
      <span class="ep-rew ${rew >= 0 ? 'pos' : 'neg'}">${rew >= 0 ? '+' : ''}${rew.toFixed(1)}</span>
      <span class="ep-steps">${fmt(ep.steps)}s · ${fmt(ep.hands_played)}h</span>
    </div>
  `;
  row.onclick = () => send({cmd:'load', episode: ep.index});
  return row;
}

// One-time-built sidebar shell; rows are virtualized on scroll so DOM size
// stays small regardless of how many episodes were recorded.
function renderReplaySidebar(){
  let side = $('side');
  if(!side.querySelector('.ep-panel')){
    side.innerHTML = '';
    const panel = mk('div', 'panel ep-panel');
    panel.appendChild(mk('div', 'ph', '<span>📺 Episodes</span><span class="ph-right" id="ep-count" style="color:#9dd89d"></span>'));

    const controls = mk('div', 'ep-controls');
    controls.innerHTML = `
      <select id="ep-sort" class="ep-sort-sel">
        <option value="index">Episode #</option>
        <option value="index_desc">Episode # ↓</option>
        <option value="reward_desc">Reward ↓</option>
        <option value="reward_asc">Reward ↑</option>
        <option value="ante_desc">Ante reached ↓</option>
        <option value="rounds_desc">Rounds beaten ↓</option>
        <option value="steps_desc">Steps ↓</option>
        <option value="steps_asc">Steps ↑</option>
      </select>
      <div class="ep-filter-row">
        <button class="ep-filter-btn active" data-f="all">All</button>
        <button class="ep-filter-btn" data-f="won">Won</button>
        <button class="ep-filter-btn" data-f="lost">Lost</button>
      </div>
    `;
    panel.appendChild(controls);

    const epListScroll = mk('div', 'ep-list');
    epListScroll.id = 'ep-list-scroll';
    panel.appendChild(epListScroll);
    side.appendChild(panel);

    controls.querySelector('#ep-sort').value = replaySort;
    controls.querySelector('#ep-sort').onchange = (e) => { replaySort = e.target.value; recomputeReplayView(); };
    controls.querySelectorAll('.ep-filter-btn').forEach(btn => {
      btn.onclick = () => {
        replayFilter = btn.dataset.f;
        controls.querySelectorAll('.ep-filter-btn').forEach(b => b.classList.toggle('active', b===btn));
        recomputeReplayView();
      };
    });
  }

  $('ep-count').textContent = `${replayView.length}${replayView.length !== replayEpisodesRaw.length ? ` / ${replayEpisodesRaw.length}` : ''}`;
  renderEpisodeListVirtual();
}

// Virtual-scroll the episode list: only DOM nodes for rows in (or near) the
// viewport are created, so the list stays fast with thousands of episodes.
let _epVirtualHandlersBound = false;
function renderEpisodeListVirtual(){
  const epList = $('ep-list-scroll');
  if(!epList) return;

  let spacerTop = epList.querySelector('.ep-spacer-top');
  let rowsContainer = epList.querySelector('.ep-rows');
  let spacerBot = epList.querySelector('.ep-spacer-bot');
  if(!rowsContainer){
    epList.innerHTML = '';
    spacerTop = mk('div', 'ep-spacer-top');
    rowsContainer = mk('div', 'ep-rows');
    spacerBot = mk('div', 'ep-spacer-bot');
    epList.appendChild(spacerTop);
    epList.appendChild(rowsContainer);
    epList.appendChild(spacerBot);
  }

  if(!replayView.length){
    spacerTop.style.height = '0px'; spacerBot.style.height = '0px';
    rowsContainer.innerHTML = '';
    rowsContainer.appendChild(mk('span', 'empty', 'No episodes match filter'));
    return;
  }

  function update(){
    const total = replayView.length;
    const scrollTop = epList.scrollTop;
    const viewH = epList.clientHeight || 400;
    const buffer = 6;
    const start = Math.max(0, Math.floor(scrollTop / EP_ROW_H) - buffer);
    const end = Math.min(total, Math.ceil((scrollTop + viewH) / EP_ROW_H) + buffer);
    spacerTop.style.height = (start * EP_ROW_H) + 'px';
    spacerBot.style.height = Math.max(0, (total - end) * EP_ROW_H) + 'px';
    rowsContainer.innerHTML = '';
    for(let i = start; i < end; i++) rowsContainer.appendChild(buildEpRow(replayView[i]));
  }

  if(!_epVirtualHandlersBound){
    epList.addEventListener('scroll', () => renderEpisodeListVirtual());
    _epVirtualHandlersBound = true;
  }
  update();

  // Scroll the active episode into view the first time it loads
  if(replayActiveEp >= 0 && !epList.dataset.scrolledTo){
    const idx = replayView.findIndex(e => e.episode === replayActiveEp);
    if(idx >= 0){
      epList.scrollTop = Math.max(0, idx * EP_ROW_H - viewHalf(epList));
      epList.dataset.scrolledTo = '1';
      update();
    }
  }
}
function viewHalf(el){ return (el.clientHeight || 400) / 2; }

function renderReplayTimeline(ep, stepIdx){
  const m = $('main'); m.innerHTML = '';
  const log = ep.action_log || [];
  replayRowEls = new Array(log.length);

  if(!log.length){
    const p = mk('div', 'panel');
    p.innerHTML = '<div class="ph">Timeline</div>';
    const b = mk('div', 'pb');
    b.appendChild(mk('span', 'empty', 'No actions recorded — re-run training with --record-dir'));
    p.appendChild(b); m.appendChild(p);
    return;
  }

  // Group consecutive actions by (ante, round)
  const groups = [];
  let curGroup = null;
  log.forEach((a, i) => {
    const key = `${a.ante}-${a.round}`;
    if(!curGroup || curGroup.key !== key){
      curGroup = {key, ante: a.ante, round: a.round, items: []};
      groups.push(curGroup);
    }
    curGroup.items.push({...a, _idx: i});
  });

  let scrollTarget = null;
  const container = mk('div', 'tl-container');

  groups.forEach(g => {
    const grp = mk('div', 'tl-group');

    const head = mk('div', 'tl-group-head');
    head.innerHTML = `<span class="tl-gh-ante">Ante ${g.ante}</span><span class="tl-gh-sep">·</span><span class="tl-gh-round">Round ${g.round}</span>`;
    grp.appendChild(head);

    g.items.forEach(a => {
      const isActive = a._idx === stepIdx;
      const row = mk('div', 'tl-row' + (isActive ? ' active' : ''));
      replayRowEls[a._idx] = row;
      if(isActive) scrollTarget = row;

      row.appendChild(mk('span', `tl-type tl-${a.type}`, a.type.toUpperCase()));

      const detail = mk('div', 'tl-detail');

      if(a.type === 'play'){
        const top = mk('div', 'tl-top-line');
        top.innerHTML = `<span class="tl-hand">${esc(a.hand_type||'?')}</span><span class="tl-pts">${fmt(a.total)} pts</span>`;
        detail.appendChild(top);
        // Cumulative chips progress toward blind
        if(a.chips_total != null){
          const bar = mk('div', 'tl-score-bar');
          const pct = a.blind_chips ? Math.min(100, (a.chips_total / a.blind_chips) * 100) : 0;
          const won = a.chips_total >= (a.blind_chips || Infinity);
          bar.innerHTML = `<span class="tl-score-val ${won?'won':''}">${fmt(a.chips_total)}</span>`
            + (a.blind_chips ? `<span class="tl-score-sep">/</span><span class="tl-score-blind">${fmt(a.blind_chips)}</span>` : '')
            + (a.blind_chips ? `<span class="tl-score-track"><span class="tl-score-fill ${won?'won':''}" style="width:${pct.toFixed(1)}%"></span></span>` : '');
          detail.appendChild(bar);
        }
        if(a.cards?.length){
          const cr = mk('div', 'tl-cards');
          a.cards.forEach(c => cr.appendChild(tlChip(c)));
          detail.appendChild(cr);
        }
        if(a.jokers?.length){
          const jr = mk('div', 'tl-jokers');
          a.jokers.forEach(j => { const jc = mk('span','tl-jchip'); jc.textContent = j; jr.appendChild(jc); });
          detail.appendChild(jr);
        }
      } else if(a.type === 'skip'){
        const top = mk('div', 'tl-top-line');
        top.innerHTML = `<span class="tl-skip-lbl">${esc(a.blind||'?')} Blind</span><span class="tl-skip-arrow">→</span><span class="tl-skip-next">${esc(a.next||'?')} Blind</span>`;
        detail.appendChild(top);
      } else if(a.type === 'discard'){
        const top = mk('div', 'tl-top-line');
        top.appendChild(mk('span', 'tl-discard-lbl', `${(a.cards||[]).length} card${(a.cards||[]).length!==1?'s':''}`));
        detail.appendChild(top);
        if(a.cards?.length){
          const cr = mk('div', 'tl-cards');
          a.cards.forEach(c => cr.appendChild(tlChip(c)));
          detail.appendChild(cr);
        }
      } else if(a.type === 'buy'){
        const top = mk('div', 'tl-top-line');
        top.innerHTML = `<span class="tl-item">${esc(a.card||'?')}</span><span class="tl-cost">$${a.cost||0}</span>`;
        detail.appendChild(top);
      } else if(a.type === 'sell'){
        const top = mk('div', 'tl-top-line');
        top.innerHTML = `<span class="tl-item">${esc(a.card||'?')}</span><span class="tl-gold">+$${a.gold||0}</span>`;
        detail.appendChild(top);
      } else if(a.type === 'blind'){
        const top = mk('div', 'tl-top-line');
        top.innerHTML = `<span class="tl-bname">${esc(a.name||'?')}</span><span class="tl-bchips">${fmt(a.chips)} chips</span>`;
        detail.appendChild(top);
      } else {
        detail.textContent = a.type;
      }

      row.appendChild(detail);

      if(a.dollars != null){
        row.appendChild(mk('span', 'tl-dol', '$'+a.dollars));
      }

      row.onclick = () => send({cmd:'step', index: a._idx});
      grp.appendChild(row);
    });

    container.appendChild(grp);
  });

  m.appendChild(container);

  if(scrollTarget){
    requestAnimationFrame(() => scrollTarget.scrollIntoView({block:'nearest', behavior:'smooth'}));
  }
}

function renderReplayNav(total, idx){
  const row = $('acts-row'); row.innerHTML = '';
  $('acts-label').textContent = 'Navigation';
  $('watch-note').classList.remove('show');
  replayTotalSteps = total;

  const nav = mk('div', 'replay-nav');

  const prev = mk('button', 'btn', '◀ Prev');
  prev.title = 'Previous step (← arrow key)';
  prev.disabled = idx === 0;
  prev.onclick = () => send({cmd:'prev'});

  const slider = document.createElement('input');
  slider.type = 'range'; slider.min = 0; slider.max = Math.max(0, total-1);
  slider.value = idx; slider.className = 'rn-slider';
  slider.oninput = () => send({cmd:'step', index: parseInt(slider.value)});

  const next = mk('button', 'btn success', 'Next ▶');
  next.title = 'Next step (→ arrow key)';
  next.disabled = idx >= total - 1;
  next.onclick = () => send({cmd:'next'});

  const lbl = mk('div', 'rn-step', `${fmt(idx + 1)} / ${fmt(total)}`);

  nav.appendChild(prev);
  nav.appendChild(slider);
  nav.appendChild(next);
  nav.appendChild(lbl);
  row.appendChild(nav);
}

// Full episode payload arrives once per 'load' — cache it, rebuild the
// timeline once, and refresh the sidebar's active-row highlighting.
function onReplayEpisodeLoaded(ep){
  replayCur = ep;
  replayActiveEp = ep.episode_index ?? 0;
  replayStepIdx = ep.step_index ?? 0;

  const epList = $('ep-list-scroll');
  if(epList) delete epList.dataset.scrolledTo;
  renderReplaySidebar();

  renderHeaderForStep();
  renderReplayTimeline(ep, replayStepIdx);
  updateHandHistoryForStep();
  renderReplayNav(ep.total_steps || 0, replayStepIdx);
}

// Lightweight cursor arrives on every step/next/prev — no full rebuild,
// just move the active-row highlight and patch the few text fields that
// depend on step position.
function onReplayCursor(c){
  if(!replayCur) return;
  replayStepIdx = c.step_index ?? 0;

  const prevActive = document.querySelector('#main .tl-row.active');
  if(prevActive) prevActive.classList.remove('active');
  const newActive = replayRowEls[replayStepIdx];
  if(newActive){
    newActive.classList.add('active');
    newActive.scrollIntoView({block:'nearest', behavior:'smooth'});
  }

  renderHeaderForStep();
  updateHandHistoryForStep();
  renderReplayNav(c.total_steps || replayCur.total_steps || 0, replayStepIdx);
}

function renderHeaderForStep(){
  const ep = replayCur; if(!ep) return;
  $('h-phase').textContent = 'Replay';
  $('h-ante').textContent  = `A${ep.ante_reached||0} · ${ep.rounds_beaten||0} rounds${ep.won?' ✓':''}`;
  $('h-money').textContent = `Ep ${ep.episode_index??0}`;
  $('h-hands').textContent = `${visibleHandHistory().length} hands`;
  $('h-disc').textContent  = `${ep.total_steps||0} steps`;
  $('h-won').style.display = ep.won ? '' : 'none';

  const pct = ep.total_steps ? replayStepIdx / ep.total_steps : 0;
  $('bar').style.width   = (pct*100).toFixed(1) + '%';
  $('s-cur').textContent = `Step ${replayStepIdx+1}`;
  $('s-max').textContent = `${ep.total_steps||0}`;
  $('s-pct').textContent = ep.won ? '🏆 Won' : `${visibleHandHistory().length} hands`;
  $('heval').classList.remove('visible');
}

// Hand history visible "so far" is derived client-side from the cached
// full hand_history (tagged with _step at load time) — no re-fetch needed.
function visibleHandHistory(){
  if(!replayCur) return [];
  return (replayCur.hand_history || []).filter(h => h._step <= replayStepIdx);
}

function updateHandHistoryForStep(){
  renderHandHistory(visibleHandHistory());
}

/* ── Init ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', ()=>{
  const mode = new URLSearchParams(location.search).get('mode');
  isPlay   = mode === 'play';
  isReplay = mode === 'replay';

  if(isReplay){
    $('h-mode').textContent = '⏪ Replay';
    document.title = 'Jackdaw — Replay';
    $('watch-note').style.display = 'none';
    $('app').classList.add('replay-mode');
    // Keyboard navigation
    document.addEventListener('keydown', e => {
      if(!isReplay) return;
      if(e.key === 'ArrowLeft')  { e.preventDefault(); send({cmd:'prev'}); }
      if(e.key === 'ArrowRight') { e.preventDefault(); send({cmd:'next'}); }
    });
    connect();
  } else {
    $('h-mode').textContent = isPlay ? '▶ Play' : '👁 Watch';
    document.title = isPlay ? 'Jackdaw — Play' : 'Jackdaw — Watch';
    connect();
  }
});