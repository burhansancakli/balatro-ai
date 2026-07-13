'use strict';

let ws = null, reconnT = null;

const PHASE_LABELS = {
  blind_select:'Blind Select', selecting_hand:'Playing', round_eval:'Round Complete',
  shop:'Shop', pack_opening:'Pack Opening', game_over:'Game Over'
};

function $(id){ return document.getElementById(id); }
function mk(tag,cls,html){ const e=document.createElement(tag); if(cls)e.className=cls; if(html!==undefined)e.innerHTML=html; return e; }
function esc(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmt(n){ return n==null?'—':Number(n).toLocaleString(); }

/* ── WebSocket ───────────────────────────────────────────── */
function connect(){
  const wsPort = window.WS_PORT || (parseInt(location.port||'8765') + 1);
  ws = new WebSocket(`ws://${location.hostname}:${wsPort}`);
  ws.onopen = ()=>{ $('conn').className='ok'; $('connecting').style.display='none'; clearTimeout(reconnT); };
  ws.onmessage = e=>{
    const msg = JSON.parse(e.data);
    if(msg.type === 'episodes'){
      replayEpisodesRaw = msg.episodes || [];
      recomputeReplayView();
    } else if(msg.type === 'episode'){
      onReplayEpisodeLoaded(msg);
    } else if(msg.type === 'cursor'){
      onReplayCursor(msg);
    }
  };
  ws.onclose = ()=>{ $('conn').className='err'; reconnT=setTimeout(connect,1500); };
  ws.onerror = ()=>{ $('conn').className='err'; };
}
function send(data){ ws&&ws.readyState===1&&ws.send(JSON.stringify(data)); }

/* ── Replay mode ─────────────────────────────────────────── */
let replayEpisodesRaw = [];
let replayView = [];
let replayActiveEp = -1;
let replayTotalSteps = 0;
let replaySort = 'index';
let replayFilter = 'all';

let replayCur = null;
let replayRowEls = [];
let replayStepIdx = 0;

function prettyCard(label){
  if(!label) return '?';
  const sym = {S:'♠',H:'♥',D:'♦',C:'♣',s:'♠',h:'♥',d:'♦',c:'♣'};
  const last = label.slice(-1);
  return sym[last] ? label.slice(0,-1) + sym[last] : label;
}

function isRedCard(label){
  const last = (label||'').slice(-1);
  return last==='H'||last==='h'||last==='D'||last==='d'||
         label.includes('♥')||label.includes('♦');
}

function tlChip(label){
  const d = mk('span', 'tl-chip' + (isRedCard(label) ? ' red' : ''));
  d.textContent = prettyCard(label);
  return d;
}

// "FLUSH_BUILD" / "Flush Build" → "FLUSH"
function prettyStrat(s){
  return String(s||'').split(/[_\s]/)[0].toUpperCase();
}

function stratChip(strategy){
  const s = prettyStrat(strategy);
  const d = mk('span', 'tl-strat strat-' + s.toLowerCase());
  d.textContent = s;
  d.title = 'Declared strategy: ' + strategy;
  return d;
}

const EP_ROW_H = 54;

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
    default:             arr.sort((a,b)=>a.index-b.index);
  }
  return arr;
}

function filterEpisodes(eps, mode){
  if(mode==='won')  return eps.filter(e=>e.won);
  if(mode==='lost') return eps.filter(e=>!e.won);
  return eps;
}

function recomputeReplayView(){
  replayView = sortEpisodes(filterEpisodes(replayEpisodesRaw, replayFilter), replaySort);
  renderReplaySidebar();
}

function buildEpRow(ep){
  const rew = ep.total_reward || 0;
  const row = mk('div', 'ep-row' + (ep.won?' won':'') + (ep.episode===replayActiveEp?' active':''));
  row.title = `Seed: ${ep.seed||'?'} · ${ep.hands_played||0} hands played`;
  row.innerHTML = `
    <div class="ep-top">
      <span class="ep-num">#${ep.episode}</span>
      <span class="ep-badge">A${ep.ante_reached}·R${ep.rounds_beaten}</span>
      <span class="ep-status ${ep.won?'win':''}">${ep.won?'WIN':'—'}</span>
    </div>
    <div class="ep-bot">
      <span class="ep-rew ${rew>=0?'pos':'neg'}">${rew>=0?'+':''}${rew.toFixed(1)}</span>
      <span class="ep-steps">${fmt(ep.steps)}s · ${fmt(ep.hands_played)}h</span>
    </div>
  `;
  row.onclick = ()=>send({cmd:'load', episode: ep.index});
  return row;
}

// Reconstruct the joker set held at a given step by replaying
// buy/sell events from the action log.
function jokersAtStep(stepIdx){
  if(!replayCur) return [];
  const held = [];
  const log = replayCur.action_log || [];
  const last = Math.min(stepIdx, log.length - 1);
  for(let i = 0; i <= last; i++){
    const a = log[i];
    if(a.type === 'buy' && a.card){
      held.push(a.card);
    } else if(a.type === 'sell' && a.card){
      const idx = held.indexOf(a.card);
      if(idx >= 0) held.splice(idx, 1);
    }
  }
  return held;
}

function updateActiveJokers(){
  const body = $('replay-jokers');
  if(!body) return;
  const held = jokersAtStep(replayStepIdx);
  $('rj-count').textContent = held.length ? `${held.length} / 5` : '';
  body.innerHTML = '';
  if(!held.length){
    body.appendChild(mk('span', 'empty', 'No jokers yet'));
    return;
  }
  held.forEach(label => {
    const chip = mk('span', 'rj-chip');
    chip.textContent = label;
    body.appendChild(chip);
  });
}

let _epVirtualHandlersBound = false;
function renderReplaySidebar(){
  let side = $('side');
  if(!side.querySelector('.ep-panel')){
    side.innerHTML = '';

    const jpanel = mk('div', 'panel');
    jpanel.innerHTML = `<div class="ph">🃏 Active Jokers <span class="ph-right" id="rj-count"></span></div>`;
    const jbody = mk('div', 'pb rj-body');
    jbody.id = 'replay-jokers';
    jpanel.appendChild(jbody);
    side.appendChild(jpanel);

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
    controls.querySelector('#ep-sort').onchange = (e)=>{ replaySort=e.target.value; recomputeReplayView(); };
    controls.querySelectorAll('.ep-filter-btn').forEach(btn=>{
      btn.onclick = ()=>{
        replayFilter = btn.dataset.f;
        controls.querySelectorAll('.ep-filter-btn').forEach(b=>b.classList.toggle('active',b===btn));
        recomputeReplayView();
      };
    });
  }

  $('ep-count').textContent = `${replayView.length}${replayView.length!==replayEpisodesRaw.length?` / ${replayEpisodesRaw.length}`:''}`;
  renderEpisodeListVirtual();
}

function renderEpisodeListVirtual(){
  const epList = $('ep-list-scroll');
  if(!epList) return;

  let spacerTop = epList.querySelector('.ep-spacer-top');
  let rowsContainer = epList.querySelector('.ep-rows');
  let spacerBot = epList.querySelector('.ep-spacer-bot');
  if(!rowsContainer){
    epList.innerHTML = '';
    spacerTop = mk('div','ep-spacer-top');
    rowsContainer = mk('div','ep-rows');
    spacerBot = mk('div','ep-spacer-bot');
    epList.appendChild(spacerTop);
    epList.appendChild(rowsContainer);
    epList.appendChild(spacerBot);
  }

  if(!replayView.length){
    spacerTop.style.height='0px'; spacerBot.style.height='0px';
    rowsContainer.innerHTML='';
    rowsContainer.appendChild(mk('span','empty','No episodes match filter'));
    return;
  }

  function update(){
    const total = replayView.length;
    const scrollTop = epList.scrollTop;
    const viewH = epList.clientHeight||400;
    const buffer = 6;
    const start = Math.max(0, Math.floor(scrollTop/EP_ROW_H)-buffer);
    const end   = Math.min(total, Math.ceil((scrollTop+viewH)/EP_ROW_H)+buffer);
    spacerTop.style.height = (start*EP_ROW_H)+'px';
    spacerBot.style.height = Math.max(0,(total-end)*EP_ROW_H)+'px';
    rowsContainer.innerHTML='';
    for(let i=start;i<end;i++) rowsContainer.appendChild(buildEpRow(replayView[i]));
  }

  if(!_epVirtualHandlersBound){
    epList.addEventListener('scroll',()=>renderEpisodeListVirtual());
    _epVirtualHandlersBound=true;
  }
  update();

  if(replayActiveEp>=0 && !epList.dataset.scrolledTo){
    const idx = replayView.findIndex(e=>e.episode===replayActiveEp);
    if(idx>=0){
      epList.scrollTop = Math.max(0,idx*EP_ROW_H-viewHalf(epList));
      epList.dataset.scrolledTo='1';
      update();
    }
  }
}
function viewHalf(el){ return (el.clientHeight||400)/2; }

function renderReplayTimeline(ep, stepIdx){
  const m = $('main'); m.innerHTML='';
  const log = ep.action_log||[];
  replayRowEls = new Array(log.length);

  if(!log.length){
    const p=mk('div','panel');
    p.innerHTML='<div class="ph">Timeline</div>';
    const b=mk('div','pb');
    b.appendChild(mk('span','empty','No actions recorded — train with EpisodeRecorderWrapper enabled'));
    p.appendChild(b); m.appendChild(p);
    return;
  }

  const groups=[];
  let curGroup=null;
  log.forEach((a,i)=>{
    const key=`${a.ante}-${a.round}`;
    if(!curGroup||curGroup.key!==key){
      curGroup={key,ante:a.ante,round:a.round,items:[]};
      groups.push(curGroup);
    }
    curGroup.items.push({...a,_idx:i});
  });

  let scrollTarget=null;
  const container=mk('div','tl-container');

  groups.forEach(g=>{
    const grp=mk('div','tl-group');
    const head=mk('div','tl-group-head');
    head.innerHTML=`<span class="tl-gh-ante">Ante ${g.ante}</span><span class="tl-gh-sep">·</span><span class="tl-gh-round">Round ${g.round}</span>`;
    grp.appendChild(head);

    g.items.forEach(a=>{
      const isActive=a._idx===stepIdx;
      const row=mk('div','tl-row'+(isActive?' active':''));
      replayRowEls[a._idx]=row;
      if(isActive) scrollTarget=row;

      row.appendChild(mk('span',`tl-type tl-${a.type}`,a.type.toUpperCase()));
      if(a.strategy) row.appendChild(stratChip(a.strategy));
      const detail=mk('div','tl-detail');

      if(a.type==='play'){
        const top=mk('div','tl-top-line');
        top.innerHTML=`<span class="tl-hand">${esc(a.hand_type||'?')}</span>${a.total?`<span class="tl-pts">${fmt(a.total)} pts</span>`:''}`;
        detail.appendChild(top);
        if(a.chips_total!=null){
          const bar=mk('div','tl-score-bar');
          const pct=a.blind_chips?Math.min(100,(a.chips_total/a.blind_chips)*100):0;
          const won=a.chips_total>=(a.blind_chips||Infinity);
          bar.innerHTML=`<span class="tl-score-val ${won?'won':''}">${fmt(a.chips_total)}</span>`
            +(a.blind_chips?`<span class="tl-score-sep">/</span><span class="tl-score-blind">${fmt(a.blind_chips)}</span>`:'')
            +(a.blind_chips?`<span class="tl-score-track"><span class="tl-score-fill ${won?'won':''}" style="width:${pct.toFixed(1)}%"></span></span>`:'');
          detail.appendChild(bar);
        }
        if(a.cards?.length){
          const cr=mk('div','tl-cards');
          a.cards.forEach(c=>cr.appendChild(tlChip(c)));
          detail.appendChild(cr);
        }
        if(a.jokers?.length){
          const jr=mk('div','tl-jokers');
          a.jokers.forEach(j=>{const jc=mk('span','tl-jchip');jc.textContent=j;jr.appendChild(jc);});
          detail.appendChild(jr);
        }
      } else if(a.type==='skip'){
        detail.appendChild(mk('div','tl-top-line','<span class="tl-skip-lbl">Skipped shop</span>'));
      } else if(a.type==='discard'){
        const top=mk('div','tl-top-line');
        top.appendChild(mk('span','tl-discard-lbl',`${(a.cards||[]).length} card${(a.cards||[]).length!==1?'s':''}`));
        detail.appendChild(top);
        if(a.cards?.length){
          const cr=mk('div','tl-cards');
          a.cards.forEach(c=>cr.appendChild(tlChip(c)));
          detail.appendChild(cr);
        }
      } else if(a.type==='buy'){
        const top=mk('div','tl-top-line');
        top.innerHTML=`<span class="tl-item">${esc(a.card||'?')}</span>${a.cost?`<span class="tl-cost">$${a.cost}</span>`:''}`;
        detail.appendChild(top);
      } else if(a.type==='sell'){
        const top=mk('div','tl-top-line');
        top.innerHTML=`<span class="tl-item">${esc(a.card||'?')}</span>${a.gold?`<span class="tl-gold">+$${a.gold}</span>`:''}`;
        detail.appendChild(top);
      } else {
        detail.textContent=a.type;
      }

      row.appendChild(detail);
      if(a.dollars!=null) row.appendChild(mk('span','tl-dol','$'+a.dollars));
      row.onclick=()=>send({cmd:'step',index:a._idx});
      grp.appendChild(row);
    });

    container.appendChild(grp);
  });

  m.appendChild(container);
  if(scrollTarget) requestAnimationFrame(()=>scrollTarget.scrollIntoView({block:'nearest',behavior:'smooth'}));
}

function renderReplayNav(total, idx){
  const row=$('acts-row'); row.innerHTML='';
  $('acts-label').textContent='Navigation';
  $('watch-note').classList.remove('show');
  replayTotalSteps=total;

  const nav=mk('div','replay-nav');

  const prev=mk('button','btn','◀ Prev');
  prev.title='Previous step (← arrow key)';
  prev.disabled=idx===0;
  prev.onclick=()=>send({cmd:'prev'});

  const slider=document.createElement('input');
  slider.type='range'; slider.min=0; slider.max=Math.max(0,total-1);
  slider.value=idx; slider.className='rn-slider';
  slider.oninput=()=>send({cmd:'step',index:parseInt(slider.value)});

  const next=mk('button','btn success','Next ▶');
  next.title='Next step (→ arrow key)';
  next.disabled=idx>=total-1;
  next.onclick=()=>send({cmd:'next'});

  const lbl=mk('div','rn-step',`${fmt(idx+1)} / ${fmt(total)}`);

  nav.appendChild(prev);
  nav.appendChild(slider);
  nav.appendChild(next);
  nav.appendChild(lbl);
  row.appendChild(nav);
}

function onReplayEpisodeLoaded(ep){
  replayCur=ep;
  replayActiveEp=ep.episode_index??0;
  replayStepIdx=ep.step_index??0;

  const epList=$('ep-list-scroll');
  if(epList) delete epList.dataset.scrolledTo;
  renderReplaySidebar();

  renderHeaderForStep();
  renderReplayTimeline(ep,replayStepIdx);
  renderHandHistory(visibleHandHistory());
  renderReplayNav(ep.total_steps||0,replayStepIdx);
  updateActiveJokers();
}

function onReplayCursor(c){
  if(!replayCur) return;
  replayStepIdx=c.step_index??0;

  const prevActive=document.querySelector('#main .tl-row.active');
  if(prevActive) prevActive.classList.remove('active');
  const newActive=replayRowEls[replayStepIdx];
  if(newActive){
    newActive.classList.add('active');
    newActive.scrollIntoView({block:'nearest',behavior:'smooth'});
  }

  renderHeaderForStep();
  renderHandHistory(visibleHandHistory());
  renderReplayNav(c.total_steps||replayCur.total_steps||0,replayStepIdx);
  updateActiveJokers();
}

function renderHeaderForStep(){
  const ep=replayCur; if(!ep) return;
  $('h-phase').textContent='Replay';
  $('h-ante').textContent=`A${ep.ante_reached||0} · ${ep.rounds_beaten||0} rounds${ep.won?' ✓':''}`;
  $('h-money').textContent=`Ep ${ep.episode_index??0}`;
  $('h-hands').textContent=`${visibleHandHistory().length} hands`;
  $('h-disc').textContent=`${ep.total_steps||0} steps`;
  $('h-won').style.display=ep.won?'':'none';

  const pct=ep.total_steps?replayStepIdx/ep.total_steps:0;
  $('bar').style.width=(pct*100).toFixed(1)+'%';
  $('s-cur').textContent=`Step ${replayStepIdx+1}`;
  $('s-max').textContent=`${ep.total_steps||0}`;
  $('s-pct').textContent=ep.won?'🏆 Won':`${visibleHandHistory().length} hands`;
  $('heval').classList.remove('visible');
}

function visibleHandHistory(){
  if(!replayCur) return [];
  return (replayCur.hand_history||[]).filter(h=>h._step<=replayStepIdx);
}

/* ── Hand History ────────────────────────────────────────── */
const HH_MAX_RENDER = 60;

function renderHandHistory(entries){
  entries=entries||[];
  const sec=$('hist'); sec.innerHTML='';
  if(!entries.length) return;

  const hdr=mk('div','hh-header');
  const capped=entries.length>HH_MAX_RENDER;
  hdr.innerHTML=`Hand History <span style="color:#4a6a4a;font-size:10px;font-weight:400;letter-spacing:0;text-transform:none;">(${entries.length} played${capped?`, showing last ${HH_MAX_RENDER}`:''})</span>`;
  sec.appendChild(hdr);

  const scroll=mk('div','hh-scroll');
  const shown=capped?entries.slice(entries.length-HH_MAX_RENDER):entries;
  [...shown].reverse().forEach(e=>{
    const card=mk('div','hh-entry'+(e.debuffed?' debuffed':''));
    const meta=mk('div','hh-meta');
    meta.innerHTML=`<span class="hh-tag">A${e.ante}</span><span class="hh-tag">R${e.round}</span>${e.hand_num!=null?`<span class="hh-tag">H${e.hand_num}</span>`:''}${e.strategy?`<span class="hh-tag hh-strat">${esc(prettyStrat(e.strategy))}</span>`:''}`;
    card.appendChild(meta);

    if(e.hand_type){
      const ht=mk('div','hh-type');
      ht.textContent=e.debuffed?`${e.hand_type} ✗`:e.hand_type;
      card.appendChild(ht);
    }

    if(e.chips||e.mult||e.total){
      const sc=mk('div','hh-score');
      sc.innerHTML=`<span class="hh-chips">${fmt(e.chips)}</span><span class="hh-dim">chips ×</span><span class="hh-mult">${fmt(e.mult)}</span><span class="hh-dim">mult =</span><span class="hh-total">${fmt(e.total)}</span>`;
      card.appendChild(sc);
    }

    if(e.cards?.length){
      const row=mk('div','hh-cards');
      e.cards.forEach(c=>{
        const isStr=typeof c==='string';
        const label=isStr?c:(c.label||c.name||c.key||'?');
        const chip=mk('span','hh-card'+(isRedCard(label)?' red':'')+(c.scored?' scored':'')+(c.destroyed?' destroyed':''));
        chip.textContent=prettyCard(label);
        row.appendChild(chip);
      });
      card.appendChild(row);
    }

    if(e.jokers?.length){
      const jrow=mk('div','hh-jokers');
      e.jokers.forEach(j=>{
        const jc=mk('span','hh-joker');
        jc.textContent=typeof j==='string'?j:(j.name||j.key||'?');
        jrow.appendChild(jc);
      });
      card.appendChild(jrow);
    }

    if(e.dollars_earned>0){
      const d=mk('div','hh-dollars');
      d.textContent=`+$${e.dollars_earned}`;
      card.appendChild(d);
    }
    scroll.appendChild(card);
  });
  sec.appendChild(scroll);
}

/* ── Init ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', ()=>{
  $('h-mode').textContent='⏪ Replay';
  document.title='Balatro AI — Replay';
  $('watch-note').style.display='none';
  $('app').classList.add('replay-mode');

  document.addEventListener('keydown',e=>{
    if(e.key==='ArrowLeft')  { e.preventDefault(); send({cmd:'prev'}); }
    if(e.key==='ArrowRight') { e.preventDefault(); send({cmd:'next'}); }
  });

  connect();
});
