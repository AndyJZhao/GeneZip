(() => {

const ns='http://www.w3.org/2000/svg', colors=['#df9b65','#577cb5','#577cb5','#577cb5','#7d909d','#7d909d','#7d909d'];
const labels=['Promoter','CDS','UTR','Exon','Intron','NIG','DIG'];
const $=id=>document.getElementById(id);
const svg=(tag,attrs,text='')=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));e.textContent=text;return e};
let data,selected=0,tokens=[],elapsed=0,last=0,playing=!matchMedia('(prefers-reduced-motion: reduce)').matches;
const duration=1500;
function paint(){
 const p=Math.min(elapsed/duration,1), reveal=Math.min(p/.28,1), move=Math.max(0,Math.min((p-.34)/.36,1)), ease=move*move*(3-2*move);
 tokens.forEach(t=>{t.e.setAttribute('x1',t.x+(t.to-t.x)*ease);t.e.setAttribute('x2',t.x+(t.to-t.x)*ease);t.e.setAttribute('opacity',reveal)});
 $('phase').textContent=p<.28?'1 / Set regional targets':p<.7?'2 / Allocate and compact':'3 / Compare the allocation';
}
function frame(now){if(playing){elapsed=Math.min(elapsed+(last?now-last:0),duration);if(elapsed===duration)playing=false;paint()}last=now;requestAnimationFrame(frame)}
function choose(index){
 selected=index;const p=data.priors[index];
 document.querySelectorAll('.presets button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)));
 $('prior-name').textContent=p.name;$('prior-description').textContent=p.description;
 $('regions').replaceChildren();$('input').replaceChildren();$('tokens').replaceChildren();tokens=[];
 let n=0;
 p.targets.forEach((target,i)=>{
  const x=i*142.85;
  $('regions').append(svg('text',{x:x+69,y:46,'text-anchor':'middle',class:'region-label'},labels[i]),svg('text',{x:x+69,y:82,'text-anchor':'middle',class:'target-label',fill:colors[i]},`${target}×`));
  $('input').append(svg('rect',{x:x+1,y:158,width:139,height:25,rx:3,fill:colors[i],opacity:.28}));
  for(let j=0;j<2048/target;j++){const pos=x+3+(j+.5)*136/(2048/target),to=3+n*3.4;const e=svg('line',{x1:pos,x2:pos,y1:246,y2:278,stroke:colors[i],'stroke-width':2});$('tokens').append(e);tokens.push({e,x:pos,to});n++}
 });
 $('diagram-desc').textContent=`${p.name} targets: ${labels.map((l,i)=>l+' '+p.targets[i]+' times').join(', ')}. Ideal allocation for equal regions, not measured routing.`;
 document.querySelectorAll('#prior-table tbody tr').forEach((r,i)=>r.classList.toggle('selected',i===index));
 playing=!matchMedia('(prefers-reduced-motion: reduce)').matches;elapsed=playing?0:duration;last=0;paint();
}
function renderDownstream(){
 const names=['No Prior (HNet)','Default (Promoter+Genic)','Regulome','PromoterDistal','Intergenic'];
 const models=[data.baseline,...data.priors];
 const modelColors=['#9c8058','#c64c2e','#438a91','#577cb5','#68778b'];
 const strip=$('allocation-strips'),start=220,width=780,compressedWidth=400;
 strip.append(svg('text',{x:0,y:30,class:'track-label'},'Same input DNA'));
 labels.forEach((label,i)=>{
  strip.append(svg('rect',{x:start+i*width/7,y:17,width:width/7-2,height:22,rx:2,fill:colors[i],opacity:.28}));
  strip.append(svg('text',{x:start+(i+.5)*width/7,y:57,'text-anchor':'middle',fill:'#48576d','font-size':11},label));
 });
 const tokenBudget=140;
 models.forEach((p,row)=>{
  const y=98+row*53;let count=0;
  const name=svg('text',{x:20,y:row===1?y-6:y,class:'sequence-name',fill:'#101d35'},row===1?'Default':names[row]);
  if(row===1)name.append(svg('tspan',{x:20,dy:19},'(Promoter+Genic)'));
  strip.append(svg('rect',{x:0,y:y-10,width:11,height:11,fill:modelColors[row]}),name);
  const weights=row===0?labels.map(()=>1):p.targets.map(target=>1/target);
  const total=weights.reduce((a,b)=>a+b,0);
  const quotas=weights.map(w=>w/total*tokenBudget),counts=quotas.map(Math.floor);
  const remainder=tokenBudget-counts.reduce((a,b)=>a+b,0);
  quotas.map((q,i)=>({i,fraction:q-counts[i]})).sort((a,b)=>b.fraction-a.fraction).slice(0,remainder).forEach(({i})=>counts[i]++);
  const track=svg('g',{class:'allocation-track','aria-label':`${names[row]}: ${tokenBudget} illustrative tokens`});
  counts.forEach((n,i)=>{
   for(let j=0;j<n;j++){
    const x=start+(count+.5)*compressedWidth/tokenBudget;
    track.append(svg('line',{x1:x,x2:x,y1:y-15,y2:y+7,stroke:colors[i],'stroke-width':2.2}));count++;
   }
  });
  strip.append(track);
 });
 $('prior-legend').innerHTML=names.map((name,i)=>`<span><i style="background:${modelColors[i]}" aria-hidden="true"></i>${name}</span>`).join('');
 const tasks=[['cmp','CMP','SCC',.08,.14,[.08,.10,.12,.14]],['eqtl','eQTL','AUROC',.76,.84,[.76,.78,.80,.82,.84]],['etgp','ETGP','AUPRC',.20,.55,[.20,.25,.30,.35,.40,.45,.50,.55]],['tisp','TISP','PCC',.24,.36,[.24,.27,.30,.33,.36]]];
 $('prior-charts').innerHTML=tasks.map(([key,title,metric,min,max,ticks])=>{
  const percent=v=>(v-min)/(max-min)*100;
  return `<section class="task-panel" aria-label="${title}, ${metric}, axis ${min} to ${max}${min?' (truncated)':''}"><h3>${title}</h3><p>${metric} ↑</p><div class="panel-plot"><div class="panel-axis" aria-hidden="true">${ticks.map(v=>`<span style="bottom:${percent(v)}%">${v.toFixed(2)}</span>`).join('')}</div><div class="panel-grid" aria-hidden="true">${ticks.map(v=>`<span style="bottom:${percent(v)}%"></span>`).join('')}</div><div class="panel-bars">${models.map((p,i)=>`<div class="vertical-bar" style="height:${percent(p[key])}%;background:${modelColors[i]}" role="img" aria-label="${title}, ${names[i]}: ${p[key].toFixed(4)}"><span class="vertical-value">${p[key].toFixed(4)}</span></div>`).join('')}</div></div></section>`;
 }).join('');
 $('prior-scores').innerHTML='<thead><tr><th>Prior</th><th>Measured compression</th><th>CMP SCC ↑</th><th>eQTL AUROC ↑</th><th>ETGP AUPRC ↑</th><th>TISP PCC ↑</th></tr></thead><tbody>'+models.map((p,i)=>`<tr><td>${names[i]}</td><td>${p.compression.toFixed(2)}×</td>${['cmp','eqtl','etgp','tisp'].map(k=>`<td>${p[k].toFixed(4)}</td>`).join('')}</tr>`).join('')+'</tbody>';
}
fetch('assets/region-target-data.json').then(r=>{if(!r.ok)throw Error('Cannot load source data');return r.json()}).then(d=>{
 data=d;
 document.querySelector('.presets').innerHTML=d.priors.map((p,i)=>`<button type="button" aria-pressed="${i===0}">${p.short}</button>`).join('');
 document.querySelectorAll('.presets button').forEach((b,i)=>b.onclick=()=>choose(i));
 $('prior-table').innerHTML='<thead><tr><th>Pretrained prior</th>'+labels.map(l=>`<th>${l}</th>`).join('')+'</tr></thead><tbody>'+d.priors.map(p=>`<tr><td>${p.name}</td>${p.targets.map(v=>`<td>${v}×</td>`).join('')}</tr>`).join('')+'</tbody>';
 renderDownstream();choose(0);requestAnimationFrame(frame);
}).catch(e=>{$('prior-description').textContent='Source data could not be loaded. Please reload the page.';console.error(e)});
})();
