const $=id=>document.getElementById(id);
let lastResult=null;
function api(url,options={}){return fetch(url,{headers:{'Content-Type':'application/json'},...options}).then(async r=>{const d=await r.json();if(!r.ok)throw new Error(d.error||'Request failed');return d})}

document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.tab-panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active');if(b.dataset.tab==='students')loadStudents();if(b.dataset.tab==='faculty')loadFaculty();if(b.dataset.tab==='models')loadMetrics()});
function academic(prefix=''){return {attendance:+$(prefix+'attendance').value,internal_marks:+$(prefix+'internal').value,assignment_score:+$(prefix+'assignment').value,study_hours:+$(prefix+'study').value,previous_cgpa:+$(prefix+'cgpa').value,backlog_count:+$(prefix+'backlogs')?.value||0}}
$('predictBtn').onclick=async()=>{try{const d=academic();const r=await api('/api/predict',{method:'POST',body:JSON.stringify(d)});lastResult=r;renderResult(r)}catch(e){alert(e.message)}};
function renderResult(r){$('score').textContent=r.hybrid_score;$('label').textContent=r.performance_label;$('best').textContent=(r.best_model||'').toUpperCase();$('fuzzy').textContent=r.fuzzy_score;$('riskBadge').textContent=r.risk+' Risk';$('riskBadge').className='risk-badge '+r.risk.toLowerCase().replace(' ','-');
$('riskFactors').innerHTML=(r.risk_breakdown||[]).map(x=>`<div class="bar"><div class="bar-top"><span>${x.factor}</span><b>${Math.round(x.risk_weight*100)}%</b></div><div class="track"><div class="fill" style="width:${Math.round(x.risk_weight*100)}%"></div></div></div>`).join('');
$('intervention').innerHTML=(r.intervention?.actions||[]).map(a=>`<div class="action"><strong>${a.priority} · ${a.area}</strong>${a.action}<br><small>Target: ${a.target}</small></div>`).join('');
$('memberships').innerHTML=Object.entries(r.memberships||{}).map(([k,v])=>`<div class="membership"><b>${k.replaceAll('_',' ')}</b>${Object.entries(v).map(([a,n])=>`${a}: ${Number(n).toFixed(2)}`).join('<br>')}</div>`).join('');
$('rules').innerHTML=(r.fired_rules||[]).map(x=>`<div><b>${Number(x.firing_strength).toFixed(2)}</b> · ${x.rule}</div>`).join('')||'<div>No rule fired.</div>'}
$('whatIfBtn').onclick=async()=>{if(!lastResult){alert('Run the main assessment first.');return}const d=academic();const scenario={attendance:+$('wa').value,study_hours:+$('ws').value,internal_marks:+$('wi').value,assignment_score:d.assignment_score,previous_cgpa:d.previous_cgpa};const r=await api('/api/what-if',{method:'POST',body:JSON.stringify({...d,scenarios:[scenario]})});const x=r.scenarios[0];$('whatIfResult').innerHTML=`Baseline: <b>${r.baseline}</b> → Scenario: <b>${x.score}</b> · Change: <b>${x.change>=0?'+':''}${x.change}</b>`};
$('addStudent').onclick=async()=>{try{const r=await api('/api/students',{method:'POST',body:JSON.stringify({student_code:$('scode').value,name:$('sname').value,department:$('sdept').value,division:$('sdiv').value,semester:$('ssem').value,year:$('syear').value})});$('studentMsg').textContent=`Saved ${r.name}`;loadStudents()}catch(e){$('studentMsg').textContent=e.message;$('studentMsg').className='muted error'}};
async function loadStudents(){const list=await api('/api/students');$('studentList').innerHTML=list.length?list.map(s=>`<div class="student-item"><div><b>${s.student_code} · ${s.name}</b><br><small>${s.department} · ${s.division} · ${s.semester}</small></div><button onclick="analyzeSaved(${s.id})">Analyze latest</button></div>`).join(''):'<p class="muted">No students saved yet.</p>'}
async function analyzeSaved(id){try{const r=await api(`/api/students/${id}/analyze`,{method:'POST',body:JSON.stringify({attendance:72,internal_marks:68,assignment_score:80,study_hours:3,previous_cgpa:7.4,backlog_count:0})});lastResult=r.result;renderResult(r.result);document.querySelector('[data-tab="analysis"]').click()}catch(e){alert(e.message)}}
let subjectCount=0;function addSubject(name='Subject '+(++subjectCount),weak=.5){const row=document.createElement('div');row.className='subject-row';row.innerHTML=`<label>Subject<input class="sub-name" value="${name}"></label><label>Weakness 0-1<input class="sub-weak" type="number" min="0" max="1" step=".1" value="${weak}"></label><button type="button">Remove</button>`;row.querySelector('button').onclick=()=>row.remove();$('subjectInputs').appendChild(row)}
$('addSubject').onclick=()=>addSubject();addSubject('Data Structures',.8);addSubject('AI & ML',.6);addSubject('Mathematics',.7);addSubject('Other',.3);
$('optimizeBtn').onclick=async()=>{try{const subjects=[...document.querySelectorAll('.subject-row')].map(r=>({subject:r.querySelector('.sub-name').value,weakness:+r.querySelector('.sub-weak').value}));const d={attendance:+$('oa').value,internal_marks:+$('oi').value,assignment_score:+$('oo').value,study_hours:+$('os').value,previous_cgpa:+$('oc').value,total_hours:+$('ot').value,subjects};const r=await api('/api/optimize-study-plan',{method:'POST',body:JSON.stringify(d)});renderPlans(r)}catch(e){alert(e.message)}};
function renderPlans(r){$('optimizationResult').innerHTML=['ga','pso'].map(k=>{const x=r[k];return `<div class="plan"><h3>${x.algorithm} optimized plan</h3><p class="muted">Estimated model score: <b>${x.estimated_score}</b></p>${x.plan.map(p=>`<div class="plan-row"><span>${p.subject}</span><b>${p.hours} h</b></div>`).join('')}</div>`}).join('')}
async function loadFaculty(){const r=await api('/api/class-summary');$('summaryCards').innerHTML=[['Students',r.total_students],['Average score',r.average_score??'—'],['High risk',r.risk_distribution.High],['Very high risk',r.risk_distribution['Very High']]].map(x=>`<div class="summary"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('');const students=await api('/api/students');let rows=[];for(const s of students){const h=await api(`/api/students/${s.id}/history`);const a=h.analyses.at(-1);rows.push(`<div class="student-item"><div><b>${s.student_code} · ${s.name}</b><br><small>${s.department} · ${s.semester} · ${s.division}</small>${a?`<br><small>Latest score ${a.hybrid_score} · ${a.risk} Risk · ${a.performance_label}</small>`:'<br><small>No report generated yet.</small>'}</div><div class="student-actions"><button class="small" data-action="report" data-id="${s.id}">View report</button><button class="small danger" data-action="remove" data-id="${s.id}">Remove</button></div></div>`)}$('facultyTable').innerHTML=rows.join('')||'No saved analyses yet.';$('facultyReport').innerHTML='Select a student to view the detailed report.'}

$('facultyTable').addEventListener('click', async event => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const id = Number(button.dataset.id);

  if (button.dataset.action === 'report') {
    try {
      const report = await api(`/api/students/${id}/report`);
      const latest = report.latest_analysis || {};
      const record = report.latest_record || {};
      const history = (report.analyses || []).slice(-5).reverse();
      $('facultyReport').innerHTML = `
        <div class="report-box">
          <h3>${report.student.name}</h3>
          <div class="report-grid">
            <div><span>Student code</span><b>${report.student.student_code}</b></div>
            <div><span>Department</span><b>${report.student.department}</b></div>
            <div><span>Semester</span><b>${report.student.semester}</b></div>
            <div><span>Division</span><b>${report.student.division}</b></div>
            <div><span>Year</span><b>${report.student.year}</b></div>
            <div><span>Records</span><b>${report.total_records}</b></div>
          </div>
          <div class="report-metrics">
            <div><span>Latest score</span><b>${latest.hybrid_score ?? '—'}</b></div>
            <div><span>Risk</span><b>${latest.risk ?? '—'}</b></div>
            <div><span>Performance</span><b>${latest.performance_label ?? '—'}</b></div>
            <div><span>Fuzzy score</span><b>${latest.fuzzy_score ?? '—'}</b></div>
          </div>
          <h4>Latest academic record</h4>
          <div class="report-grid">
            <div><span>Attendance</span><b>${record.attendance ?? '—'}</b></div>
            <div><span>Internal</span><b>${record.internal_marks ?? '—'}</b></div>
            <div><span>Assignment</span><b>${record.assignment_score ?? '—'}</b></div>
            <div><span>Study hours</span><b>${record.study_hours ?? '—'}</b></div>
            <div><span>CGPA</span><b>${record.previous_cgpa ?? '—'}</b></div>
            <div><span>Backlogs</span><b>${record.backlog_count ?? '—'}</b></div>
          </div>
          ${history.length ? `
            <h4>Recent analysis history</h4>
            <div class="history-list">
              ${history.map(a => `<div><b>${a.risk}</b> · ${a.hybrid_score} score · ${a.performance_label}</div>`).join('')}
            </div>
          ` : '<p class="muted">No saved analyses yet.</p>'}
        </div>
      `;
    } catch (e) {
      $('facultyReport').innerHTML = `<p class="muted error">${e.message}</p>`;
    }
  } else if (button.dataset.action === 'remove') {
    const ok = window.confirm('Remove this student from the faculty list?');
    if (!ok) return;
    try {
      await api(`/api/students/${id}`, { method: 'DELETE' });
      await loadFaculty();
      $('facultyReport').innerHTML = 'Student removed. Select another student to view a report.';
    } catch (e) {
      alert(e.message);
    }
  }
});
async function loadMetrics(){const r=await api('/api/metrics');const rows=Object.entries(r.metrics).map(([k,v])=>`<tr><td>${k.toUpperCase()}</td><td>${v.mse}</td><td>${v.mae}</td><td>${v.r2}</td></tr>`).join('');$('metricsTable').innerHTML=`<table><thead><tr><th>Model</th><th>MSE</th><th>MAE</th><th>R²</th></tr></thead><tbody>${rows}</tbody></table>`;$('trainInfo').textContent=`Training samples: ${r.n_samples} · Training time: ${r.train_time_sec}s · Best neural model: ${r.best_model}`;renderChart('gaChart',r.history.ga,'generation');renderChart('psoChart',r.history.pso,'iteration')}
function renderChart(id,h,key){const vals=h.map(x=>x.best_mse);const min=Math.min(...vals),max=Math.max(...vals);$(id).innerHTML=vals.map((v,i)=>{const pct=max===min?70:10+((max-v)/(max-min))*85;return `<div class="cbar" title="${key} ${i+1}: ${v}" style="height:${pct}%"><span>${v}</span></div>`}).join('')}
$('retrainBtn').onclick=async()=>{await api('/api/train?force=true',{method:'POST'});loadMetrics()};
(async()=>{try{const s=await api('/api/status');if(!s.trained)await api('/api/train',{method:'POST'})}catch(e){console.error(e)}})();
