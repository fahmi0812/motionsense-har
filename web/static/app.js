// Frontend MotionSense. Jika frontend di-host terpisah (Vercel), set window.API_BASE = "https://backend-anda.onrender.com"
const API = window.API_BASE || "";
const LBL = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"];
const WARNA = ["#e6194b","#f58231","#ffe119","#3cb44b","#4363d8","#911eb4"];
const $ = id => document.getElementById(id);
let charts = {};
function draw(id, cfg){ if(charts[id]) charts[id].destroy(); charts[id] = new Chart($(id), cfg); }

async function loadSessions(){
  const h = await (await fetch(API+"/api/health")).json(); $("mode").textContent = "Mode: "+h.mode;
  const r = await fetch(API+"/api/sessions"); const s = await r.json();
  if(!r.ok){ $("status").textContent = "Gagal ambil sesi: "+(s.detail||r.status); return; }
  $("sesi").innerHTML = s.map(x=>`<option value="${x.session_id}">${x.session_id.slice(0,24)} | ${x.user_id} | ${x.label}${x.n?" | "+x.n+" baris":""}</option>`).join("");
}
function show(d){
  $("status").textContent = `${d.n_sampel} sampel (${d.durasi_total_s} s), ${d.windows.length} window`;
  const ds = ["ax","ay","az","gx","gy","gz"].map((c,i)=>({label:c,data:d.sinyal[c],borderWidth:1,pointRadius:0,borderColor:WARNA[i]}));
  draw("cSig",{type:"line",data:{labels:d.sinyal.t,datasets:ds},options:{animation:false,scales:{x:{title:{display:true,text:"detik"},ticks:{maxTicksLimit:15}}}}});
  draw("cTime",{type:"bar",data:{labels:d.windows.map(w=>w.mulai_s),datasets:[{label:"keyakinan (warna = aktivitas)",
     data:d.windows.map(w=>w.conf),backgroundColor:d.windows.map(w=>WARNA[LBL.indexOf(w.label)])}]},
     options:{animation:false,plugins:{tooltip:{callbacks:{label:c=>`${d.windows[c.dataIndex].label} (${(c.raw*100).toFixed(0)}%)`}}},
     scales:{y:{min:0,max:1},x:{title:{display:true,text:"mulai window (detik)"},ticks:{maxTicksLimit:20}}}}});
  const k = Object.keys(d.ringkasan_durasi_s);
  draw("cDur",{type:"doughnut",data:{labels:k,datasets:[{data:k.map(x=>d.ringkasan_durasi_s[x]),backgroundColor:k.map(x=>WARNA[LBL.indexOf(x)])}]}});
  $("tDur").innerHTML = "<tr><th>Aktivitas</th><th>Durasi (detik)</th></tr>"+k.map(x=>`<tr><td>${x}</td><td>${d.ringkasan_durasi_s[x]}</td></tr>`).join("");
}
async function run(p){ $("status").textContent="Memproses..."; const r = await p; const d = await r.json();
  if(!r.ok){ $("status").textContent = "Error: "+(d.detail||r.status); return; } show(d); }
$("btnSesi").onclick = ()=> run(fetch(API+"/api/sessions/"+encodeURIComponent($("sesi").value)+"/predict"));
$("btnReload").onclick = loadSessions;
$("btnCsv").onclick = ()=>{ const f=$("file").files[0]; if(!f) return alert("Pilih file CSV dulu");
  const fd=new FormData(); fd.append("file",f); run(fetch(API+"/api/predict_csv",{method:"POST",body:fd})); };
loadSessions();
