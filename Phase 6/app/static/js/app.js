/* Akshar — App Logic */

let sessionId = null, currentVoice = 'female', audio = null, file = null;

// DOM
const $  = id => document.getElementById(id);
const uploadPage     = $('uploadPage');
const uploadCard     = $('uploadCard');
const uploadZone     = $('uploadZone');
const fileInput      = $('fileInput');
const fileSelectedBox= $('fileSelectedBox');
const processingBox  = $('processingBox');
const resultsPage    = $('resultsPage');

// ── Upload Zone click ────────────────────────────────
uploadZone.addEventListener('click', function(e) {
    e.stopPropagation();
    fileInput.click();
});

uploadZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadCard.style.borderColor = 'var(--accent)';
});
uploadZone.addEventListener('dragleave', function() {
    uploadCard.style.borderColor = '';
});
uploadZone.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadCard.style.borderColor = '';
    if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', function() {
    if (fileInput.files.length) pickFile(fileInput.files[0]);
});

function pickFile(f) {
    file = f;
    $('fileName').textContent = f.name;
    $('fileSize').textContent = fmtSize(f.size);
    uploadZone.classList.add('hidden');
    fileSelectedBox.classList.remove('hidden');
}

$('changeFileBtn').addEventListener('click', function() {
    file = null;
    fileInput.value = '';
    fileSelectedBox.classList.add('hidden');
    uploadZone.classList.remove('hidden');
});

// ── Smooth page transition (upload ⇄ results) ────────
function switchPage(hideEl, showEl) {
    hideEl.classList.add('leave-anim');
    hideEl.addEventListener('animationend', function onLeave() {
        hideEl.removeEventListener('animationend', onLeave);
        hideEl.classList.remove('leave-anim');
        hideEl.classList.add('hidden');
        showEl.classList.remove('hidden');
        showEl.classList.add('enter-anim');
        showEl.addEventListener('animationend', function onEnter() {
            showEl.removeEventListener('animationend', onEnter);
            showEl.classList.remove('enter-anim');
        });
    });
}

function fmtSize(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
}

// ── Process ──────────────────────────────────────────
$('processBtn').addEventListener('click', async function() {
    if (!file) return;
    var btn = $('processBtn');
    btn.disabled = true;
    $('processBtnText').textContent = 'Processing...';
    $('processSpinner').classList.remove('hidden');
    fileSelectedBox.classList.add('hidden');
    processingBox.classList.remove('hidden');
    setStep('detect');
    $('processingText').textContent = 'Detecting words...';

    var fd = new FormData();
    fd.append('file', file);

    try {
        var res = await fetch('/api/process', { method:'POST', body:fd });
        if (!res.ok) {
            var err = await res.json();
            throw new Error(err.detail || 'Processing failed');
        }
        setStep('recognize');
        $('processingText').textContent = 'Recognizing text...';
        var data = await res.json();
        sessionId = data.session_id;
        setStep('done');
        $('processingText').textContent = 'Done!';
        setTimeout(function() { showResults(data); }, 400);
    } catch(e) {
        toast(e.message, 'error');
        processingBox.classList.add('hidden');
        fileSelectedBox.classList.remove('hidden');
        btn.disabled = false;
        $('processBtnText').textContent = 'Process File';
        $('processSpinner').classList.add('hidden');
    }
});

function setStep(s) {
    ['s1','s2','s3'].forEach(function(id) {
        $(id).className = 'step';
    });
    if (s === 'detect')    { $('s1').classList.add('active'); }
    if (s === 'recognize') { $('s1').classList.add('done'); $('s2').classList.add('active'); }
    if (s === 'done')      { $('s1').classList.add('done'); $('s2').classList.add('done'); $('s3').classList.add('done'); }
}

// ── Show Results ─────────────────────────────────────
function showResults(data) {
    $('rFileName').textContent = file.name;
    $('rFileSize').textContent = fmtSize(file.size);
    var txt = data.combined_text || '';
    $('wordCount').textContent = txt ? txt.split(/\s+/).length + ' words' : '0 words';
    $('pageCount').textContent = data.total_pages + (data.total_pages > 1 ? ' pages' : ' page');
    $('recText').textContent = txt || '(No text recognized)';
    if (data.total_pages > 1) $('pngOpt').classList.add('hidden');
    else $('pngOpt').classList.remove('hidden');

    // Reset audio
    $('audioPlayer').classList.add('hidden');
    $('audioDlRow').classList.add('hidden');
    $('voiceToggle').classList.add('hidden');
    $('genAudioBtn').disabled = !txt.trim();
    currentVoice = 'female';
    $('vFemale').classList.add('active');
    $('vMale').classList.remove('active');

    switchPage(uploadPage, resultsPage);
}

// ── Upload New ───────────────────────────────────────
$('uploadNewBtn').addEventListener('click', function() {
    sessionId = null; file = null; currentVoice = 'female';
    if (audio) { audio.pause(); audio = null; }
    closeDD();

    // Clear old output so nothing lingers on the next upload
    $('recText').textContent = '—';
    $('wordCount').textContent = '— words';
    $('pageCount').textContent = '— page';
    $('audioPlayer').classList.add('hidden');
    $('audioDlRow').classList.add('hidden');
    $('voiceToggle').classList.add('hidden');
    $('genAudioBtn').disabled = false;
    $('vFemale').classList.add('active');
    $('vMale').classList.remove('active');

    fileSelectedBox.classList.add('hidden');
    processingBox.classList.add('hidden');
    uploadZone.classList.remove('hidden');
    $('processBtn').disabled = false;
    $('processBtnText').textContent = 'Process File';
    $('processSpinner').classList.add('hidden');
    fileInput.value = '';

    switchPage(resultsPage, uploadPage);
});

// ── Audio Generation ─────────────────────────────────
$('genAudioBtn').addEventListener('click', function() { synth(currentVoice); });

$('vFemale').addEventListener('click', function() {
    if (currentVoice !== 'female') { currentVoice = 'female'; $('vFemale').classList.add('active'); $('vMale').classList.remove('active'); synth('female'); }
});
$('vMale').addEventListener('click', function() {
    if (currentVoice !== 'male') { currentVoice = 'male'; $('vMale').classList.add('active'); $('vFemale').classList.remove('active'); synth('male'); }
});

async function synth(voice) {
    if (!sessionId) return;
    $('genAudioBtn').disabled = true;
    $('ttsLoading').classList.remove('hidden');
    $('ttsVoiceLabel').textContent = voice;
    $('audioPlayer').classList.add('hidden');
    $('audioDlRow').classList.add('hidden');

    try {
        var res = await fetch('/api/synthesize', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({session_id: sessionId, voice: voice})
        });
        if (!res.ok) { var e = await res.json(); throw new Error(e.detail || 'Synthesis failed'); }
        var data = await res.json();
        setupPlayer(data.audio_url);
        $('voiceToggle').classList.remove('hidden');
        $('audioDlRow').classList.remove('hidden');
        toast((voice === 'female' ? 'Female' : 'Male') + ' voice generated (' + data.duration_seconds + 's)', 'success');
    } catch(e) {
        toast(e.message, 'error');
    } finally {
        $('genAudioBtn').disabled = false;
        $('ttsLoading').classList.add('hidden');
    }
}

// ── Audio Player ─────────────────────────────────────
function setupPlayer(url) {
    if (audio) { audio.pause(); }
    audio = new Audio(url);
    audio.addEventListener('timeupdate', function() {
        if (!audio.duration) return;
        $('progFill').style.width = (audio.currentTime / audio.duration * 100) + '%';
        $('curTime').textContent = fmtTime(audio.currentTime);
    });
    audio.addEventListener('loadedmetadata', function() {
        $('totTime').textContent = fmtTime(audio.duration);
    });
    audio.addEventListener('ended', function() {
        $('iPlay').classList.remove('hidden');
        $('iPause').classList.add('hidden');
        $('progFill').style.width = '0%';
        $('curTime').textContent = '0:00';
    });
    $('progFill').style.width = '0%';
    $('curTime').textContent = '0:00';
    $('totTime').textContent = '0:00';
    $('iPlay').classList.remove('hidden');
    $('iPause').classList.add('hidden');
    $('audioPlayer').classList.remove('hidden');
}

$('playPauseBtn').addEventListener('click', function() {
    if (!audio) return;
    if (audio.paused) {
        audio.play();
        $('iPlay').classList.add('hidden');
        $('iPause').classList.remove('hidden');
    } else {
        audio.pause();
        $('iPlay').classList.remove('hidden');
        $('iPause').classList.add('hidden');
    }
});

$('progBar').addEventListener('click', function(e) {
    if (!audio || !audio.duration) return;
    var rect = $('progBar').getBoundingClientRect();
    audio.currentTime = ((e.clientX - rect.left) / rect.width) * audio.duration;
});

function fmtTime(s) {
    var m = Math.floor(s/60), sec = Math.floor(s%60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
}

// ── Downloads ────────────────────────────────────────
function dlText(fmt) { if (sessionId) { closeDD(); window.open('/api/download/text/' + fmt + '/' + sessionId, '_blank'); } }
function dlAudio(fmt) { if (sessionId) { closeDD(); window.open('/api/download/audio/' + fmt + '/' + sessionId + '/' + currentVoice, '_blank'); } }

// ── Dropdowns ────────────────────────────────────────
function toggleDd(id) {
    var el = $(id), was = el.classList.contains('open');
    closeDD();
    if (!was) el.classList.add('open');
}
function closeDD() { document.querySelectorAll('.dropdown.open').forEach(function(d){d.classList.remove('open')}); }
document.addEventListener('click', function(e) { if (!e.target.closest('.dropdown')) closeDD(); });

// ── Toast ────────────────────────────────────────────
function toast(msg, type) {
    var t = document.createElement('div');
    t.className = 'toast toast-' + (type || 'success');
    t.textContent = msg;
    $('toastContainer').appendChild(t);
    setTimeout(function() { t.remove(); }, 4000);
}
