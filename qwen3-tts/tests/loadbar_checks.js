
// ---------------- drive the state machine ----------------
var FAILS = [];
function ck(name, cond, extra) {
  if (cond) { LOG.push('  ok  ' + name); }
  else { LOG.push('FAIL  ' + name + '  ' + (extra || '')); FAILS.push(name); }
}
function bar() { return DOC._all.filter(function (e) { return e.id === 'chang-loadbar'; })[0]; }
function fill() { return bar().querySelector('.clb-fill'); }
function pillTxt() { return bar().querySelector('.clb-txt'); }

ck('bar created on start', !!bar());
ck('bar starts hidden', !bar().classList.contains('on'));

// --- 1. a Gradio tracker appears with no fraction (queued / no-progress event)
var wrap = new El('div');
wrap._classes.wrap = 1; wrap._classes.full = 1;
DOC._all.push(wrap);
POLL();
ck('bar turns on', bar().classList.contains('on'));
ck('indeterminate with no fraction', bar().classList.contains('indeterminate'));
ck('default label', pillTxt().textContent === 'Generating…', pillTxt().textContent);

// --- 2. a fraction + desc arrives
var pb = new El('div'); pb._classes['progress-bar'] = 1; pb.style.width = '42.5%';
var lv = new El('div'); lv._classes['progress-level-inner'] = 1; lv.textContent = 'Synthesizing part 2 of 5…';
wrap.children.push(pb, lv);
POLL();
ck('goes determinate', !bar().classList.contains('indeterminate'));
ck('fill follows the fraction', fill().style.width === '42.5%', fill().style.width);
ck('label follows the desc', pillTxt().textContent === 'Synthesizing part 2 of 5…', pillTxt().textContent);

// --- 3. a brief gap between two chained events must NOT end the run
wrap._classes.hide = 1;
POLL(); POLL();
ck('no finish after 2 quiet ticks', bar().classList.contains('on') && !bar().classList.contains('done'));
delete wrap._classes.hide;
POLL();
ck('run resumes', !bar().classList.contains('done'));

// --- 4. the event really completes -> bar fills, chime plays
elapse(2000);
wrap._classes.hide = 1;
POLL(); POLL(); POLL();
ck('marked done', bar().classList.contains('done'));
ck('done label', pillTxt().textContent === 'Done', pillTxt().textContent);
ck('silent before the error-check delay', CHIMES.length === 0);
runTimers(300);
ck('success chime played', CHIMES.filter(function (c) { return c === 'note'; }).length === 3,
   JSON.stringify(CHIMES));
runTimers(1200);
ck('bar hidden again', !bar().classList.contains('on'));
ck('fill reset', fill().style.width === '0%', fill().style.width);

// --- 5. a failed run chimes differently
CHIMES.length = 0;
delete wrap._classes.hide; POLL();
var toast = new El('div'); toast._classes['toast-body'] = 1; toast._classes.error = 1;
DOC._all.push(toast);
elapse(2000);
wrap._classes.hide = 1; POLL(); POLL(); POLL();
runTimers(300);
ck('error chime is 2 notes', CHIMES.filter(function (c) { return c === 'note'; }).length === 2,
   JSON.stringify(CHIMES));
DOC._all.splice(DOC._all.indexOf(toast), 1);
runTimers(1200);

// --- 6. muting silences it
CHIMES.length = 0;
window.changToggleChime();
ck('toggle writes localStorage', STORE.chang_chime === 'off', JSON.stringify(STORE));
delete wrap._classes.hide; POLL();
elapse(2000);
wrap._classes.hide = 1; POLL(); POLL(); POLL();
runTimers(300);
ck('muted run is silent', CHIMES.filter(function (c) { return c === 'note'; }).length === 0);
runTimers(1200);
STORE.chang_chime = 'on';

// --- 7. the Generator tab's own bar drives it too
CHIMES.length = 0;
DOC._all.splice(DOC._all.indexOf(wrap), 1);
var gen = new El('div'); gen.id = 'gen-progress';
gen.dataset.run = 'running'; gen.dataset.pct = '61.5'; gen.dataset.label = '苹果 — 12/20';
DOC._all.push(gen);
POLL();
ck('generator drives the fill', fill().style.width === '61.5%', fill().style.width);
ck('generator drives the label', pillTxt().textContent === '苹果 — 12/20', pillTxt().textContent);
elapse(2000);
gen.dataset.run = 'idle';
POLL(); POLL(); POLL();
runTimers(300);
ck('chimes when the queue finishes', CHIMES.filter(function (c) { return c === 'note'; }).length === 3);

// --- 8. a run shorter than the floor stays quiet
CHIMES.length = 0;
gen.dataset.run = 'running'; POLL();
elapse(400);
gen.dataset.run = 'idle'; POLL(); POLL(); POLL();
runTimers(300);
ck('sub-second run stays silent', CHIMES.filter(function (c) { return c === 'note'; }).length === 0);

LOG.push(FAILS.length ? '\nFAILED: ' + FAILS.join(', ') : '\nAll load-bar / chime checks passed.');
LOG.join('\n');
