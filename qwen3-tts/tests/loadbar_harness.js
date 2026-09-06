// Minimal DOM stub so the real head script can be driven step by step.
var LOG = [];
function El(tag) {
  this.tag = tag; this.id = ''; this.title = ''; this.textContent = '';
  this.children = []; this.style = {}; this.dataset = {};
  this._classes = {};
  var self = this;
  this.classList = {
    add: function () { for (var i = 0; i < arguments.length; i++) self._classes[arguments[i]] = 1; },
    remove: function () { for (var i = 0; i < arguments.length; i++) delete self._classes[arguments[i]]; },
    toggle: function (c, on) { if (on) self._classes[c] = 1; else delete self._classes[c]; },
    contains: function (c) { return !!self._classes[c]; },
  };
}
El.prototype.classes = function () { return Object.keys(this._classes).sort().join(' '); };
Object.defineProperty(El.prototype, 'innerHTML', {
  set: function (h) {
    var m, re = /class="([^"]+)"/g, self = this;
    while ((m = re.exec(h))) { var c = new El('div'); c._classes[m[1].split(' ')[0]] = 1; self.children.push(c); }
  },
});
El.prototype.querySelector = function (sel) {
  var c = sel.replace('.', '');
  for (var i = 0; i < this.children.length; i++) if (this.children[i]._classes[c]) return this.children[i];
  return null;
};
El.prototype.getClientRects = function () { return [1]; };
El.prototype.appendChild = function (e) { this.children.push(e); DOC._all.push(e); };

var DOC = {
  readyState: 'complete', _all: [], body: null,
  createElement: function (t) { return new El(t); },
  addEventListener: function () {},
  querySelector: function (sel) { var r = this.querySelectorAll(sel); return r.length ? r[0] : null; },
  querySelectorAll: function (sel) {
    var out = [];
    sel.split(',').forEach(function (part) {
      part = part.trim();
      DOC._all.forEach(function (e) {
        if (part === '#gen-progress') { if (e.id === 'gen-progress') out.push(e); return; }
        if (part === '#chang-chime') { if (e.id === 'chang-chime') out.push(e); return; }
        if (part === '.toast-body.error') { if (e._classes['toast-body'] && e._classes.error) out.push(e); return; }
        if (part === '#lang-pick input:checked') return;
        var neg = part.indexOf(':not(.hide)') >= 0;
        var need = part.replace(':not(.hide)', '').split('.').filter(Boolean);
        var ok = need.every(function (c) { return e._classes[c]; });
        if (ok && (!neg || !e._classes.hide)) out.push(e);
      });
    });
    return out;
  },
};
DOC.body = new El('body');
DOC.body.contains = function (e) { return DOC._all.indexOf(e) >= 0; };
var document = DOC;

var STORE = {};
var localStorage = { getItem: function (k) { return k in STORE ? STORE[k] : null; },
                     setItem: function (k, v) { STORE[k] = String(v); } };

var POLL = null, TIMERS = [], NOW = 0;
function setInterval(fn) { POLL = fn; return 1; }
function setTimeout(fn, ms) { TIMERS.push({ fn: fn, at: NOW + ms }); return TIMERS.length; }
function clearTimeout(id) { if (id) TIMERS[id - 1] = null; }
function runTimers(advance) {
  NOW += advance;
  TIMERS.forEach(function (t, i) { if (t && t.at <= NOW) { TIMERS[i] = null; t.fn(); } });
}
var CHIMES = [];
function FakeOsc() { this.frequency = { setValueAtTime: function () {} }; }
FakeOsc.prototype.connect = function () {}; FakeOsc.prototype.start = function () {}; FakeOsc.prototype.stop = function () {};
function FakeGain() { var r = function () {};
  this.gain = { setValueAtTime: r, exponentialRampToValueAtTime: r }; }
FakeGain.prototype.connect = function () {};
function AudioContext() { this.state = 'running'; this.currentTime = 0;
  this.destination = {}; CHIMES.push('ctx'); }
AudioContext.prototype.createOscillator = function () { CHIMES.push('note'); return new FakeOsc(); };
AudioContext.prototype.createGain = function () { return new FakeGain(); };
AudioContext.prototype.resume = function () {};
var FAKENOW = 1000000;
Date.now = function () { return FAKENOW; };   // control "how long the run took"
function elapse(ms) { FAKENOW += ms; }
var window = this;
window.AudioContext = AudioContext;
window.Date = Date;

// Real clock is fine: the script measures elapsed with Date.now(), and the
// harness controls how long a "run" lasts by faking it below.
