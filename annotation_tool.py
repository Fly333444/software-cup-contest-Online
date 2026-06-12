# -*- coding: utf-8 -*-
"""
Web annotation tool for Fire Detection COCO dataset.
Loads 405 images + labels from A_train/Image + A_train/label.
Usage: python annotation_tool.py
Open http://localhost:5001 in browser.
"""
import json
import os
import threading
from flask import Flask, request, jsonify, render_template_string, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_A_TRAIN = os.path.join(BASE_DIR, 'A_train')
IMAGE_DIR = os.path.join(ROOT_A_TRAIN, 'Image')
LABEL_DIR = os.path.join(ROOT_A_TRAIN, 'label')

CATEGORIES = [
    {"id": 1, "name": "battery", "color": "#00FF00"},
    {"id": 2, "name": "board",   "color": "#FF0000"},
    {"id": 3, "name": "fire",    "color": "#FFA500"},
]
NAME2ID = {c['name']: c['id'] for c in CATEGORIES}
ID2COLOR = {c['id']: c['color'] for c in CATEGORIES}

app = Flask(__name__)

# Thread-safe cache
_lock = threading.Lock()
label_data = {}

def parse_labelme_json(fn, js):
    """Parse LabelMe JSON shapes into internal annotation format."""
    anns = []
    for s in js.get('shapes', []):
        label = s.get('label', '')
        if label not in NAME2ID:
            continue
        pts = s['points']
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
        anns.append({
            'category_id': NAME2ID[label],
            'bbox': [int(x), int(y), int(w), int(h)],
        })
    return anns

def load_all_labels():
    """Load all LabelMe JSON files from disk."""
    data = {}
    if not os.path.isdir(LABEL_DIR):
        return data
    for fn in sorted(os.listdir(LABEL_DIR)):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(LABEL_DIR, fn)
        with open(path, encoding='utf-8') as f:
            js = json.load(f)
        fname = js.get('imagePath', fn.replace('.json', '.jpg'))
        data[fname] = parse_labelme_json(fn, js)
    return data

def load_one_label(fn):
    """Reload a single label file from disk. Returns list of anns or None."""
    safe_fn = os.path.basename(fn)
    label_fn = safe_fn.replace('.jpg', '.json').replace('.png', '.json')
    path = os.path.join(LABEL_DIR, label_fn)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        js = json.load(f)
    return parse_labelme_json(fn, js)

# Initialize
label_data = load_all_labels()
image_list = sorted(
    fn for fn in os.listdir(IMAGE_DIR)
    if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
)

print(f"[Init] {len(image_list)} images, {len(label_data)} label files loaded")
counts = {}
for v in label_data.values():
    for a in v:
        c = a['category_id']
        counts[c] = counts.get(c, 0) + 1
for c in [1, 2, 3]:
    print(f"  {CATEGORIES[c-1]['name']}: {counts.get(c, 0)}")

HTML = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Annotation Tool - Fire Detection</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI',sans-serif; background: #1a1a2e; color: #eee; display: flex; height: 100vh; }
#sidebar { width: 260px; background: #16213e; padding: 10px; overflow-y: auto; flex-shrink: 0; }
#sidebar h3 { color: #e94560; margin: 8px 0 4px; font-size: 13px; }
#sidebar select, #sidebar input { width: 100%; padding: 5px; border: 1px solid #333; border-radius: 3px; background: #0f3460; color: #fff; font-size: 12px; margin-bottom: 6px; }
.img-list .item { padding: 5px 8px; cursor: pointer; border-radius: 3px; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.img-list .item:hover { background: #0f3460; }
.img-list .item.active { background: #e94560; }
.img-list .item .badge { font-size: 10px; }
#main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#toolbar { padding: 6px 12px; background: #16213e; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
#toolbar button { padding: 4px 12px; border: none; border-radius: 3px; cursor: pointer; font-size: 12px; }
.btn-del { background: #e94560; color: #fff; }
.btn-save { background: #0f3460; color: #fff; }
.btn-nav { background: #533483; color: #fff; }
.btn-reload { background: #b8860b; color: #fff; }
.btn-class { padding: 4px 10px; border-radius: 3px; cursor: pointer; border: 2px solid transparent; font-size: 12px; }
.btn-class.sel { border-color: #fff !important; }
#canvas-wrap { flex: 1; position: relative; overflow: auto; background: #000; display: flex; align-items: flex-start; justify-content: center; }
#imgContainer { position: relative; display: inline-block; }
#mainImg { display: block; max-width: none; }
#mainCanvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
#info { padding: 4px 12px; background: #16213e; font-size: 11px; color: #888; }
.hint { color: #e94560; }
</style>
</head>
<body>
<div id="sidebar">
  <h3>filter (missing class)</h3>
  <select id="missingFilter" onchange="applyFilter()">
    <option value="">all</option>
    <option value="1">no battery</option>
    <option value="2">no board</option>
    <option value="3">no fire</option>
  </select>
  <h3>search</h3>
  <input id="search" placeholder="filename..." oninput="applyFilter()">
  <h3>images <span id="imgCount"></span></h3>
  <div class="img-list" id="imgList" style="max-height:450px;overflow-y:auto"></div>
</div>
<div id="main">
  <div id="toolbar">
    <button class="btn-nav" onclick="nav(-1)">prev</button>
    <span id="posLabel" style="font-size:12px;min-width:70px"></span>
    <button class="btn-nav" onclick="nav(1)">next</button>
    <span style="color:#555">|</span>
    <button class="btn-class sel" id="btnCls1" style="background:#00FF0033;color:#0f0" onclick="setClass(1)">1:battery</button>
    <button class="btn-class" id="btnCls2" style="background:#FF000033;color:#f55" onclick="setClass(2)">2:board</button>
    <button class="btn-class" id="btnCls3" style="background:#FFA50033;color:#fa0" onclick="setClass(3)">3:fire</button>
    <span style="color:#555">|</span>
    <button class="btn-del" onclick="deleteSelected()">del selected</button>
    <button class="btn-save" onclick="saveCurrent()">save</button>
    <button class="btn-reload" onclick="reloadCurrent()">reload</button>
    <span id="saveStatus" style="font-size:11px;color:#0f0"></span>
  </div>
  <div id="canvas-wrap">
    <div id="imgContainer">
      <img id="mainImg" src="">
      <canvas id="mainCanvas"></canvas>
    </div>
  </div>
  <div id="info">
    <span id="infoFile"></span> |
    <span id="infoAnnots"></span>
    <span class="hint"> drag=draw | rightClick=select | dblClick=changeClass | del=remove | 1/2/3=class | arrows=navigate | Ctrl+S=save</span>
  </div>
</div>

<script>
var IMAGES = {{ images | tojson }};
var categories = {{ categories | tojson }};
var id2color = {};
categories.forEach(function(c) { id2color[c.id] = c.color; });

var currentIdx = -1;
var currentClass = 1;
var currentAnns = [];
var selectedBox = -1;
var drawing = false, drawStart = null;
var pendingSave = null;  // Promise of in-flight save
var dirty = false;

function applyFilter() {
    var q = (document.getElementById('search').value || '').toLowerCase();
    var missId = document.getElementById('missingFilter').value;
    var filtered = [];
    for (var i = 0; i < IMAGES.length; i++) {
        var fn = IMAGES[i];
        if (q && fn.toLowerCase().indexOf(q) === -1) continue;
        if (missId) {
            if (!window._counts) continue;
            var cts = window._counts[fn] || {};
            if (cts[parseInt(missId)]) continue;
        }
        filtered.push({idx: i, fn: fn});
    }
    document.getElementById('imgCount').textContent = '(' + filtered.length + ')';
    var html = '';
    for (var j = 0; j < filtered.length; j++) {
        var item = filtered[j];
        var cts = (window._counts || {})[item.fn] || {};
        var safeFn = item.fn.replace(/[&<>"']/g, function(c) {
            return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;'}[c];
        });
        var cls = item.idx === currentIdx ? 'active' : '';
        html += '<div class="item ' + cls + '" onclick="goto(' + item.idx + ')">' +
            safeFn +
            ' <span class="badge" style="color:#0f0">B' + (cts[1]||0) + '</span>' +
            ' <span class="badge" style="color:#f55">R' + (cts[2]||0) + '</span>' +
            ' <span class="badge" style="color:#fa0">F' + (cts[3]||0) + '</span>' +
            '</div>';
    }
    document.getElementById('imgList').innerHTML = html;
}

// -- reload from disk (server re-reads file) --
function reloadCurrent() {
    if (currentIdx < 0) return;
    var fn = IMAGES[currentIdx];
    fetch('/reload/' + encodeURIComponent(fn))
        .then(function(r) { return r.json(); })
        .then(function(data) {
            currentAnns = data.anns || [];
            window._counts[fn] = data.counts || {};
            selectedBox = -1;
            dirty = false;
            redraw();
            updateInfo();
            applyFilter();
            document.getElementById('saveStatus').textContent = ' reloaded from disk';
            setTimeout(function() { document.getElementById('saveStatus').textContent = ''; }, 1500);
        });
}

// -- load annotations for a file (from server disk) --
function loadAnns(fn) {
    return fetch('/reload/' + encodeURIComponent(fn))
        .then(function(r) { return r.json(); })
        .then(function(data) {
            window._counts[fn] = data.counts || {};
            return data.anns || [];
        });
}

// -- navigate to image --
function goto(idx) {
    if (idx < 0 || idx >= IMAGES.length) return;
    // Wait for any in-flight save + force save if dirty, then navigate
    var go = function() {
        var fn = IMAGES[idx];
        loadAnns(fn).then(function(anns) {
            currentIdx = idx;
            currentAnns = anns;
            selectedBox = -1;
            dirty = false;
            document.getElementById('saveStatus').textContent = '';
            document.getElementById('infoFile').textContent = fn;
            document.getElementById('posLabel').textContent = (idx+1) + ' / ' + IMAGES.length;
            updateInfo();
            var img = document.getElementById('mainImg');
            img.onload = function() {
                document.getElementById('mainCanvas').width = img.naturalWidth;
                document.getElementById('mainCanvas').height = img.naturalHeight;
                redraw();
            };
            img.onerror = function() {
                document.getElementById('infoAnnots').textContent = ' [IMAGE NOT FOUND]';
            };
            img.src = '/image/' + fn;
            applyFilter();
        });
    };

    if (dirty && currentIdx >= 0) {
        // Save current first, then navigate
        var fn = IMAGES[currentIdx];
        fetch('/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: fn, annotations: currentAnns})
        }).then(function() {
            dirty = false;
            go();
        }).catch(function() { go(); });
    } else {
        go();
    }
}

function redraw() {
    var canvas = document.getElementById('mainCanvas');
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < currentAnns.length; i++) {
        var a = currentAnns[i];
        var b = a.bbox;
        var color = id2color[a.category_id] || '#fff';
        ctx.strokeStyle = color;
        ctx.lineWidth = (i === selectedBox) ? 3 : 2;
        ctx.strokeRect(b[0], b[1], b[2], b[3]);
        var lbl = categories[a.category_id-1].name;
        ctx.font = 'bold 13px monospace';
        var tw = ctx.measureText(lbl).width + 4;
        ctx.fillStyle = color;
        ctx.fillRect(b[0], Math.max(0, b[1]-16), tw, 16);
        ctx.fillStyle = '#000';
        ctx.fillText(lbl, b[0]+2, Math.max(0, b[1]-4));
    }
}

function setClass(cid) {
    currentClass = cid;
    document.querySelectorAll('.btn-class').forEach(function(b) { b.classList.remove('sel'); });
    document.getElementById('btnCls'+cid).classList.add('sel');
}

function updateInfo() {
    var cts = {1:0,2:0,3:0};
    currentAnns.forEach(function(a) { cts[a.category_id]++; });
    document.getElementById('infoAnnots').textContent =
        'battery:' + cts[1] + ' board:' + cts[2] + ' fire:' + cts[3] + ' | total:' + currentAnns.length;
}

// -- Drawing --
var canvas = document.getElementById('mainCanvas');
canvas.addEventListener('mousedown', function(e) {
    if (e.button === 2) return;
    drawing = true;
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    drawStart = [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
});

canvas.addEventListener('mousemove', function(e) {
    if (!drawing) return;
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    var x = (e.clientX - rect.left) * scaleX, y = (e.clientY - rect.top) * scaleY;
    redraw();
    var ctx = canvas.getContext('2d');
    ctx.strokeStyle = id2color[currentClass];
    ctx.lineWidth = 2;
    ctx.setLineDash([5,5]);
    ctx.strokeRect(drawStart[0], drawStart[1], x-drawStart[0], y-drawStart[1]);
    ctx.setLineDash([]);
});

canvas.addEventListener('mouseup', function(e) {
    if (!drawing) return;
    drawing = false;
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    var x = (e.clientX - rect.left) * scaleX, y = (e.clientY - rect.top) * scaleY;
    var w = x - drawStart[0], h = y - drawStart[1];
    if (Math.abs(w) < 4 || Math.abs(h) < 4) { redraw(); return; }
    var bbox = [
        Math.max(0, Math.min(drawStart[0], x)),
        Math.max(0, Math.min(drawStart[1], y)),
        Math.abs(w), Math.abs(h)
    ];
    currentAnns.push({
        category_id: currentClass,
        bbox: bbox
    });
    selectedBox = currentAnns.length - 1;
    dirty = true;
    redraw();
    updateInfo();
    saveCurrent();
});

canvas.addEventListener('contextmenu', function(e) {
    e.preventDefault();
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    var mx = (e.clientX - rect.left) * scaleX, my = (e.clientY - rect.top) * scaleY;
    selectedBox = -1;
    for (var i = currentAnns.length-1; i >= 0; i--) {
        var b = currentAnns[i].bbox;
        if (mx >= b[0] && mx <= b[0]+b[2] && my >= b[1] && my <= b[1]+b[3]) {
            selectedBox = i; break;
        }
    }
    redraw();
});

canvas.addEventListener('dblclick', function(e) {
    var rect = canvas.getBoundingClientRect();
    var scaleX = canvas.width / rect.width;
    var scaleY = canvas.height / rect.height;
    var mx = (e.clientX - rect.left) * scaleX, my = (e.clientY - rect.top) * scaleY;
    for (var i = currentAnns.length-1; i >= 0; i--) {
        var b = currentAnns[i].bbox;
        if (mx >= b[0] && mx <= b[0]+b[2] && my >= b[1] && my <= b[1]+b[3]) {
            var cls = prompt('class: 1=battery 2=board 3=fire', currentAnns[i].category_id);
            if (cls && (cls === '1' || cls === '2' || cls === '3')) {
                currentAnns[i].category_id = parseInt(cls);
                dirty = true;
                redraw(); updateInfo();
                saveCurrent();
            }
            return;
        }
    }
});

function deleteSelected() {
    if (selectedBox < 0 || selectedBox >= currentAnns.length) return;
    currentAnns.splice(selectedBox, 1);
    selectedBox = -1;
    dirty = true;
    redraw(); updateInfo();
    saveCurrent();
}

// -- save to disk (async) --
function saveCurrent() {
    if (currentIdx < 0) return;
    var fn = IMAGES[currentIdx];
    // Strip any _mod markers and old ids
    var clean = currentAnns.map(function(a) {
        return {category_id: a.category_id, bbox: a.bbox.slice()};
    });
    pendingSave = fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: fn, annotations: clean})
    }).then(function(r) { return r.json(); }).then(function(d) {
        if (d.status === 'ok') {
            dirty = false;
            window._counts[fn] = d.counts || {};
            document.getElementById('saveStatus').textContent = ' saved';
            setTimeout(function() { document.getElementById('saveStatus').textContent = ''; }, 1500);
            applyFilter();
        } else {
            alert('save failed: ' + (d.error || ''));
        }
    }).catch(function(e) {
        console.error('save error:', e);
    });
}

function nav(dir) {
    var next = currentIdx + dir;
    if (next >= 0 && next < IMAGES.length) goto(next);
}

document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'Delete') { e.preventDefault(); deleteSelected(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); nav(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); nav(1); }
    else if (e.key === '1') setClass(1);
    else if (e.key === '2') setClass(2);
    else if (e.key === '3') setClass(3);
    else if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveCurrent(); }
});

// -- init: load counts for sidebar --
fetch('/counts')
    .then(function(r) { return r.json(); })
    .then(function(data) {
        window._counts = data;
        applyFilter();
        if (IMAGES.length > 0) goto(0);
    });
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML, images=image_list, categories=CATEGORIES)

@app.route('/counts')
def counts():
    """Return annotation counts for all images (for sidebar badges)."""
    result = {}
    with _lock:
        for fn, anns in label_data.items():
            cts = {}
            for a in anns:
                cid = a['category_id']
                cts[cid] = cts.get(cid, 0) + 1
            result[fn] = cts
    return jsonify(result)

@app.route('/reload/<path:filename>')
def reload(filename):
    """Re-read a single label file from disk and return its annotations."""
    anns = load_one_label(filename)
    cts = {}
    for a in anns:
        cid = a['category_id']
        cts[cid] = cts.get(cid, 0) + 1
    # Update memory cache too
    with _lock:
        label_data[filename] = anns
    return jsonify({'anns': anns, 'counts': cts})

@app.route('/image/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    fn = data['filename']
    anns = data['annotations']

    # Build LabelMe shapes
    shapes = []
    for a in anns:
        name = {1: 'battery', 2: 'board', 3: 'fire'}.get(a['category_id'], 'battery')
        b = a['bbox']
        shapes.append({
            'label': name,
            'score': None,
            'points': [[b[0], b[1]], [b[0]+b[2], b[1]], [b[0]+b[2], b[1]+b[3]], [b[0], b[1]+b[3]]],
            'group_id': None,
            'description': '',
            'difficult': False,
            'shape_type': 'rectangle',
            'flags': {},
            'attributes': {},
            'kie_linking': []
        })

    safe_fn = os.path.basename(fn)
    label_path = os.path.join(LABEL_DIR, safe_fn.replace('.jpg', '.json').replace('.png', '.json'))

    # Load existing raw JSON if exists, else create new
    if os.path.exists(label_path):
        with open(label_path, encoding='utf-8') as f:
            raw = json.load(f)
    else:
        raw = {'imagePath': fn, 'imageWidth': 1920, 'imageHeight': 1080}

    raw['shapes'] = shapes

    # Atomic write
    with open(label_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    os.replace(label_path + '.tmp', label_path)

    # Update memory cache
    with _lock:
        label_data[fn] = [{'category_id': a['category_id'], 'bbox': a['bbox']} for a in anns]

    # Return updated counts
    cts = {}
    for a in anns:
        cid = a['category_id']
        cts[cid] = cts.get(cid, 0) + 1

    return jsonify({'status': 'ok', 'counts': cts})

@app.route('/export_coco')
def export_coco():
    # Re-read all from disk for export
    all_data = load_all_labels()
    images = []
    annotations = []
    ann_id = 0
    for idx, fn in enumerate(image_list):
        images.append({'id': idx+1, 'file_name': fn, 'width': 1920, 'height': 1080})
        for a in all_data.get(fn, []):
            ann_id += 1
            annotations.append({
                'id': ann_id,
                'image_id': idx+1,
                'category_id': a['category_id'],
                'bbox': a['bbox'],
                'area': a['bbox'][2] * a['bbox'][3],
                'iscrowd': 0,
                'segmentation': [],
            })
    coco = {'images': images, 'annotations': annotations, 'categories': CATEGORIES}
    return jsonify(coco)


if __name__ == '__main__':
    print(f"\nAnnotation tool: http://localhost:5001")
    print("  Keys: 1/2/3=class | drag=draw | rightClick=select | del | Ctrl+S=save | arrows=navigate")
    app.run(host='0.0.0.0', port=5001, debug=False)
