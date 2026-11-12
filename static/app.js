// static/app.js
const GEOAI_SESSION = {
    user: document.body ? document.body.dataset.user || '' : '',
    role: document.body ? document.body.dataset.role || '' : '',
    roleTitle: document.body ? document.body.dataset.roleTitle || '' : '',
    capabilities: (document.body ? document.body.dataset.capabilities || '' : '').split(/\s+/).filter(Boolean)
};

function geoaiCan(capability) { return GEOAI_SESSION.capabilities.indexOf(capability) !== -1; }

(function installSessionGuard() {
    if (window.__geoaiFetchWrapped) return;
    const nativeFetch = window.fetch.bind(window);
    function sameOrigin(url) { try { return new URL(url, window.location.href).origin === window.location.origin; } catch (e) { return false; } }
    window.fetch = function (input, init) {
        const p = nativeFetch(input, init);
        if (typeof input !== 'string' && !(input instanceof URL)) return p;
        if (!sameOrigin(String(input))) return p;
        return p.then(function (resp) {
            if (resp.status === 401 && !window.__geoaiRedirecting) {
                window.__geoaiRedirecting = true;
                const here = window.location.pathname + window.location.search;
                window.location.href = '/login?next=' + encodeURIComponent(here);
            }
            return resp;
        });
    };
    window.__geoaiFetchWrapped = true;
})();

function geoaiApplyCapabilities() { document.querySelectorAll('[data-requires-capability]').forEach(function (el) { if (!geoaiCan(el.dataset.requiresCapability)) el.style.display = 'none'; }); }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', geoaiApplyCapabilities);
else geoaiApplyCapabilities();

function geoaiWsUrl(path) { const scheme = window.location.protocol === 'https:' ? 'wss://' : 'ws://'; return scheme + window.location.host + path; }

let MAP_SERVICES = { "satellite": {"name":"Satellite (Esri World Imagery)","url":"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}","attribution":"(c) Esri, Maxar","max_zoom":19,"type":"Satellite","resolution":"~0.3-1 m"} };
let MAP_RESIDENCY = { status: "UNKNOWN", iga_hosted: false, message: "Basemap source not yet confirmed by the server." };
let DEFAULT_MAP_SERVICE_KEY = "satellite";
let SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
let SATELLITE_ATTR = "(c) Esri, Maxar";
let mapServicesLoaded = false;
let basemapPolicyKnown = false;

async function loadMapServices() {
    try {
        const r = await fetch("/api/map-services");
        if (!r.ok) { SATELLITE_URL = null; return; }
        const data = await r.json();
        if (data && data.services && Object.keys(data.services).length) {
            MAP_SERVICES = data.services;
            const def = data.services[data.default] || Object.values(data.services)[0];
            if (def && def.url) { SATELLITE_URL = def.url; SATELLITE_ATTR = def.attribution || SATELLITE_ATTR; }
            DEFAULT_MAP_SERVICE_KEY = data.default || Object.keys(data.services)[0];
        }
        if (data && data.residency) MAP_RESIDENCY = data.residency;
        basemapPolicyKnown = true;
    } catch (e) { SATELLITE_URL = null; }
    finally { mapServicesLoaded = true; populateMapServiceSelect(); renderBasemapResidency(); }
}

function populateMapServiceSelect() {
    const sel = document.getElementById('mapServiceSelect');
    if (!sel) return;
    const keys = Object.keys(MAP_SERVICES);
    if (!keys.length) return;
    sel.innerHTML = '';
    for (const key of keys) {
        const s = MAP_SERVICES[key];
        const opt = document.createElement('option');
        opt.value = key; opt.textContent = (s.internal ? 'IGA - ' : 'PUBLIC - ') + (s.name || key);
        if (key === DEFAULT_MAP_SERVICE_KEY) opt.selected = true;
        sel.appendChild(opt);
    }
}

function renderBasemapResidency(activeKey) {
    const el = document.getElementById('basemapResidencyBadge');
    if (!el) return;
    let internal = !!MAP_RESIDENCY.iga_hosted;
    let name = MAP_RESIDENCY.default_name || '';
    let detail = MAP_RESIDENCY.message || '';
    const sel = document.getElementById('mapServiceSelect');
    const key = activeKey || (sel && sel.value) || MAP_RESIDENCY.default_key;
    if (key && MAP_SERVICES[key]) {
        internal = !!MAP_SERVICES[key].internal;
        name = MAP_SERVICES[key].name || key;
        detail = internal ? 'Imagery is served by the IGA Image Server.' : 'Imagery is served by a PUBLIC service.';
    }
    if (MAP_RESIDENCY.status === 'UNKNOWN' && !key) { el.textContent = 'BASEMAP ?'; el.style.background = '#334155'; el.style.color = '#cbd5e1'; el.title = detail; return; }
    el.textContent = internal ? 'BASEMAP: IGA' : 'BASEMAP: PUBLIC';
    el.style.background = internal ? '#064e3b' : '#7f1d1d';
    el.style.color = internal ? '#6ee7b7' : '#fecaca';
    el.title = detail;
}

function basemapLayer(extra) { if (!basemapPolicyKnown || !SATELLITE_URL) return null; return L.tileLayer(SATELLITE_URL, Object.assign({ attribution: SATELLITE_ATTR, maxZoom: 19, maxNativeZoom: 18, crossOrigin: true }, extra || {})); }

function showBasemapBlocked(container) {
    if (!container || container.querySelector('.basemap-blocked')) return;
    const d = document.createElement('div');
    d.className = 'basemap-blocked';
    d.style.cssText = 'position:absolute;z-index:500;top:8px;left:50%;transform:translateX(-50%);background:#7f1d1d;color:#fecaca;padding:6px 12px;border-radius:6px;font-size:12px;max-width:90%;text-align:center;pointer-events:none;';
    d.textContent = 'No basemap: the imagery policy could not be confirmed.';
    container.style.position = container.style.position || 'relative';
    container.appendChild(d);
}

const mapServicesReady = loadMapServices();

let map, mapPredict, evalMap;
let labelClusterAOILayer = null;
let drawnItems, drawControl;
let userAois = [];
let userAoisLayer = null;
let isDrawingAoiMode = false;
let currentAoiDrawer = null;
let selectedAnnotationLayer = null;
let projectCreateInFlight = false;
let predictDrawControl, predictDrawnItems;
let evalOverlays;
let evalDrawnItems = null;
let evalAoiBounds = null;
let ws = null;
let isTraining = false;
let currentProjectId = null;
let classNames = ['background'];
let allLabels = ['background'];
let currentLabel = 'st';
let drawnShapes = [];
let selectedProjectData = null;
let loadedModelPath = null;
let availableModels = [];
let currentTileLayer = null;
let currentPredictTileLayer = null;
let predictResultLayer = null;
let lastPredictMaskLayer = null;
let trainedModelPath = null;
let trainedModelMetrics = null;
let lastTrainingModelName = '';
let mapInitialized = false;
let predictMapInitialized = false;
let evalMapInitialized = false;
let statusCache = null;
let lastStatusUpdate = 0;
const STATUS_UPDATE_INTERVAL = 1000;
let isStatusUpdating = false;
let lossData = [];
let trainingStartTime = null;
let trainingTimerInterval = null;
let trainingStats = { epochs: 0, currentEpoch: 0, trainLoss: [], valLoss: [], iou: [], bestIou: 0, isComplete: false, isModelLoaded: false };
let availableAnnotationFiles = [];
let modalAnnotationFiles = [];
let selectedModalIndex = -1;
let loadedAnnotationFeatures = [];
let isAnnotationLoaded = false;
let aoiBounds = null;
let aoiLayer = null;
let projectsData = [];
let selectedSaveFolder = './models/saved';
let lastPredictionGeojson = null;
let lastPredictionShp = null;
let currentBackbone = 'unet';
let currentMode = 'binary';
let lastEvaluateMetrics = null;

const FIXED_CAPTURE_ZOOM = 17;
let trainingZoom = FIXED_CAPTURE_ZOOM;
let trainingGsd = null;
let trainingService = null;
window.__lastPredFeatures = null;
window.trainingState = window.trainingState || {};

function getCurrentModelType() { return (document.getElementById('modelTypeSelect')?.value) || 'unet'; }
function getSelectedProjectAnnotationModelType() {
    const project = selectedProjectData;
    if (project && currentProjectId && String(project.id) === String(currentProjectId)) { return normalizeAnnotationModelType(project.model_type_key || project.model_type || getCurrentModelType()); }
    return normalizeAnnotationModelType(getCurrentModelType());
}
function getAnnotationModelDisplayName(modelType) { const mt = normalizeAnnotationModelType(modelType); return ({ unet: 'U-Net', classic_cnn: 'Classic CNN', rcnn: 'R-CNN', fast_rcnn: 'Fast R-CNN', mask_rcnn: 'Mask R-CNN', yolo: 'YOLO' })[mt] || String(modelType || 'Model'); }
function isMaskRCNNModel(modelType) { const mt = String(modelType || '').trim().toLowerCase(); return mt === 'mask_rcnn' || mt === 'maskrcnn' || mt === 'mask-r-cnn'; }
function normalizeAnnotationModelType(modelType) { const mt = String(modelType || '').trim().toLowerCase().replace(/[- ]+/g, '_'); return ({ 'u_net': 'unet', 'faster_rcnn': 'rcnn', 'faster_r_cnn': 'rcnn', 'fast_r_cnn': 'fast_rcnn', 'mask_r_cnn': 'mask_rcnn', 'maskrcnn': 'mask_rcnn', 'cnn': 'classic_cnn' })[mt] || mt; }
function isBoxAnnotationModel(modelType) { const mt = normalizeAnnotationModelType(modelType || getCurrentModelType()); return mt === 'rcnn' || mt === 'fast_rcnn' || mt === 'yolo'; }
function isClassificationAnnotationModel(modelType) { return normalizeAnnotationModelType(modelType || getCurrentModelType()) === 'classic_cnn'; }
function getAnnotationTargetContract(modelType) {
    const mt = normalizeAnnotationModelType(modelType || getCurrentModelType());
    if (isClassificationAnnotationModel(mt)) return 'class_label';
    if (isBoxAnnotationModel(mt)) return 'boxes_labels';
    if (mt === 'mask_rcnn') return 'boxes_labels_masks';
    return 'semantic_mask';
}

function featureToBoundingBoxFeature(feat, extraProps) {
    if (!feat || !feat.geometry) return null;
    const coords = [];
    const walk = (v) => { if (!Array.isArray(v)) return; if (v.length >= 2 && Number.isFinite(Number(v[0])) && Number.isFinite(Number(v[1]))) { coords.push([Number(v[0]), Number(v[1])]); return; } v.forEach(walk); };
    walk(feat.geometry.coordinates);
    if (!coords.length) return null;
    let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
    coords.forEach(([x,y]) => { minX=Math.min(minX,x); minY=Math.min(minY,y); maxX=Math.max(maxX,x); maxY=Math.max(maxY,y); });
    if (![minX,minY,maxX,maxY].every(Number.isFinite) || maxX <= minX || maxY <= minY) return null;
    const ring = [[minX,minY],[maxX,minY],[maxX,maxY],[minX,maxY],[minX,minY]];
    const props = Object.assign({}, feat.properties || {}, extraProps || {}, { target_type: 'boxes_labels', annotation_geometry: 'bounding_box', shape_type: 'rectangle', bbox_lonlat: [minX,minY,maxX,maxY], source_geometry_type: feat.geometry.type });
    return { type:'Feature', geometry:{type:'Polygon',coordinates:[ring]}, properties:props };
}

function layerToAnnotationFeature(layer, props, modelType) {
    const mt = normalizeAnnotationModelType(modelType || getCurrentModelType());
    if (layer && typeof layer.getLatLngs === 'function' && typeof layer.getRadius !== 'function' && layer instanceof L.Polyline && !(layer instanceof L.Polygon)) {
        const latlngs = layer.getLatLngs();
        if (Array.isArray(latlngs) && latlngs.length >= 2 && latlngs[0] && typeof latlngs[0].lat === 'number') {
            const coordinates = latlngs.map(ll => [Number(ll.lng), Number(ll.lat)]);
            if (coordinates.length >= 2) {
                const lineFeature = { type: 'Feature', geometry: { type: 'LineString', coordinates }, properties: Object.assign({ label: currentLabel, class_name: currentLabel, shape_type: 'line', original_draw_type: 'polyline', created_at: new Date().toISOString() }, props || {}) };
                if (isBoxAnnotationModel(mt)) return featureToBoundingBoxFeature(lineFeature, props);
                return lineFeature;
            }
        }
    }
    const feature = layerToPolygonFeature(layer, props);
    if (!feature) return null;
    if (isBoxAnnotationModel(mt)) return featureToBoundingBoxFeature(feature, props);
    return feature;
}

function layerToPolygonRing(layer) {
    if (!layer) return null;
    if (typeof layer.getLatLngs === 'function' && typeof layer.getRadius !== 'function') {
        let lls = layer.getLatLngs();
        while (Array.isArray(lls) && Array.isArray(lls[0]) && Array.isArray(lls[0][0])) { lls = lls[0]; }
        if (Array.isArray(lls) && lls.length >= 3 && lls[0] && typeof lls[0].lat === 'number') {
            const ring = lls.map(ll => [Number(ll.lng), Number(ll.lat)]);
            if (ring.length >= 3) { const first = ring[0], last = ring[ring.length - 1]; if (first[0] !== last[0] || first[1] !== last[1]) ring.push(first); return ring; }
        }
    }
    if (typeof layer.getRadius === 'function' && typeof layer.getLatLng === 'function') {
        const ctr = layer.getLatLng(), r = layer.getRadius(), n = 64, ring = [];
        for (let i = 0; i < n; i++) { const a = (i / n) * 2 * Math.PI, dLat = r / 111320, dLng = r / (111320 * Math.cos(ctr.lat * Math.PI / 180)); ring.push([ctr.lng + dLng * Math.sin(a), ctr.lat + dLat * Math.cos(a)]); }
        ring.push(ring[0]); return ring;
    }
    if (typeof layer.getLatLng === 'function') {
        const ll = layer.getLatLng(), dLat = 0.5 / 111320, dLng = 0.5 / (111320 * Math.cos(ll.lat * Math.PI / 180));
        return [[ll.lng - dLng, ll.lat - dLat], [ll.lng + dLng, ll.lat - dLat], [ll.lng + dLng, ll.lat + dLat], [ll.lng - dLng, ll.lat + dLat], [ll.lng - dLng, ll.lat - dLat]];
    }
    if (typeof layer.getBounds === 'function') {
        try { const b = layer.getBounds(); if (b && b.isValid()) return [[b.getWest(), b.getSouth()], [b.getEast(), b.getSouth()], [b.getEast(), b.getNorth()], [b.getWest(), b.getNorth()], [b.getWest(), b.getSouth()]]; } catch (e) {}
    }
    return null;
}

function ringToPolygonGeometry(ring) { return (!ring || ring.length < 4) ? null : { type: 'Polygon', coordinates: [ring] }; }

function layerToPolygonFeature(layer, props) {
    if (!layer) return null;
    if (typeof layer.toGeoJSON === 'function') {
        try {
            const live = layer.toGeoJSON(), g = live && live.geometry;
            if (g && (g.type === 'Polygon' || g.type === 'MultiPolygon')) {
                const p = Object.assign({}, (layer.feature && layer.feature.properties) || {}, live.properties || {}, props || {});
                return { type: 'Feature', geometry: JSON.parse(JSON.stringify(g)), properties: Object.assign({ label: currentLabel, class_name: currentLabel, shape_type: g.type === 'MultiPolygon' ? 'multipolygon' : 'polygon' }, p) };
            }
        } catch (e) {}
    }
    if (layer.feature && layer.feature.geometry) {
        const g = layer.feature.geometry, p = Object.assign({}, layer.feature.properties || {}, props || {});
        if (g.type === 'Polygon' || g.type === 'MultiPolygon') { return { type: 'Feature', geometry: JSON.parse(JSON.stringify(g)), properties: Object.assign({ label: currentLabel, class_name: currentLabel, shape_type: g.type === 'MultiPolygon' ? 'multipolygon' : 'polygon' }, p) }; }
    }
    const isCircle = typeof layer.getRadius === 'function' && typeof layer.getLatLng === 'function';
    const ring = layerToPolygonRing(layer), geom = ringToPolygonGeometry(ring);
    if (!geom) return null;
    const properties = Object.assign({ label: currentLabel, class_name: currentLabel, shape_type: isCircle ? 'circle' : 'polygon', created_at: new Date().toISOString() }, props || {});
    if (isCircle) {
        const c = layer.getLatLng(), r = Number(layer.getRadius());
        properties.original_draw_type = 'circle'; properties.shape_type = 'circle'; properties.center_lon = Number(c.lng); properties.center_lat = Number(c.lat); properties.radius_m = r; properties.radius_meters = r;
    }
    return { type: 'Feature', geometry: geom, properties };
}

function featureToPolygonRing(feat) {
    if (!feat || !feat.geometry) return null;
    const g = feat.geometry, p = feat.properties || {};
    if (g.type === 'Polygon' && Array.isArray(g.coordinates) && g.coordinates[0]) return g.coordinates[0].map(c => [Number(c[0]), Number(c[1])]);
    if (g.type === 'MultiPolygon' && Array.isArray(g.coordinates) && g.coordinates[0] && g.coordinates[0][0]) return g.coordinates[0][0].map(c => [Number(c[0]), Number(c[1])]);
    if (g.type === 'Point' && Array.isArray(g.coordinates) && (p.radius || p.radius_m || p.radius_px)) {
        const lng = Number(g.coordinates[0]), lat = Number(g.coordinates[1]), r = Number(p.radius || p.radius_m || p.radius_px || 5), n = 64, ring = [];
        for (let i = 0; i < n; i++) { const a = (i / n) * 2 * Math.PI, dLat = r / 111320, dLng = r / (111320 * Math.cos(lat * Math.PI / 180)); ring.push([lng + dLng * Math.sin(a), lat + dLat * Math.cos(a)]); }
        ring.push(ring[0]); return ring;
    }
    if (g.type === 'Point' && Array.isArray(g.coordinates)) {
        const lng = Number(g.coordinates[0]), lat = Number(g.coordinates[1]), dLat = 0.5 / 111320, dLng = 0.5 / (111320 * Math.cos(lat * Math.PI / 180));
        return [[lng - dLng, lat - dLat], [lng + dLng, lat - dLat], [lng + dLng, lat + dLat], [lng - dLng, lat + dLat], [lng - dLng, lat - dLat]];
    }
    return null;
}

function toggleAdvanced(event) {
    if (event && typeof event.stopPropagation === 'function') event.stopPropagation();
    const panel = document.getElementById('advancedPanel'), btn = document.getElementById('advancedToggleBtn');
    if (!panel) return;
    const willOpen = !panel.classList.contains('open');
    panel.classList.toggle('open', willOpen);
    panel.style.display = willOpen ? 'block' : 'none';
    if (btn) { btn.setAttribute('aria-expanded', willOpen ? 'true' : 'false'); btn.innerHTML = willOpen ? '<i class="fas fa-sliders-h"></i> Advanced Regularisation ▴' : '<i class="fas fa-sliders-h"></i> Advanced Regularisation ▾'; }
}

function getAdvancedTrainingSettings() {
    return { weight_decay: parseFloat(document.getElementById('weightDecayInput')?.value || '0.0001'), dropout: parseFloat(document.getElementById('dropoutInput')?.value || '0'), grad_clip: parseFloat(document.getElementById('gradClipInput')?.value || '1.0'), early_stop: parseInt(document.getElementById('earlyStopInput')?.value || '0', 10), scheduler: document.getElementById('schedulerSelect')?.value || 'plateau', augment: document.getElementById('augmentSelect')?.value || 'basic' };
}

document.addEventListener('DOMContentLoaded', function() {
    initWebSocket();
    loadProjects();
    loadLabels();
    loadModels();
    updateStatus();
    setInterval(function() { const el = document.getElementById('footerTime'); if (el) el.textContent = new Date().toLocaleTimeString(); }, 1000);
    setInterval(updateStatus, STATUS_UPDATE_INTERVAL);
    const mSel = document.getElementById('mapServiceSelect'); if (mSel) mSel.addEventListener('change', changeMapService);
    const pSel = document.getElementById('projectSelect'); if (pSel) pSel.addEventListener('change', function() { if (this.value) selectProject(this.value); });
    const bSel = document.getElementById('backboneSelect'); if (bSel) bSel.addEventListener('change', function() { currentBackbone = this.value; });
    const modeSel = document.getElementById('modeSelect'); if (modeSel) modeSel.addEventListener('change', function() { currentMode = this.value; });
    const thr = document.getElementById('predictThreshold'); if (thr) thr.addEventListener('input', function() { const v = document.getElementById('predictThresholdVal'); if (v) v.textContent = this.value; const d = document.getElementById('predictThresholdDisplay'); if (d) d.textContent = this.value; });
    setTimeout(function() { if (document.getElementById('map')) initMapDirect(); if (document.getElementById('mapPredict')) initPredictMapDirect(); if (document.getElementById('mapEval')) initEvalMap(); }, 600);
    showDashboard();
    window.addEventListener('resize', function() { setTimeout(function() { if (map) map.invalidateSize(); if (mapPredict) mapPredict.invalidateSize(); if (evalMap) evalMap.invalidateSize(); }, 200); });
});

function initMapDirect() {
    if (!mapServicesLoaded) { mapServicesReady.then(() => initMapDirect()); return; }
    const c = document.getElementById('map'); if (!c) return;
    if (map) { try { map.invalidateSize(); return; } catch(e) { map = null; } }
    if (typeof L === 'undefined') { setTimeout(initMapDirect, 500); return; }
    try {
        c.innerHTML = '';
        map = L.map('map', {center: [25.0, 55.06], zoom: 12, preferCanvas: true});
        currentTileLayer = basemapLayer(); if (currentTileLayer) currentTileLayer.addTo(map); else showBasemapBlocked(c);
        drawnItems = L.featureGroup().addTo(map);
        userAoisLayer = L.featureGroup().addTo(map);
        drawControl = new L.Control.Draw({ position: 'topright', draw: { polygon: { allowIntersection: false, showArea: true, shapeOptions: { color: '#4ecdc4', weight: 3, fillColor: '#4ecdc4', fillOpacity: 0.4 } }, rectangle: { shapeOptions: { color: '#a78bfa', weight: 3, fillColor: '#a78bfa', fillOpacity: 0.4 } }, circle: { shapeOptions: { color: '#f87171', weight: 3, fillColor: '#f87171', fillOpacity: 0.4 } }, polyline: { shapeOptions: { color: '#38bdf8', weight: 4, opacity: 0.95 } }, marker: {draggable: true}, circlemarker: {radius: 6, shapeOptions: {color:'#fbbf24', weight:2, fillColor:'#fbbf24', fillOpacity:0.6}} }, edit: {featureGroup: drawnItems, remove: true} });
        map.addControl(drawControl);
        installAnnotationSelectionToolbar();
        map.on(L.Draw.Event.CREATED, function(ev) {
            if (isDrawingAoiMode) {
                handleAoiCreated(ev.layer);
                return;
            }
            const layer = ev.layer, t = ev.layerType; const annotationModelType = getSelectedProjectAnnotationModelType();
            const feat = layerToAnnotationFeature(layer, { original_draw_type: t, project_model_type: annotationModelType }, annotationModelType);
            if (!feat) { return; }
            layer.feature = feat; layer.properties = feat.properties;
            if (typeof layer.setStyle === 'function') { const isLine = feat.geometry && feat.geometry.type === 'LineString'; layer.setStyle(isLine ? { color: '#38bdf8', weight: 4, opacity: 0.95 } : { color: '#4ecdc4', weight: 2, opacity: 0.9, fillColor: '#4ecdc4', fillOpacity: 0.4 }); }
            bindAnnotationSelection(layer); drawnItems.addLayer(layer); drawnShapes.push(layer); updateObjectCount();
            const st = document.getElementById('annotationStatus'); if (st) { st.textContent = '📝 ' + drawnItems.getLayers().length + ' annotations on map'; st.style.color = '#fbbf24'; }
        });
        map.on(L.Draw.Event.DELETED, updateObjectCount);
        map.on('moveend', function() { const z = document.getElementById('zoomDisplay'); if (z) z.textContent = map.getZoom(); const ctr = map.getCenter(); fetch('/api/map/location', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({lat: ctr.lat, lng: ctr.lng, zoom: map.getZoom()}) }).catch(function(){}); });
        setTimeout(function() { if (map) { map.invalidateSize(); mapInitialized = true; } }, 400);
    } catch (e) { console.error('Map init error:', e); }
}

function initPredictMapDirect() {
    if (!mapServicesLoaded) { mapServicesReady.then(() => initPredictMapDirect()); return; }
    const c = document.getElementById('mapPredict'); if (!c) return;
    if (mapPredict) { try { mapPredict.invalidateSize(); return; } catch(e) { mapPredict = null; } }
    if (typeof L === 'undefined') { setTimeout(initPredictMapDirect, 500); return; }
    try {
        c.innerHTML = '';
        mapPredict = L.map('mapPredict', {center: [25.0, 55.06], zoom: 12, preferCanvas: true});
        currentPredictTileLayer = basemapLayer(); if (currentPredictTileLayer) currentPredictTileLayer.addTo(mapPredict); else showBasemapBlocked(c);
        predictDrawnItems = L.featureGroup().addTo(mapPredict);
        predictDrawControl = new L.Control.Draw({ position: 'topright', draw: { polygon: {allowIntersection: false, shapeOptions:{color:'#fbbf24',weight:3,fillOpacity:0.3}}, rectangle: {shapeOptions:{color:'#60a5fa',weight:3,fillOpacity:0.3}}, circle: {shapeOptions:{color:'#34d399',weight:3,fillOpacity:0.3}}, polyline: false, marker: false, circlemarker: false }, edit: {featureGroup: predictDrawnItems, remove: true} });
        mapPredict.addControl(predictDrawControl);
        mapPredict.on(L.Draw.Event.CREATED, function(ev) { predictDrawnItems.clearLayers(); predictDrawnItems.addLayer(ev.layer); aoiLayer = ev.layer; const b = ev.layer.getBounds(); if (b.isValid()) { aoiBounds = b; const s = document.getElementById('predictStatusText'); if (s) s.textContent = '📍 AOI set — click PREDICT'; } });
        mapPredict.on(L.Draw.Event.DELETED, function() { aoiBounds = null; aoiLayer = null; });
        setTimeout(function() { if (mapPredict) { mapPredict.invalidateSize(); predictMapInitialized = true; } }, 400);
    } catch (e) { console.error('Predict map init error:', e); }
}

function initEvalMap() {
    if (!mapServicesLoaded) { mapServicesReady.then(() => initEvalMap()); return; }
    const c = document.getElementById('mapEval'); if (!c) return;
    if (evalMap) { try { evalMap.invalidateSize(); return; } catch(e) { evalMap = null; } }
    if (typeof L === 'undefined') { setTimeout(initEvalMap, 500); return; }
    try {
        c.innerHTML = '';
        evalMap = L.map('mapEval', {center: [25.0, 55.06], zoom: 12, preferCanvas: true});
        const evalBase = basemapLayer(); if (evalBase) evalBase.addTo(evalMap); else showBasemapBlocked(c);
        evalOverlays = L.layerGroup().addTo(evalMap);
        if (!evalDrawnItems) evalDrawnItems = L.featureGroup().addTo(evalMap);
        const evalDrawControl = new L.Control.Draw({ position: 'topright', draw: { polygon: {allowIntersection: false, shapeOptions:{color:'#fbbf24',weight:3,fillOpacity:0.15}}, rectangle: {shapeOptions:{color:'#fbbf24',weight:3,fillOpacity:0.15}}, circle: false, polyline: false, marker: {draggable: true}, circlemarker: {radius: 6, shapeOptions: {color:'#fbbf24', weight:2, fillColor:'#fbbf24', fillOpacity:0.6}} }, edit: { featureGroup: evalDrawnItems, remove: true } });
        evalMap.addControl(evalDrawControl);
        evalMap.on(L.Draw.Event.CREATED, function(ev) { evalDrawnItems.clearLayers(); evalDrawnItems.addLayer(ev.layer); const b = ev.layer.getBounds(); if (b.isValid()) { evalAoiBounds = b; const es = document.getElementById('evalStatusText'); if (es) es.textContent = `📍 AOI set`; } });
        evalMap.on(L.Draw.Event.DELETED, function() { evalAoiBounds = null; });
        setTimeout(function() { if (evalMap) evalMap.invalidateSize(); }, 400);
        evalMapInitialized = true;
    } catch (e) { evalMapInitialized = false; }
}

function changeMapService() {
    const key = document.getElementById('mapServiceSelect').value, s = MAP_SERVICES[key];
    if (!s || !map) return;
    if (currentTileLayer) map.removeLayer(currentTileLayer);
    currentTileLayer = L.tileLayer(s.url, { attribution: s.attribution, maxZoom: s.max_zoom || 19, maxNativeZoom: Math.min(s.max_zoom || 19, 18), crossOrigin: true, subdomains: s.url.includes('{s}') ? ['a','b','c'] : undefined }).addTo(map);
    renderBasemapResidency(key);
}

function searchLocation() { const q = document.getElementById('searchInput').value.trim(); if (!q || !map) return; const m = q.match(/^([-+]?\d*\.?\d+)\s*[,;]\s*([-+]?\d*\.?\d+)$/); if (m) { map.flyTo([parseFloat(m[1]), parseFloat(m[2])], 14); return; } fetch('/api/geocode?q=' + encodeURIComponent(q)).then(r => r.json()).then(d => { if (d && d.length) map.flyTo([parseFloat(d[0].lat), parseFloat(d[0].lon)], 14); }); }
function searchPredictLocation() { const q = document.getElementById('predictSearchInput').value.trim(); if (!q || !mapPredict) return; const m = q.match(/^([-+]?\d*\.?\d+)\s*[,;]\s*([-+]?\d*\.?\d+)$/); if (m) { mapPredict.flyTo([parseFloat(m[1]), parseFloat(m[2])], 14); return; } fetch('/api/geocode?q=' + encodeURIComponent(q)).then(r => r.json()).then(d => { if (d && d.length) mapPredict.flyTo([parseFloat(d[0].lat), parseFloat(d[0].lon)], 14); }); }
function searchEvalLocation() { const q = document.getElementById('evalSearchInput').value.trim(); if (!q || !evalMap) return; const m = q.match(/^([-+]?\d*\.?\d+)\s*[,;]\s*([-+]?\d*\.?\d+)$/); if (m) { evalMap.flyTo([parseFloat(m[1]), parseFloat(m[2])], 14); return; } fetch('/api/geocode?q=' + encodeURIComponent(q)).then(r => r.json()).then(d => { if (d && d.length) evalMap.flyTo([parseFloat(d[0].lat), parseFloat(d[0].lon)], 14); }); }

function finalizeTrainingUI(d = {}) {
    isTraining = false; trainingStats.isComplete = true;
    const trainBtn = document.getElementById('trainBtn'), stopBtn = document.getElementById('stopBtn');
    if (trainBtn) { trainBtn.style.display = 'flex'; trainBtn.disabled = false; trainBtn.innerHTML = '<i class="fas fa-play"></i> TRAIN'; }
    if (stopBtn) stopBtn.style.display = 'none';
    const pct = document.getElementById('progressFill'); if (pct) pct.style.width = '100%';
    const percent = document.getElementById('runPercent'); if (percent) percent.textContent = '100%';
    const total = Number(d.total_epochs || trainingStats.epochs || d.epoch || 0), epoch = Number(d.epoch || total || 0);
    const epochEl = document.getElementById('runEpoch'); if (epochEl && total) epochEl.textContent = `${Math.min(epoch || total, total)}/${total}`;
    const best = Number(d.best_iou ?? d.best_map ?? trainingStats.bestIou ?? 0), metricName = d.metric_name || 'IoU';
    const statusEl = document.getElementById('runStatus');
    if (statusEl) { const bestEpoch = d.best_epoch ? ` at epoch ${d.best_epoch}` : ''; statusEl.textContent = `✅ Completed! Best ${metricName} ${best.toFixed(4)}${bestEpoch}`; }
    updateStatusBadge('completed'); stopTrainingTimer();
    const modelPath = d.model_path || (statusCache && statusCache.model_path) || trainedModelPath;
    if (modelPath) { trainedModelPath = modelPath; trainedModelMetrics = { best_iou: best }; showSaveModelSection(modelPath, trainedModelMetrics); }
    addLog('success', `Training complete! Best ${metricName}: ${best.toFixed(4)}`); loadModels();
}

function initWebSocket() {
    try {
        ws = new WebSocket(geoaiWsUrl('/ws/training'));
        ws.onmessage = function(ev) {
            try {
                const d = JSON.parse(ev.data);
                if (d && (d.status === 'training' || d.status === 'training_update' || d.status === 'training_progress' || d.status === 'epoch')) { const panel = document.getElementById('runProgress'); if (panel) panel.style.display = 'block'; }
                if (Array.isArray(d.train_losses)) trainingStats.trainLoss = d.train_losses.slice();
                if (Array.isArray(d.val_losses)) trainingStats.valLoss = d.val_losses.slice();
                if (Array.isArray(d.ious)) trainingStats.iou = d.ious.slice();
                if (d.status === 'training' || d.status === 'training_update' || d.status === 'training_progress' || d.status === 'epoch' || d.status === 'diag' || (isTraining && d.epoch !== undefined && (d.train_loss !== undefined || d.loss !== undefined))) {
                    if (d.epoch !== undefined && d.epoch !== null) {
                        const eE = document.getElementById('runEpoch'); if (eE) eE.textContent = d.epoch + '/' + (d.total_epochs || trainingStats.epochs || '?');
                        const liveTrainLoss = (d.train_loss !== undefined) ? d.train_loss : d.loss;
                        if (liveTrainLoss !== undefined && !isNaN(liveTrainLoss)) { const lE = document.getElementById('runLoss'); if (lE) lE.textContent = Number(liveTrainLoss).toFixed(4); }
                        const epochIndex = Math.max(0, Number(d.epoch || 1) - 1);
                        if (Array.isArray(d.train_losses)) trainingStats.trainLoss = d.train_losses.map(Number).filter(Number.isFinite);
                        if (liveTrainLoss !== undefined && Number.isFinite(Number(liveTrainLoss))) { trainingStats.trainLoss[epochIndex] = Number(liveTrainLoss); trainingStats.trainLoss = trainingStats.trainLoss.slice(0, epochIndex + 1); }
                        if (Array.isArray(d.val_losses)) trainingStats.valLoss = d.val_losses.map(Number).filter(Number.isFinite);
                        if (d.val_loss !== undefined && Number.isFinite(Number(d.val_loss))) { trainingStats.valLoss[epochIndex] = Number(d.val_loss); trainingStats.valLoss = trainingStats.valLoss.slice(0, epochIndex + 1); }
                        updateLossChart();
                        if (d.st_iou !== undefined || d.iou !== undefined || d.accuracy !== undefined) {
                            const v = (d.accuracy !== undefined) ? Number(d.accuracy) : Number((d.st_iou !== undefined) ? d.st_iou : d.iou);
                            const iE = document.getElementById('runIou'); if (iE) iE.textContent = v.toFixed(4);
                            const metricName = d.metric_name || 'IoU';
                            document.querySelectorAll('[data-training-metric-label]').forEach(el => el.textContent = metricName);
                            if (v > trainingStats.bestIou) { trainingStats.bestIou = v; const bE = document.getElementById('runBest'); if (bE) bE.textContent = v.toFixed(4); }
                        }
                        if (d.best_iou !== undefined) { const bE = document.getElementById('runBest'); if (bE) bE.textContent = Number(d.best_iou).toFixed(4); }
                        const bestEpochVal = (d.best_epoch !== undefined) ? d.best_epoch : null;
                        if (bestEpochVal !== null) { const be = document.getElementById('runBestEpoch'); if (be) be.textContent = bestEpochVal; }
                        const pct = (d.progress_percent !== undefined) ? d.progress_percent : d.progress;
                        if (pct !== undefined) { const pF = document.getElementById('progressFill'); if (pF) pF.style.width = pct + '%'; const pP = document.getElementById('runPercent'); if (pP) pP.textContent = Math.round(pct) + '%'; }
                    }
                }
                if (backendReportsTrainingComplete(d)) { finalizeTrainingUI(d); }
                if (d.status === 'stopped') { isTraining = false; document.getElementById('trainBtn').style.display = 'flex'; document.getElementById('stopBtn').style.display = 'none'; document.getElementById('runStatus').textContent = '⏹️ Stopped'; stopTrainingTimer(); updateStatusBadge('stopped'); }
                if (d.status === 'error' || d.status === 'failed') { isTraining = false; document.getElementById('trainBtn').style.display = 'flex'; document.getElementById('stopBtn').style.display = 'none'; document.getElementById('runStatus').textContent = '❌ Error'; stopTrainingTimer(); updateStatusBadge('error'); addLog('error', d.message || d.error || 'Training error'); }
            } catch(e) { console.error('[ws]', e); }
        };
        ws.onclose = function() { setTimeout(initWebSocket, 3000); };
    } catch(e) { setTimeout(initWebSocket, 3000); }
}

function updateLossChart() {
    const cv = document.getElementById('lossChart'); if (!cv) return;
    const parent = cv.parentElement;
    const progressPanel = document.getElementById('runProgress'); if (progressPanel && isTraining && getComputedStyle(progressPanel).display === 'none') progressPanel.style.display = 'block';
    const w = Math.max(120, parent.clientWidth || 300), h = Math.max(100, parent.clientHeight || 150), dpr = window.devicePixelRatio || 1;
    cv.width = Math.floor(w * dpr); cv.height = Math.floor(h * dpr); cv.style.width = w + 'px'; cv.style.height = h + 'px';
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    const values = (Array.isArray(trainingStats.trainLoss) ? trainingStats.trainLoss : []).map(Number).filter(Number.isFinite).slice(-500);
    if (!values.length) { ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('Waiting for epoch 1...', w / 2, h / 2); return; }
    const pad = { left: 8, right: 8, top: 10, bottom: 14 }, plotW = Math.max(1, w - pad.left - pad.right), plotH = Math.max(1, h - pad.top - pad.bottom);
    let mn = Math.min(...values), mx = Math.max(...values);
    if (Math.abs(mx - mn) < 1e-9) { const delta = Math.max(Math.abs(mx) * 0.05, 0.001); mn -= delta; mx += delta; }
    const range = mx - mn, xAt = i => values.length === 1 ? pad.left + plotW / 2 : pad.left + (plotW * i / (values.length - 1)), yAt = v => pad.top + plotH - ((v - mn) / range) * plotH;
    ctx.strokeStyle = 'rgba(100, 116, 139, 0.28)'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 3; i++) { const y = pad.top + plotH * i / 3; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke(); }
    ctx.strokeStyle = '#7dd3fc'; ctx.fillStyle = '#7dd3fc'; ctx.lineWidth = 2;
    ctx.beginPath(); values.forEach((v, i) => { const x = xAt(i), y = yAt(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    if (values.length > 1) ctx.stroke();
    const last = values.length - 1; ctx.beginPath(); ctx.arc(xAt(last), yAt(values[last]), 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#94a3b8'; ctx.font = '8px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(mx.toFixed(3), pad.left + 1, 1);
    ctx.textBaseline = 'bottom'; ctx.fillText(mn.toFixed(3), pad.left + 1, h - 13); ctx.textAlign = 'right'; ctx.fillText(`Epoch ${values.length}`, w - pad.right, h - 1);
}

(function installLossChartResizeHandler() {
    const attach = () => {
        const container = document.querySelector('.loss-chart-container'); if (!container || container.dataset.resizeObserved === '1') return;
        container.dataset.resizeObserved = '1';
        if (typeof ResizeObserver !== 'undefined') { const observer = new ResizeObserver(() => { if (typeof updateLossChart === 'function') updateLossChart(); }); observer.observe(container); }
        else window.addEventListener('resize', () => updateLossChart());
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', attach, {once:true}); else attach();
})();

function updateStatus() {
    if (isStatusUpdating) return;
    isStatusUpdating = true;
    fetch('/api/status')
        .then(r => r.json())
        .then(d => {
            statusCache = d;
            lastStatusUpdate = Date.now();
            applyStatusData(d);
        })
        .catch(e => {
            if (statusCache) applyStatusData(statusCache);
        })
        .finally(() => {
            isStatusUpdating = false;
        });
}

function backendReportsTrainingComplete(d) {
    if (!d) return false;
    const status = String(d.status || '').toLowerCase();
    if (d.training_completed === true || ['completed','complete','success','succeeded'].includes(status)) return true;
    const epoch = Number(d.epoch || 0), total = Number(d.total_epochs || 0);
    return d.is_training === false && total > 0 && epoch >= total && !['failed','error','stopped'].includes(status);
}

function applyStatusData(d) {
    if (backendReportsTrainingComplete(d)) { finalizeTrainingUI(d); }
    else if (d && (d.status === 'Failed' || d.status === 'failed')) { isTraining = false; const tb = document.getElementById('trainBtn'); if (tb) { tb.style.display = 'flex'; tb.disabled = false; } const sb = document.getElementById('stopBtn'); if (sb) sb.style.display = 'none'; const rs = document.getElementById('runStatus'); if (rs) rs.textContent = '❌ Training failed'; updateStatusBadge('error'); stopTrainingTimer(); }
    else if (d && (d.status === 'Stopped' || d.status === 'stopped')) { isTraining = false; const tb = document.getElementById('trainBtn'); if (tb) { tb.style.display = 'flex'; tb.disabled = false; } const sb = document.getElementById('stopBtn'); if (sb) sb.style.display = 'none'; const rs = document.getElementById('runStatus'); if (rs) rs.textContent = '⏹️ Stopped'; updateStatusBadge('stopped'); stopTrainingTimer(); }
    if (d && (isTraining || d.is_training || d.training_completed || d.status === 'Completed' || d.status === 'training' || d.status === 'training_update' || d.status === 'training_progress')) {
        const epochNum = Number(d.epoch);
        if (Array.isArray(d.train_losses) && d.train_losses.length >= trainingStats.trainLoss.length) trainingStats.trainLoss = d.train_losses.map(Number).filter(Number.isFinite);
        else if (Array.isArray(d.losses) && d.losses.length >= trainingStats.trainLoss.length) trainingStats.trainLoss = d.losses.map(Number).filter(Number.isFinite);
        const scalarLoss = d.train_loss !== undefined ? d.train_loss : d.loss;
        if (Number.isFinite(epochNum) && epochNum > 0 && scalarLoss !== undefined && Number.isFinite(Number(scalarLoss))) { trainingStats.trainLoss[epochNum - 1] = Number(scalarLoss); trainingStats.trainLoss = trainingStats.trainLoss.slice(0, epochNum); }
        if (Array.isArray(d.val_losses) && d.val_losses.length >= trainingStats.valLoss.length) trainingStats.valLoss = d.val_losses.map(Number).filter(Number.isFinite);
        if (Array.isArray(d.ious)) trainingStats.iou = d.ious.map(Number).filter(Number.isFinite);
        if (Number.isFinite(epochNum)) { const epochEl = document.getElementById('runEpoch'); if (epochEl) epochEl.textContent = `${epochNum}/${d.total_epochs || trainingStats.epochs || '?'}`; }
        if (trainingStats.trainLoss.length) { const lossEl = document.getElementById('runLoss'); if (lossEl) lossEl.textContent = Number(trainingStats.trainLoss[trainingStats.trainLoss.length - 1]).toFixed(4); const panel = document.getElementById('runProgress'); if (panel) panel.style.display = 'block'; requestAnimationFrame(() => updateLossChart()); }
    }
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    const nServices = Object.keys(MAP_SERVICES).length;
    
    // Update Dashboard Cards with actual counts from backend
    set('dashMapServices', nServices); 
    set('dashMapServices2', nServices); 
    set('dashAnnotations', d.total_annotations_all || 0); 
    set('dashAnnotations2', d.total_annotations_all || 0); 
    set('dashChips', d.total_chips_all || 0); 
    set('dashChips2', d.total_chips_all || 0); 
    set('dashModels', d.models_count || 0); 
    set('dashModels2', d.models_count || 0); 
    set('pubMapServices', nServices); 
    set('publicBadge', nServices); 
    set('projectBadge', d.projects_count || 0); 
    set('modelsTotal', d.models_count || 0); 
    set('modelsTrained', d.models_count || 0); 
    set('modelsBestIoU', (d.best_map_all || 0).toFixed(4)); 
    set('modelsPrecision', (d.best_map_all || 0).toFixed(4)); // NEW LINE
    set('projectsChips', d.total_chips_all || 0); 
    set('projectsModels', d.models_count || 0);

    if (d.selected_label && !isAnnotationLoaded) { const e = document.getElementById('activeLabelDisplay'); if (e) e.textContent = d.selected_label; currentLabel = d.selected_label; }
    if (d.model_backbone) currentBackbone = d.model_backbone;
    if (d.bounds) { window.trainingState = window.trainingState || {}; window.trainingState.bounds = d.bounds; }
    if (d.training_zoom !== undefined) trainingZoom = d.training_zoom;
    if (d.training_gsd !== undefined) trainingGsd = d.training_gsd;
    if (d.training_service !== undefined) trainingService = d.training_service;
    if (d.is_training) { isTraining = true; const tb = document.getElementById('trainBtn'); if (tb) tb.style.display = 'none'; const sb = document.getElementById('stopBtn'); if (sb) sb.style.display = 'flex'; const rp = document.getElementById('runProgress'); if (rp) rp.style.display = 'block'; updateStatusBadge('training'); }
    if (d.total_chips > 0) { const e = document.getElementById('datasetStatus'); if (e) e.innerHTML = '✅ ' + d.total_chips + ' chips'; }
    if (d.model_path && !loadedModelPath) { const ep = document.getElementById('evalModelPath'); if (ep) ep.textContent = d.model_path.split(/[\\/]/).pop(); }
    if (!isTraining && d.training_completed && d.model_path && !trainedModelPath) { trainedModelPath = d.model_path; trainedModelMetrics = { best_iou: d.best_map || 0 }; showSaveModelSection(trainedModelPath, trainedModelMetrics); }
}

function updateStatusBadge(s) {
    const b = document.getElementById('statusBadge'); if (!b) return;
    b.className = 'status-badge ' + s;
    b.textContent = ({ idle: 'IDLE', training: 'TRAINING', completed: 'COMPLETED', error: 'ERROR', stopped: 'STOPPED', evaluated: 'EVALUATED' })[s] || s.toUpperCase();
}

function updateObjectCount() { const c = drawnItems ? drawnItems.getLayers().length : 0; const e = document.getElementById('objectCount'); if (e) e.textContent = c; }

function addLog(type, msg) {
    const c = document.getElementById('logContent'); const t = new Date().toLocaleTimeString();
    console.log('[' + type.toUpperCase() + '] ' + msg); if (!c) return;
    const e = document.createElement('div'); e.className = 'log-entry log-' + type; e.innerHTML = '<span class="time">[' + t + ']</span> ' + msg;
    c.insertBefore(e, c.firstChild); while (c.children.length > 100) c.removeChild(c.lastChild);
}
function clearCurrentLog() { const e = document.getElementById('logContent'); if (e) e.innerHTML = ''; addLog('info', 'Log cleared'); }

async function importFineTuneModel() {
    const input = document.getElementById('fineTuneModelFile'), status = document.getElementById('fineTuneUploadStatus'), file = input?.files?.[0];
    if (!file) { alert('Choose a .pth, .pt, or .ckpt model first.'); return; }
    const fd = new FormData(); fd.append('file', file); status.textContent = 'Uploading and validating model...';
    try {
        const r = await fetch('/api/models/import', {method:'POST', body:fd}); const d = await r.json();
        if (!r.ok || !d.success) throw new Error(d.error || 'Model import failed');
        document.getElementById('fineTuneModelPath').value = d.model_path; status.textContent = 'Imported: ' + d.model_name + ' — original file preserved.'; status.style.color = '#34d399';
    } catch (e) { status.textContent = e.message; status.style.color = '#f87171'; }
}

function selectFineTuneExistingModel(path) {
    const hidden = document.getElementById('fineTuneModelPath'); if (hidden) hidden.value = path || '';
    const meta = document.getElementById('fineTuneModelMeta'), select = document.getElementById('fineTuneExistingModelSelect'), opt = select?.selectedOptions?.[0];
    if (meta) { meta.style.display = path ? 'block' : 'none'; meta.textContent = path ? (opt?.dataset?.meta || path) : ''; }
    
    if (path && opt) {
        const modelType = document.getElementById('modelTypeSelect'), backbone = document.getElementById('backboneSelect'), mode = document.getElementById('modeSelect');
        
        // Auto-select the correct architecture from the .dlpk
        if (modelType && [...modelType.options].some(o => o.value === opt.dataset.modelType)) modelType.value = opt.dataset.modelType;
        if (backbone && [...backbone.options].some(o => o.value === opt.dataset.backbone)) backbone.value = opt.dataset.backbone;
        if (mode && [...mode.options].some(o => o.value === opt.dataset.mode)) mode.value = opt.dataset.mode;
        
        // Force the UI to reflect that these are locked
        if (modelType) modelType.disabled = true;
        if (mode) mode.disabled = true;
        
        if (typeof updateModelHint === 'function') updateModelHint();
    } else {
        // Re-enable if they clear the selection
        const modelType = document.getElementById('modelTypeSelect'), mode = document.getElementById('modeSelect');
        if (modelType) modelType.disabled = false;
        if (mode) mode.disabled = false;
    }
}

async function loadFineTuneModelChoices() {
    const select = document.getElementById('fineTuneExistingModelSelect');
    if (!select) return;
    const previous = select.value; select.innerHTML = '<option value="">Loading all available models...</option>';
    try {
        const response = await fetch('/api/models/all', {cache:'no-store'});
        if (!response.ok) throw new Error(`Models list HTTP ${response.status}`);
        const data = await response.json(); const rows = (data && (data.models || data.rows)) || [];
        if (!data.success && !rows.length) throw new Error((data && data.error) || 'Could not list models');
        select.innerHTML = '<option value="">-- Select existing trained model --</option>';
        rows.forEach(m => {
            const path = m.filepath; if (!path) return;
            const tag = m.is_dlpk ? '.dlpk' : (m.source || 'project');
            const opt = new Option(`${m.model_name} | ${m.model_type || 'unknown'} | ${tag} | epoch ${m.epoch || 0}`, path);
            opt.dataset.meta = `Architecture: ${m.architecture || m.model_type || 'unknown'}; backbone: ${m.backbone || 'unknown'}; mode: ${m.mode || 'binary'}; tile: ${m.tile_size || 256}; source: ${tag}; label: ${m.label || 'unknown'}`;
            opt.dataset.modelType = m.model_type || 'unet'; opt.dataset.backbone = m.backbone || 'unet'; opt.dataset.mode = m.mode || 'binary'; opt.dataset.tileSize = String(m.tile_size || 256); opt.dataset.epoch = String(m.epoch || 0);
            select.appendChild(opt);
        });
        if (previous && [...select.options].some(o => o.value === previous)) select.value = previous;
        if (!rows.length) select.options[0].textContent = '-- No compatible saved models found --';
        selectFineTuneExistingModel(select.value);
    } catch (e) { select.innerHTML = '<option value="">Model list unavailable</option>'; console.error('Fine-tune model list:', e); }
}

function startTrainingOrFineTune() {
    const row = document.getElementById('fineTuneExistingModelRow');
    if (row && row.style.display !== 'none') return startFineTune();
    return startTraining();
}

function startFineTune() {
    const select = document.getElementById('fineTuneExistingModelSelect'), path = (select?.value || document.getElementById('fineTuneModelPath')?.value || '').trim();
    if (!path) { addLog('warning', 'Select an existing trained model in Step 5 first.'); alert('Select an existing trained model in Step 5 first.'); return; }
    if (!currentProjectId) { addLog('warning', 'Create/select the new-area project first.'); alert('Create/select the new-area project first.'); return; }
    const hiddenPath = document.getElementById('fineTuneModelPath'); if (hiddenPath) hiddenPath.value = path;
    const nameInput = document.getElementById('modelNameInput');
    if (nameInput && !nameInput.value.trim()) { const base = (currentLabel || 'model').toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'model'; nameInput.value = `${base}_finetuned`; }
    startTraining();
}

function switchWorkflow(p) {
    document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.workflow-panel').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.workflow-tab').forEach(t => t.classList.remove('active'));
    const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'flex';
    const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'flex';
    const ds = document.getElementById('dashboardStats'); if (ds) ds.style.display = 'grid';
    const dv = document.getElementById('dashboardView'); if (dv) dv.classList.remove('active');
    const tab = document.querySelector('.workflow-tab[data-panel="' + p + '"]'); if (tab) tab.classList.add('active');
    const panelKey = p === 'finetune' ? 'train' : p;
    const el = document.getElementById('panel-' + panelKey); if (el) el.classList.add('active');
    const fineMode = p === 'finetune';
    const step5Title = document.getElementById('step5Title'), trainText = document.getElementById('trainButtonText'), modelRow = document.getElementById('fineTuneExistingModelRow');
    if (step5Title) step5Title.textContent = fineMode ? 'FINE TUNE' : 'TRAIN';
    if (trainText) trainText.textContent = fineMode ? 'START FINE TUNING' : 'TRAIN'; // <-- THIS IS THE LINE
    if (modelRow) modelRow.style.display = fineMode ? 'flex' : 'none';
    if (fineMode) loadFineTuneModelChoices();
    const titles = {train:'TRAIN', evaluate:'EVALUATE', predict:'PREDICT', finetune:'FINE TUNE'};
    document.getElementById('pageTitle').textContent = titles[p] || 'Image Segmentation';
    document.getElementById('pageSubtitle').textContent = 'Deep Learning / ' + (titles[p] || '');
    if (p === 'train' || p === 'finetune') { setTimeout(function() { if (!mapInitialized || !map) initMapDirect(); else try { map.invalidateSize(); } catch(e) { initMapDirect(); } updateLossChart(); }, 200); }
    if (p === 'evaluate') { setTimeout(function() { if (!evalMapInitialized || !evalMap) initEvalMap(); else try { evalMap.invalidateSize(); } catch(e) { initEvalMap(); } setTimeout(function() { if (evalMap) evalMap.invalidateSize(); }, 400); refreshEvaluateModelStatus(); }, 200); }
    if (p === 'predict') {
        setTimeout(function() {
            if (!predictMapInitialized || !mapPredict) initPredictMapDirect();
            else { try { mapPredict.invalidateSize(); } catch(e) { initPredictMapDirect(); } }
        }, 200);
    }
    setTimeout(function() { if (map) map.invalidateSize(); if (mapPredict) mapPredict.invalidateSize(); if (evalMap) evalMap.invalidateSize(); updateLossChart(); }, 700);
}

function showDashboard() {
    document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active'));
    const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'flex';
    const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'flex';
    const ds = document.getElementById('dashboardStats'); if (ds) ds.style.display = 'grid';
    document.querySelectorAll('.workflow-panel').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.workflow-tab').forEach(t => t.classList.remove('active'));
    const trainTab = document.querySelector('.workflow-tab[data-panel="train"]'); if (trainTab) trainTab.classList.add('active');
    const trainPanel = document.getElementById('panel-train'); if (trainPanel) trainPanel.classList.add('active');
    document.getElementById('pageTitle').textContent = 'TRAIN';
    document.getElementById('pageSubtitle').textContent = 'Deep Learning / TRAIN';
    loadProjects();
    if (statusCache) applyStatusData(statusCache); else updateStatus();
    setTimeout(function() { if (!mapInitialized || !map) initMapDirect(); else try { map.invalidateSize(); } catch(e) { initMapDirect(); } updateLossChart(); }, 200);
}

function showProjects() { document.querySelectorAll('.workflow-panel').forEach(x => x.classList.remove('active')); document.querySelectorAll('.workflow-tab').forEach(t => t.classList.remove('active')); document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active')); const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'none'; const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'none'; document.getElementById('projectsView').classList.add('active'); document.getElementById('pageTitle').textContent = 'Projects'; document.getElementById('pageSubtitle').textContent = 'Home / Projects'; loadProjects(); }
function showModels() { document.querySelectorAll('.workflow-panel').forEach(x => x.classList.remove('active')); document.querySelectorAll('.workflow-tab').forEach(t => t.classList.remove('active')); document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active')); const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'none'; const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'none'; document.getElementById('yourModelsView').classList.add('active'); document.getElementById('pageTitle').textContent = 'Your Models'; document.getElementById('pageSubtitle').textContent = 'Home / Models'; loadModels(); }
function showPublicDatasets() { document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active')); const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'none'; const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'none'; document.getElementById('publicDatasetsView').classList.add('active'); document.getElementById('pageTitle').textContent = 'Public Datasets'; document.getElementById('pageSubtitle').textContent = 'Datasets / Public'; refreshPublicDatasets(); }
function showUploadedDatasets() { document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active')); const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'none'; const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'none'; document.getElementById('uploadedDatasetsView').classList.add('active'); document.getElementById('pageTitle').textContent = 'Uploaded Datasets'; document.getElementById('pageSubtitle').textContent = 'Datasets / Uploaded'; loadUploadedDatasets(); }
function showPurchasedDatasets() { document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active')); const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'none'; const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'none'; document.getElementById('purchasedDatasetsView').classList.add('active'); document.getElementById('pageTitle').textContent = 'Purchased Datasets'; document.getElementById('pageSubtitle').textContent = 'Datasets / Purchased'; }
function showDownload() { document.querySelectorAll('.datasets-view').forEach(v => v.classList.remove('active')); const wc = document.querySelector('.workflow-content'); if (wc) wc.style.display = 'none'; const tabs = document.getElementById('workflowTabs'); if (tabs) tabs.style.display = 'none'; document.getElementById('downloadView').classList.add('active'); document.getElementById('pageTitle').textContent = 'Download GIS Analysis'; document.getElementById('pageSubtitle').textContent = 'Home / Download'; }

function loadProjects() {
    return fetch('/api/projects').then(r => r.json()).then(d => {
        projectsData = d.projects || [];
        const sel = document.getElementById('projectSelect');
        if (sel) { const cur = sel.value; sel.innerHTML = '<option value="">-- Select Project --</option>'; projectsData.forEach(p => { sel.innerHTML += `<option value="${p.id}">${p.name} (${p.chips_count || 0} chips)</option>`; }); if (cur && projectsData.some(p => p.id === cur)) sel.value = cur; }
        const list = document.getElementById('projectsList');
        if (list) {
            list.innerHTML = '';
            if (projectsData.length === 0) list.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:40px;color:#64748b;">No projects yet.</div>';
            else { projectsData.forEach(p => { const chips = p.chips_count || 0, labels = p.labels || ['background'], tz = p.training_zoom ? ` | z=${p.training_zoom}` : ''; list.innerHTML += `<div class="project-card"><div class="project-header"><h4><i class="fas fa-folder-open" style="color:#4ecdc4;"></i> ${p.name}</h4></div><div class="project-details"><span><span class="label">🧩</span> ${chips} chips${tz}</span><span><span class="label">🏷️</span> ${labels.join(', ')}</span></div><div class="project-actions"><button class="btn btn-info" onclick="selectProject('${p.id}')"><i class="fas fa-edit"></i> Open</button><button class="btn btn-danger" onclick="deleteProject('${p.id}')"><i class="fas fa-trash"></i></button></div></div>`; }); }
        }
        const badge = document.getElementById('projectBadge'); if (badge) badge.textContent = projectsData.length;
        return projectsData;
    }).catch(e => { console.error(e); return []; });
}

function showCreateProject() { document.getElementById('createProjectModal').classList.add('active'); }
function closeCreateProject(force = false) { if (projectCreateInFlight && !force) return; document.getElementById('createProjectModal').classList.remove('active'); }
function deleteProject(pid) { if (!confirm('Delete project?')) return; fetch('/api/projects/' + pid, {method:'DELETE'}).then(r => r.json()).then(d => { if (d.success) { addLog('success','Deleted'); loadProjects(); } }); }

async function createProjectFromNewModal() {
    if (projectCreateInFlight) return;
    const name = document.getElementById('modalProjectName').value.trim();
    if (!name) { addLog('warning','Enter project name'); return; }
    const labels = (document.getElementById('modalLabels')?.value || 'st').split(',').map(l => l.trim()).filter(Boolean); if (!labels.length) labels.push('st');
    const modal = document.querySelector('#createProjectModal .modal'), button = document.getElementById('createProjectBtn'), busy = document.getElementById('createProjectBusy'), cancel = document.querySelector('#createProjectModal .btn-cancel');
    projectCreateInFlight = true;
    if (button) { button.disabled = true; button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating…'; } if (busy) busy.style.display = 'flex'; if (modal) modal.setAttribute('aria-busy', 'true'); if (cancel) cancel.disabled = true;
    try {
        const response = await fetch('/api/projects', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name, task_type: document.getElementById('modalTaskType')?.value, dataset_type: document.getElementById('modalDatasetType')?.value, model_type: document.getElementById('modalModelType')?.value, labels, description: document.getElementById('modalDescription')?.value }) });
        const d = await response.json();
        if (!response.ok || !d.success || !d.project?.id) throw new Error(d.error || `Server returned HTTP ${response.status}`);
        addLog('success', 'Project created: ' + name); if (busy) busy.innerHTML = '<i class="fas fa-check-circle"></i><span>Project created. Loading project…</span>';
        await loadProjects(); await selectProject(d.project.id); closeCreateProject(true); switchWorkflow('train');
    } catch (e) { addLog('error', 'Create project failed: ' + (e.message || e)); if (busy) { busy.innerHTML = '<i class="fas fa-exclamation-circle"></i><span>Could not create project. Please retry.</span>'; busy.style.display = 'flex'; } }
    finally { projectCreateInFlight = false; if (button) { button.disabled = false; button.innerHTML = '<i class="fas fa-plus-circle"></i> Create Project'; } if (modal) modal.removeAttribute('aria-busy'); if (cancel) cancel.disabled = false; if (busy && !busy.querySelector('.fa-exclamation-circle')) busy.style.display = 'none'; }
}

async function selectProject(projectId) {
    if (!projectId) return; isAnnotationLoaded = false;
    try { await fetch('/api/projects/' + projectId + '/select', { method:'POST', headers:{'Content-Type':'application/json'} }); } catch(e) {}
    try {
        const r = await fetch('/api/projects/' + projectId), d = await r.json();
        if (!d.success) return;
        currentProjectId = projectId; selectedProjectData = d.project; classNames = d.project.labels || ['background']; allLabels = [...classNames];
        if (d.project.training_zoom) { trainingZoom = d.project.training_zoom; trainingGsd = d.project.training_gsd; trainingService = d.project.training_service; }
        const firstLabel = classNames.find(l => l !== 'background' && l !== 'default') || 'st'; currentLabel = firstLabel;
        document.getElementById('projectSelect').value = projectId; document.getElementById('projectInfo').style.display = 'block'; document.getElementById('projectName').textContent = d.project.name; document.getElementById('projectLabels').textContent = classNames.join(', '); document.getElementById('projectAnnCount').textContent = '0'; document.getElementById('projectChipCount').textContent = d.project.chips_count || 0; document.getElementById('evalProjectName').textContent = d.project.name; document.getElementById('activeLabelDisplay').textContent = currentLabel; document.getElementById('objectLabel').value = currentLabel;
        refreshLabelDropdown(); if (drawnItems) { drawnItems.clearLayers(); drawnShapes = []; updateObjectCount(); }
        loadAnnotationList(projectId); addLog('success', 'Loaded project: ' + d.project.name);
    } catch(e) { addLog('error', 'Select error: ' + e.message); }
}

function refreshLabelDropdown() {
    const sel = document.getElementById('labelDropdown'); if (!sel) return;
    const labels = new Set();
    if (selectedProjectData && selectedProjectData.labels) selectedProjectData.labels.forEach(l => { if (l && l !== 'background') labels.add(l); });
    if (allLabels && allLabels.length) allLabels.forEach(l => { if (l && l !== 'background' && l !== 'default' && l !== 'unknown') labels.add(l); });
    if (currentLabel && currentLabel !== 'background') labels.add(currentLabel);
    const sorted = Array.from(labels).sort(); sel.innerHTML = '';
    if (sorted.length === 0) { const o = document.createElement('option'); o.value = ''; o.textContent = '(no labels yet — click ＋ New Label)'; sel.appendChild(o); }
    else sorted.forEach(l => { const o = document.createElement('option'); o.value = l; o.textContent = l; if (l === currentLabel) o.selected = true; sel.appendChild(o); });
    const newOpt = document.createElement('option'); newOpt.value = '__new__'; newOpt.textContent = '＋ New Label…'; newOpt.style.color = '#4ecdc4'; sel.appendChild(newOpt);
    if (currentLabel && sel.querySelector(`option[value="${currentLabel}"]`)) sel.value = currentLabel;
    sel.onchange = function() { if (this.value === '__new__') { if (currentLabel && sel.querySelector(`option[value="${currentLabel}"]`)) this.value = currentLabel; const name = prompt('Enter new label name:'); if (name && name.trim()) createNewLabel(name.trim()); return; } if (this.value && this.value !== currentLabel) selectLabel(this.value); };
}

function createNewLabel(name) {
    if (!currentProjectId) { addLog('warning', 'Select a project first'); alert('Please select or create a project first.'); return; }
    const safe = name.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '');
    if (!safe) { addLog('warning', 'Invalid label name'); return; }
    fetch('/api/labels/create', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ label_name: safe, project_id: currentProjectId }) })
    .then(r => r.json()).then(d => {
        if (!d.success) { addLog('error', 'Failed: ' + (d.error || 'unknown')); return; }
        if (!allLabels.includes(safe)) allLabels.push(safe); if (selectedProjectData) { if (!selectedProjectData.labels) selectedProjectData.labels = ['background']; if (!selectedProjectData.labels.includes(safe)) selectedProjectData.labels.push(safe); } if (!classNames.includes(safe)) classNames.push(safe);
        refreshLabelDropdown(); addLog('success', 'Created label: ' + safe); loadLabels();
    }).catch(e => addLog('error', 'Create label error: ' + e.message));
}

function selectLabel(name) {
    if (!name || name === '__new__') return;
    currentLabel = name; document.getElementById('activeLabelDisplay').textContent = name; document.getElementById('objectLabel').value = name;
    fetch('/api/labels/select', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({label_name: name}) }).catch(function(){});
}

function loadLabels() { fetch('/api/labels').then(r => r.json()).then(d => { if (d.success && d.labels) allLabels = d.labels; refreshLabelDropdown(); }).catch(function(){}); }

function getActiveAnnotationLabel() {
    const dropdown = document.getElementById('labelDropdown'), selected = dropdown && typeof dropdown.value === 'string' ? dropdown.value.trim() : '';
    if (selected && selected !== '__new__' && !['background', 'default', 'unknown', 'unlabeled'].includes(selected.toLowerCase())) {
        currentLabel = selected; const objectInput = document.getElementById('objectLabel'); if (objectInput) objectInput.value = selected; const display = document.getElementById('activeLabelDisplay'); if (display) display.textContent = selected; return selected;
    }
    const candidates = [ (typeof currentLabel !== 'undefined' ? currentLabel : ''), document.getElementById('objectLabel')?.value, document.getElementById('activeLabel')?.value, document.getElementById('activeLabelDisplay')?.textContent ];
    const label = candidates.find(v => typeof v === 'string' && v.trim() && !['background', 'default', 'unknown', 'unlabeled', '__new__'].includes(v.trim().toLowerCase()));
    return label ? label.trim() : 'annotation';
}

async function previewLabelClusterAOIs() {
    const status = document.getElementById('aoiBuildStatus');
    try {
        if (!map) throw new Error('Map is not initialized.'); if (!drawnItems || drawnItems.getLayers().length === 0) throw new Error('Load or draw annotations before previewing AOIs.');
        const activeLabelName = getActiveAnnotationLabel(), normalizeLabel = value => String(value || '').trim().toLocaleLowerCase(), features = [];
        drawnItems.getLayers().forEach(layer => {
            const existingProps = (layer.feature && layer.feature.properties) || {}, layerLabel = String(existingProps.class_name || existingProps.label || existingProps.class || existingProps.label_name || activeLabelName).trim();
            if (normalizeLabel(layerLabel) !== normalizeLabel(activeLabelName)) return;
            const feature = layerToAnnotationFeature(layer, Object.assign({}, existingProps, {label: layerLabel, class_name: layerLabel}), getSelectedProjectAnnotationModelType());
            if (feature && feature.geometry) features.push(feature);
        });
        if (!features.length) throw new Error(`No annotations found for selected label "${activeLabelName}".`);
        if (status) status.textContent = 'Calculating selected-label extrema and 1-metre boundary…';
        const response = await fetch('/api/dataset/plan-aoi', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ features, cluster_distance_m: Number(document.getElementById('aoiClusterMeters')?.value ?? 10), aoi_buffer_m: 1, min_cluster_annotations: 8, active_label: activeLabelName }) });
        const result = await response.json(); if (!response.ok || !result.success) throw new Error(result.error || 'AOI planning failed.');
        const primaryAOI = Array.isArray(result.aois) && result.aois.length ? result.aois[0] : null; if (!primaryAOI || !primaryAOI.bounds) throw new Error('AOI planner did not return a usable boundary.');
        clearLabelClusterAOIs();
        window.activeLabelAOI = { active_label: activeLabelName, geometry: primaryAOI.geometry || null, bounds: primaryAOI.bounds, annotation_count: result.annotation_count || primaryAOI.annotation_count || 0, buffer_m: Number(result.aoi_buffer_m || 1), extrema_m: result.label_extrema_m || null };
        if (!map.getPane('aoiOutlinePane')) { map.createPane('aoiOutlinePane'); map.getPane('aoiOutlinePane').style.zIndex = 650; map.getPane('aoiOutlinePane').style.pointerEvents = 'auto'; }
        const aoiRenderer = L.svg({pane: 'aoiOutlinePane', padding: 0.5}); labelClusterAOILayer = L.featureGroup();
        const aois = Array.isArray(result.aois) ? result.aois : [];
        aois.forEach(aoi => {
            const style = { pane: 'aoiOutlinePane', renderer: aoiRenderer, color: '#ff3b30', weight: 3, opacity: 1, fillColor: '#ff3b30', fillOpacity: 0.025, dashArray: '8 6', lineCap: 'butt', lineJoin: 'round', interactive: true };
            const layer = aoi.geometry ? L.geoJSON(aoi.geometry, {pane: 'aoiOutlinePane', renderer: aoiRenderer, style: () => style, interactive: true}) : L.rectangle([[aoi.bounds.south, aoi.bounds.west], [aoi.bounds.north, aoi.bounds.east]], style);
            if (typeof layer.eachLayer === 'function') layer.eachLayer(child => { if (child.bindTooltip) child.bindTooltip(`${aoi.annotation_count || 0} ${activeLabelName} annotations`, {sticky: true, direction: 'top'}); });
            labelClusterAOILayer.addLayer(layer);
        });
        if (!aois.length && features.length) { const fallback = L.geoJSON({type: 'FeatureCollection', features}, { pane: 'aoiOutlinePane', renderer: aoiRenderer, style: {color: '#ff3b30', weight: 3, opacity: 1, fillOpacity: 0, dashArray: '8 6', interactive: true} }); labelClusterAOILayer.addLayer(fallback); }
        labelClusterAOILayer.addTo(map); try { if (labelClusterAOILayer.eachLayer) labelClusterAOILayer.eachLayer(group => { if (group.bringToFront) group.bringToFront(); if (group.eachLayer) group.eachLayer(child => { if (child.bringToFront) child.bringToFront(); }); }); if (labelClusterAOILayer.bringToFront) labelClusterAOILayer.bringToFront(); map.invalidateSize(true); } catch (_) {}
        return window.activeLabelAOI;
    } catch (e) { if (typeof addLog === 'function') addLog('error', e.message || 'AOI preview failed.'); }
}

function clearLabelClusterAOIs() { if (labelClusterAOILayer && map) { try { map.removeLayer(labelClusterAOILayer); } catch (_) {} } labelClusterAOILayer = null; window.activeLabelAOI = null; }

function countAnnotationsInBounds(bounds, labelName) {
    if (!drawnItems) return 0;
    const normalize = v => String(v || '').trim().toLowerCase();
    const targetLabel = normalize(labelName);
    const aoiLatLngBounds = L.latLngBounds([bounds.south, bounds.west], [bounds.north, bounds.east]);
    let labelCount = 0, totalInside = 0;
    drawnItems.getLayers().forEach(l => {
        const b = l.getBounds ? l.getBounds() : (l.getLatLng ? L.latLngBounds(l.getLatLng(), l.getLatLng()) : null);
        if (!b) return;
        const c = b.getCenter ? b.getCenter() : null;
        const inside = (c && c.lat >= bounds.south && c.lat <= bounds.north && c.lng >= bounds.west && c.lng <= bounds.east) || aoiLatLngBounds.intersects(b);
        if (!inside) return;
        totalInside++;
        const props = (l.feature && l.feature.properties) || {};
        const lLabel = normalize(props.class_name || props.label || props.class || props.label_name);
        if (!targetLabel || targetLabel === 'unlabeled' || targetLabel === 'default' || !lLabel || lLabel === targetLabel) {
            labelCount++;
        }
    });
    return labelCount > 0 ? labelCount : totalInside;
}

function toggleDrawAoiMode() {
    if (!map) return;
    isDrawingAoiMode = !isDrawingAoiMode;
    const btn = document.getElementById('btnDrawAoi');
    if (isDrawingAoiMode) {
        if (btn) btn.classList.add('btn-aoi-active');
        if (btn) btn.innerHTML = '<i class="fas fa-crosshairs"></i> Drag on Map...';
        addLog('info', 'AOI Drawing Mode: Click and drag on the map to define an Area of Interest');
        if (currentAoiDrawer) { currentAoiDrawer.disable(); }
        currentAoiDrawer = new L.Draw.Rectangle(map, {
            shapeOptions: {
                color: '#f59e0b',
                weight: 3,
                dashArray: '6 4',
                fillColor: '#f59e0b',
                fillOpacity: 0.12
            }
        });
        currentAoiDrawer.enable();
    } else {
        if (btn) btn.classList.remove('btn-aoi-active');
        if (btn) btn.innerHTML = '<i class="fas fa-vector-square"></i> + Draw AOI';
        if (currentAoiDrawer) {
            currentAoiDrawer.disable();
            currentAoiDrawer = null;
        }
    }
}

function handleAoiCreated(layer) {
    if (!userAoisLayer) {
        userAoisLayer = L.featureGroup().addTo(map);
    }
    const b = layer.getBounds();
    if (!b || !b.isValid()) return;
    const aoiId = 'aoi_' + Date.now();
    const aoiName = 'AOI #' + (userAois.length + 1);
    const bounds = { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() };
    const activeLabelName = getActiveAnnotationLabel();
    const count = countAnnotationsInBounds(bounds, activeLabelName);
    
    layer.setStyle({
        color: '#f59e0b',
        weight: 3,
        dashArray: '6 4',
        fillColor: '#f59e0b',
        fillOpacity: 0.12
    });

    const aoiObj = {
        id: aoiId,
        name: aoiName,
        bounds: bounds,
        geometry: layerToGeoJSONGeometry(layer),
        layer: layer,
        annotation_count: count,
        active_label: activeLabelName
    };

    bindAoiEvents(aoiObj);
    userAoisLayer.addLayer(layer);
    userAois.push(aoiObj);

    toggleDrawAoiMode();
    renderAoiList();
    addLog('success', `Created ${aoiName} with ${count} annotations. You can draw more AOIs in other regions.`);
}

function layerToGeoJSONGeometry(layer) {
    if (typeof layer.toGeoJSON === 'function') {
        const gj = layer.toGeoJSON();
        return gj.geometry || null;
    }
    const b = layer.getBounds();
    return {
        type: 'Polygon',
        coordinates: [[
            [b.getWest(), b.getSouth()],
            [b.getEast(), b.getSouth()],
            [b.getEast(), b.getNorth()],
            [b.getWest(), b.getNorth()],
            [b.getWest(), b.getSouth()]
        ]]
    };
}

function bindAoiEvents(aoi) {
    const layer = aoi.layer;
    if (!layer) return;
    
    const countText = aoi.annotation_count > 0 ? `${aoi.annotation_count} objects` : `AOI extent`;
    layer.bindTooltip(`${aoi.name} (${countText})`, {
        permanent: true,
        direction: 'top',
        className: 'aoi-map-label'
    });

    const popupContent = document.createElement('div');
    popupContent.style.cssText = 'font-size:11px;min-width:140px;';
    popupContent.innerHTML = `
        <div style="font-weight:700;color:#f59e0b;margin-bottom:4px;">${aoi.name}</div>
        <div style="color:#94a3b8;margin-bottom:6px;">${countText}</div>
        <div style="display:flex;gap:4px;">
            <button class="btn btn-secondary btn-xs" onclick="zoomToAoi('${aoi.id}')" title="Zoom to this AOI"><i class="fas fa-search-plus"></i> Zoom</button>
            <button class="btn btn-info btn-xs" onclick="toggleEditAoi('${aoi.id}')" title="Edit boundary"><i class="fas fa-pen"></i> Edit</button>
            <button class="btn btn-danger btn-xs" onclick="deleteUserAoi('${aoi.id}')" title="Delete AOI"><i class="fas fa-trash"></i></button>
        </div>
    `;
    layer.bindPopup(popupContent);
}

function renderAoiList() {
    const container = document.getElementById('aoiListContainer');
    const list = document.getElementById('aoiList');
    const badge = document.getElementById('aoiCountBadge');
    const header = document.getElementById('aoiListHeader');
    if (!container || !list) return;

    if (userAois.length === 0) {
        container.style.display = 'none';
        if (badge) badge.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    if (badge) {
        badge.style.display = 'inline-block';
        badge.textContent = `${userAois.length} AOI${userAois.length > 1 ? 's' : ''}`;
    }
    if (header) {
        header.textContent = `Active AOIs (${userAois.length}):`;
    }

    list.innerHTML = '';
    userAois.forEach((aoi) => {
        const isEditing = aoi.layer && aoi.layer.editing && aoi.layer.editing.enabled && aoi.layer.editing.enabled();
        const item = document.createElement('div');
        item.className = 'aoi-item' + (isEditing ? ' active-editing' : '');
        const countDesc = aoi.annotation_count > 0 ? `(${aoi.annotation_count} objects)` : `(extent)`;
        item.innerHTML = `
            <div class="aoi-info" title="${aoi.name}">
                <i class="fas fa-vector-square" style="color:#f59e0b;font-size:10px;"></i>
                <span style="font-weight:600;color:#f1f5f9;">${aoi.name}</span>
                <span style="color:#38bdf8;font-size:9px;">${countDesc}</span>
            </div>
            <div class="aoi-actions">
                <button type="button" class="btn btn-secondary btn-xs" style="padding:1px 5px;height:20px;font-size:9px;" onclick="zoomToAoi('${aoi.id}')" title="Zoom to AOI">
                    <i class="fas fa-search-plus"></i>
                </button>
                <button type="button" class="btn ${isEditing ? 'btn-success' : 'btn-info'} btn-xs" style="padding:1px 5px;height:20px;font-size:9px;" onclick="toggleEditAoi('${aoi.id}')" title="${isEditing ? 'Done Editing' : 'Edit Boundary'}">
                    <i class="fas ${isEditing ? 'fa-check' : 'fa-pen'}"></i>
                </button>
                <button type="button" class="btn btn-danger btn-xs" style="padding:1px 5px;height:20px;font-size:9px;" onclick="deleteUserAoi('${aoi.id}')" title="Delete AOI">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        list.appendChild(item);
    });
}

function zoomToAoi(aoiId) {
    const aoi = userAois.find(a => a.id === aoiId);
    if (!aoi || !aoi.bounds || !map) return;
    map.fitBounds([[aoi.bounds.south, aoi.bounds.west], [aoi.bounds.north, aoi.bounds.east]], { padding: [40, 40], maxZoom: 18 });
}

function toggleEditAoi(aoiId) {
    const aoi = userAois.find(a => a.id === aoiId);
    if (!aoi || !aoi.layer) return;
    
    if (aoi.layer.editing) {
        if (aoi.layer.editing.enabled && aoi.layer.editing.enabled()) {
            aoi.layer.editing.disable();
            const b = aoi.layer.getBounds();
            aoi.bounds = { west: b.getWest(), south: b.getSouth(), east: b.getEast(), north: b.getNorth() };
            aoi.geometry = layerToGeoJSONGeometry(aoi.layer);
            aoi.annotation_count = countAnnotationsInBounds(aoi.bounds, aoi.active_label);
            if (aoi.layer.setTooltipContent) {
                aoi.layer.setTooltipContent(`${aoi.name} (${aoi.annotation_count} ${aoi.active_label})`);
            }
            renderAoiList();
            addLog('success', `Updated ${aoi.name}: ${aoi.annotation_count} annotations.`);
        } else {
            userAois.forEach(a => { if (a.id !== aoiId && a.layer && a.layer.editing && a.layer.editing.enabled && a.layer.editing.enabled()) a.layer.editing.disable(); });
            aoi.layer.editing.enable();
            renderAoiList();
            addLog('info', `Editing ${aoi.name}: drag corners to resize, then click Done checkmark.`);
        }
    }
}

function deleteUserAoi(aoiId) {
    const aoi = userAois.find(a => a.id === aoiId);
    if (!aoi) return;
    if (aoi.layer && userAoisLayer) {
        userAoisLayer.removeLayer(aoi.layer);
    }
    userAois = userAois.filter(a => a.id !== aoiId);
    userAois.forEach((a, idx) => {
        a.name = 'AOI #' + (idx + 1);
        if (a.layer && a.layer.setTooltipContent) {
            a.layer.setTooltipContent(`${a.name} (${a.annotation_count} ${a.active_label})`);
        }
    });
    renderAoiList();
    addLog('info', 'Deleted AOI.');
}

function clearAllUserAois() {
    if (userAois.length === 0) {
        addLog('info', 'No AOIs to clear.');
        return;
    }
    if (!confirm(`Clear all ${userAois.length} AOIs?`)) return;
    if (userAoisLayer) {
        userAoisLayer.clearLayers();
    }
    userAois = [];
    renderAoiList();
    addLog('info', 'Cleared all AOIs.');
}

function toggleClusterOptions(forceState) {
    const panel = document.getElementById('clusterAoiOptions');
    if (!panel) return;
    if (forceState !== undefined) {
        panel.style.display = forceState ? 'block' : 'none';
    } else {
        panel.style.display = (panel.style.display === 'none' || !panel.style.display) ? 'block' : 'none';
    }
}

async function executeClusterAOIs() {
    if (!map) return;
    if (!drawnItems || drawnItems.getLayers().length === 0) {
        addLog('warning', 'Load or draw annotations before clustering AOIs.');
        return;
    }
    const dist = Number(document.getElementById('clusterDistMeters')?.value || 150);
    const buf = Number(document.getElementById('clusterBufferMeters')?.value || 25);
    const minAnn = Number(document.getElementById('clusterMinLabels')?.value || 5);
    const inViewOnly = document.getElementById('clusterInCurrentViewOnly')?.checked ?? true;
    const activeLabelName = getActiveAnnotationLabel();

    addLog('info', `Clustering ${activeLabelName} annotations (dist=${dist}m, buffer=${buf}m)...`);

    const features = [];
    const normalizeLabel = value => String(value || '').trim().toLowerCase();
    drawnItems.getLayers().forEach(layer => {
        const existingProps = (layer.feature && layer.feature.properties) || {};
        const layerLabel = String(existingProps.class_name || existingProps.label || existingProps.class || existingProps.label_name || activeLabelName).trim();
        if (normalizeLabel(layerLabel) !== normalizeLabel(activeLabelName)) return;
        const feature = layerToAnnotationFeature(layer, Object.assign({}, existingProps, {label: layerLabel, class_name: layerLabel}), getSelectedProjectAnnotationModelType());
        if (feature && feature.geometry) features.push(feature);
    });

    if (!features.length) {
        addLog('error', `No annotations found matching "${activeLabelName}".`);
        return;
    }

    let boundsClip = null;
    if (inViewOnly && map) {
        const mb = map.getBounds();
        boundsClip = { north: mb.getNorth(), south: mb.getSouth(), east: mb.getEast(), west: mb.getWest() };
    }

    try {
        const res = await fetch('/api/dataset/plan-aoi', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                features,
                cluster: true,
                cluster_distance_m: dist,
                aoi_buffer_m: buf,
                min_cluster_annotations: minAnn,
                active_label: activeLabelName,
                bounds: boundsClip
            })
        });
        const result = await res.json();
        if (!res.ok || !result.success) throw new Error(result.error || 'Clustering failed.');
        
        const aois = Array.isArray(result.aois) ? result.aois : [];
        if (!aois.length) {
            addLog('warning', 'No dense clusters found with current settings. Try decreasing cluster distance or min labels.');
            return;
        }

        if (!userAoisLayer) userAoisLayer = L.featureGroup().addTo(map);

        aois.forEach(c => {
            const aoiId = 'aoi_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6);
            const aoiName = 'AOI #' + (userAois.length + 1);
            const bounds = c.bounds;
            const style = { color: '#f59e0b', weight: 3, opacity: 1, fillColor: '#f59e0b', fillOpacity: 0.1, dashArray: '6 4' };
            const layer = c.geometry ? L.geoJSON(c.geometry, { style: () => style }) : L.rectangle([[bounds.south, bounds.west], [bounds.north, bounds.east]], style);
            
            const targetLayer = layer.getLayers ? (layer.getLayers()[0] || layer) : layer;
            
            const aoiObj = {
                id: aoiId,
                name: aoiName,
                bounds: bounds,
                geometry: c.geometry || null,
                layer: targetLayer,
                annotation_count: c.annotation_count || 0,
                active_label: activeLabelName
            };

            bindAoiEvents(aoiObj);
            userAoisLayer.addLayer(targetLayer);
            userAois.push(aoiObj);
        });

        toggleClusterOptions(false);
        renderAoiList();
        
        if (userAoisLayer.getBounds().isValid()) {
            map.fitBounds(userAoisLayer.getBounds(), { padding: [30, 30] });
        }
        addLog('success', `⚡ Generated ${aois.length} cluster AOIs (${result.total_annotations_clustered || 0} annotations total).`);
    } catch (e) {
        addLog('error', 'Cluster AOI failed: ' + e.message);
    }
}

function saveAnnotations() {
    if (!currentProjectId) { addLog('warning','Select a project'); return; }
    if (!drawnItems || drawnItems.getLayers().length === 0) { addLog('warning','No shapes drawn'); return; }
    const projectModelType = getSelectedProjectAnnotationModelType(), modelDisplayName = getAnnotationModelDisplayName(projectModelType), annotationTarget = getAnnotationTargetContract(projectModelType);
    const features = []; let skipped = 0;
    drawnItems.getLayers().forEach(layer => { const feat = layerToAnnotationFeature(layer, { project_id: currentProjectId, project_model_type: projectModelType, target_type: annotationTarget }, projectModelType); if (!feat) { skipped++; return; } features.push(feat); });
    if (features.length === 0) { addLog('warning', 'No valid annotation features to save'); return; }
    const curZoom = map ? map.getZoom() : null, saveButton = document.querySelector('[onclick*="saveAnnotations"]'); if (saveButton) saveButton.disabled = true;
    fetch('/api/projects/' + currentProjectId + '/annotations', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ features, label: currentLabel, project_id: currentProjectId, model_type_key: projectModelType, annotation_target: annotationTarget, capture_zoom: curZoom, capture_service: document.getElementById('mapServiceSelect')?.value || null }) })
    .then(async r => { const d = await r.json(); if (!r.ok || !d.success) throw new Error(d.error || `Save failed`); return d; })
    .then(d => { addLog('success', `Saved ${d.num_features || features.length} annotations to ${d.geojson_path || 'project'}.`); const count = document.getElementById('projectAnnCount'); if (count) count.textContent = d.num_features || features.length; loadAnnotationList(currentProjectId); })
    .catch(e => addLog('error', 'Save failed: ' + (e.message || e))).finally(() => { if (saveButton) saveButton.disabled = false; });
}

function installAnnotationSelectionToolbar() {
    if (!map || map.getContainer().querySelector('#annotationSelectionToolbar')) return;
    const bar = L.DomUtil.create('div', 'annotation-selection-toolbar'); bar.id = 'annotationSelectionToolbar';
    bar.innerHTML = `<button type="button" id="editSelectedAnnotation" title="Edit selected shape"><i class="fas fa-pen"></i> Edit</button><button type="button" id="deleteSelectedAnnotation" title="Delete selected shape"><i class="fas fa-trash"></i> Delete</button><button type="button" id="clearSelectedAnnotation" title="Deselect shape"><i class="fas fa-times"></i></button>`;
    map.getContainer().appendChild(bar); L.DomEvent.disableClickPropagation(bar); L.DomEvent.disableScrollPropagation(bar);
    bar.querySelector('#editSelectedAnnotation').addEventListener('click', editSelectedAnnotation); bar.querySelector('#deleteSelectedAnnotation').addEventListener('click', deleteSelectedAnnotation); bar.querySelector('#clearSelectedAnnotation').addEventListener('click', clearSelectedAnnotation);
}
function bindAnnotationSelection(layer) { if (!layer || layer._annotationSelectionBound) return; layer._annotationSelectionBound = true; layer.on('click', function(ev) { if (ev && ev.originalEvent) L.DomEvent.stopPropagation(ev.originalEvent); selectAnnotationLayer(layer); }); }
function selectAnnotationLayer(layer) {
    if (!layer || !drawnItems || !drawnItems.hasLayer(layer)) return;
    if (selectedAnnotationLayer && selectedAnnotationLayer !== layer) restoreAnnotationStyle(selectedAnnotationLayer);
    selectedAnnotationLayer = layer;
    if (typeof layer.setStyle === 'function') { if (!layer._selectionOriginalStyle) layer._selectionOriginalStyle = { color: layer.options.color, weight: layer.options.weight, opacity: layer.options.opacity, fillColor: layer.options.fillColor, fillOpacity: layer.options.fillOpacity }; layer.setStyle({color:'#fbbf24', weight:5, opacity:1, fillColor:'#fbbf24', fillOpacity:0.28}); if (layer.bringToFront) layer.bringToFront(); }
    const bar = document.getElementById('annotationSelectionToolbar'); if (bar) bar.classList.add('has-selection');
}
function restoreAnnotationStyle(layer) {
    if (!layer) return;
    if (layer.editing && layer.editing.enabled && layer.editing.enabled()) layer.editing.disable();
    if (layer._selectionOriginalStyle && layer.setStyle) { const st = layer._selectionOriginalStyle; Object.keys(st).forEach(k => { if (st[k] === undefined) delete st[k]; }); layer.setStyle(st); delete layer._selectionOriginalStyle; }
}
function clearSelectedAnnotation() { if (selectedAnnotationLayer) restoreAnnotationStyle(selectedAnnotationLayer); selectedAnnotationLayer = null; const bar = document.getElementById('annotationSelectionToolbar'); if (bar) bar.classList.remove('has-selection'); }
function editSelectedAnnotation() {
    const layer = selectedAnnotationLayer; if (!layer) { addLog('warning','Click an annotation first'); return; }
    try { if (layer instanceof L.Marker && layer.dragging && typeof layer.dragging.enable === 'function') { layer.dragging.enable(); addLog('info','Point edit mode enabled'); return; } if (!layer.editing || typeof layer.editing.enable !== 'function') { addLog('warning','This annotation type does not expose an editable geometry handler'); return; } layer.editing.enable(); addLog('info','Edit mode enabled'); } catch (e) { addLog('error','Could not edit selected annotation: ' + e.message); }
}
function deleteSelectedAnnotation() {
    const layer = selectedAnnotationLayer; if (!layer || !drawnItems || !drawnItems.hasLayer(layer)) { addLog('warning','Click an annotation first'); return; }
    if (!confirm('Delete the selected annotation?')) return;
    restoreAnnotationStyle(layer); drawnItems.removeLayer(layer); drawnShapes = (drawnShapes || []).filter(x => x !== layer); selectedAnnotationLayer = null;
    const bar = document.getElementById('annotationSelectionToolbar'); if (bar) bar.classList.remove('has-selection'); updateObjectCount();
}
function clearAll() { if (!confirm('Clear all shapes from map?')) return; if (drawnItems) drawnItems.clearLayers(); clearSelectedAnnotation(); drawnShapes = []; isAnnotationLoaded = false; loadedAnnotationFeatures = []; updateObjectCount(); }
function deleteSelectedShape() { if (!drawnItems) return; const layers = drawnItems.getLayers(); if (layers.length === 0) return; drawnItems.removeLayer(layers[layers.length - 1]); updateObjectCount(); }
function setDrawMode(on) { [map, mapPredict, window.mapEval].forEach(m => { if (!m || typeof m.getContainer !== 'function') return; const c = m.getContainer(); if (c) c.classList.toggle('geoai-draw-mode', !!on); }); }
function setAOI(m) { const on = (m === 'draw'); setDrawMode(on); addLog('info', on ? 'Draw mode ON' : 'Draw mode OFF'); }

function captureMap() {
    if (!map) return; let captureStarted = false;
    const captureAtFixedZoom = () => {
        if (captureStarted || !map || map.getZoom() !== FIXED_CAPTURE_ZOOM) return; captureStarted = true;
        const b = map.getBounds();
        fetch('/api/map/capture', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bounds: {north: b.getNorth(), south: b.getSouth(), east: b.getEast(), west: b.getWest()}, zoom: FIXED_CAPTURE_ZOOM, service_id: document.getElementById('mapServiceSelect').value, project_id: currentProjectId }) })
        .then(r => r.json()).then(d => { if (d.success) addLog('success', `Captured at z=${d.zoom}: ${d.width}x${d.height}`); else addLog('error', 'Capture failed: ' + (d.error || 'unknown error')); }).catch(e => addLog('error', 'Capture failed: ' + e.message));
    };
    if (map.getZoom() !== FIXED_CAPTURE_ZOOM) { map.setZoom(FIXED_CAPTURE_ZOOM, {animate:false}); map.once('moveend', captureAtFixedZoom); setTimeout(() => { if (map && map.getZoom() === FIXED_CAPTURE_ZOOM) captureAtFixedZoom(); }, 150); } else captureAtFixedZoom();
}
function lockTrainingScale() { if (!map) return; const c = map.getCenter(), z = FIXED_CAPTURE_ZOOM, svc = document.getElementById('mapServiceSelect').value; fetch('/api/map/set_training_zoom', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({zoom: z, service_id: svc, lat: c.lat}) }).then(r => r.json()).then(d => { if (d.success) { trainingZoom = d.training_zoom; trainingGsd = d.training_gsd; trainingService = d.training_service; addLog('success', `🔒 Locked training scale: z=${d.training_zoom}`); } }); }

function loadAnnotationList(projectId) { const url = projectId ? `/api/annotations/files?project_id=${projectId}` : '/api/annotations/files'; fetch(url).then(r => r.json()).then(d => { if (d.success && d.files) { availableAnnotationFiles = d.files; const total = d.files.reduce((s, f) => s + (f.features_count || 0), 0); const e = document.getElementById('projectAnnCount'); if (e) e.textContent = total; } }).catch(() => {}); }
function loadAnnotations() { if (!currentProjectId) { addLog('warning','Select a project first'); return; } document.getElementById('annotationModal').classList.add('active'); loadAnnotationModalList(); }
function closeAnnotationModal() { document.getElementById('annotationModal').classList.remove('active'); }
function refreshAnnotationModalList() { loadAnnotationModalList(); }

function loadAnnotationModalList() {
    const c = document.getElementById('annotationModalList'); if (!c) return;
    c.innerHTML = `<div style="padding:40px;text-align:center;color:#64748b;"><i class="fas fa-spinner fa-spin" style="font-size:24px;margin-bottom:8px;display:block;"></i>Scanning annotation folders...</div>`;
    fetch('/api/annotations/files?include_all=1').then(r => { if (!r.ok) return r.text().then(text => { throw new Error(`HTTP ${r.status}: ${text.slice(0, 200)}`); }); return r.json(); })
    .then(d => {
        if (!d.success) throw new Error(d.error || 'Server returned success:false');
        const files = d.files || []; modalAnnotationFiles = files.map(f => { let lbl = f.label; if (!lbl || lbl === 'unknown' || lbl === '') lbl = 'unlabeled'; return Object.assign({}, f, {label: lbl}); });
        if (!modalAnnotationFiles.length) { c.innerHTML = `<div style="padding:40px;text-align:center;color:#64748b;">No GeoJSON files found.</div>`; return; }
        renderAnnotationModalList(modalAnnotationFiles);
    }).catch(e => { c.innerHTML = `<div style="padding:30px;text-align:center;color:#f87171;">Failed to load: ${e.message}</div>`; });
}

function renderAnnotationModalList(files) {
    const c = document.getElementById('annotationModalList'); if (!c) return;
    if (!files || !files.length) { c.innerHTML = '<div style="padding:40px;text-align:center;color:#64748b;">No files</div>'; return; }
    let h = '<table style="width:100%;border-collapse:collapse;font-size:11px;">';
    h += '<thead style="background:#0a0e17;position:sticky;top:0;z-index:2;"><tr><th style="padding:8px;text-align:left;color:#64748b;">#</th><th style="padding:8px;text-align:left;color:#64748b;">File</th><th style="padding:8px;text-align:left;color:#64748b;">Label</th><th style="padding:8px;text-align:center;color:#64748b;">Features</th><th style="padding:8px;text-align:center;color:#64748b;">Action</th></tr></thead><tbody>';
    files.forEach((f, i) => { const sel = selectedModalIndex === i, lbl = f.label || '—', lblColor = (lbl === 'unlabeled') ? '#f87171' : '#fbbf24'; h += `<tr onclick="selectAnnotationFromModal(${i})" style="cursor:pointer;border-bottom:1px solid #1e293b;background:${sel?'#1e293b':'transparent'};"><td style="padding:6px 10px;color:#64748b;">${i+1}</td><td style="padding:6px 10px;color:#e2e8f0;font-size:10px;word-break:break-all;">${f.filename}</td><td style="padding:6px 10px;"><span style="background:#1e293b;padding:2px 8px;border-radius:10px;color:${lblColor};font-size:9px;">${lbl}</span></td><td style="padding:6px 10px;text-align:center;color:#94a3b8;">${f.features_count || 0}</td><td style="padding:6px 10px;text-align:center;"><button onclick="event.stopPropagation();selectAnnotationFromModal(${i})" style="background:${sel?'#4ecdc4':'#1e293b'};border:none;color:${sel?'#0a0e17':'#94a3b8'};padding:2px 10px;border-radius:3px;cursor:pointer;font-size:9px;">${sel?'✓':'Select'}</button></td></tr>`; });
    h += '</tbody></table>'; c.innerHTML = h;
}

function selectAnnotationFromModal(i) {
    selectedModalIndex = i; const a = modalAnnotationFiles[i]; if (!a) return;
    const info = document.getElementById('selectedAnnotationInfo'); if (info) info.style.display = 'flex';
    document.getElementById('selectedAnnName').textContent = a.filename; document.getElementById('selectedAnnCount').textContent = a.features_count || 0; document.getElementById('selectedAnnLabel').textContent = a.label || '—';
    renderAnnotationModalList(modalAnnotationFiles);
}

function loadSelectedAnnotationFromModal() {
    if (selectedModalIndex < 0) { addLog('warning', 'Select a file first'); return; }
    const a = modalAnnotationFiles[selectedModalIndex]; if (!a) return;
    if (a._features && a._features.length) { displayAndZoomAnnotations(a._features, a); return; }
    fetch('/api/annotations/load_file', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({filename: a.filename, filepath: a.filepath, project_id: currentProjectId}) })
    .then(r => r.json()).then(d => { if (d.success && d.features) { a._features = d.features; if (d.label && d.label !== 'unlabeled') a.label = d.label; displayAndZoomAnnotations(d.features, a); } else addLog('error', 'Load failed: ' + (d.error || '')); }).catch(e => addLog('error', 'Load error: ' + e.message));
}

function displayAndZoomAnnotations(features, ann) {
    if (!features || !features.length || !drawnItems) return;
    drawnItems.clearLayers(); let bounds = null, count = 0;
    const fileLabel = (ann && ann.label && ann.label !== 'unlabeled') ? ann.label : currentLabel;
    const annotationModelType = getSelectedProjectAnnotationModelType(), isBoxProject = isBoxAnnotationModel(annotationModelType);
    features.forEach(sourceFeature => {
        try {
            if (!sourceFeature || !sourceFeature.geometry) return;
            const displayFeature = isBoxProject ? featureToBoundingBoxFeature(sourceFeature) : JSON.parse(JSON.stringify(sourceFeature));
            if (!displayFeature || !displayFeature.geometry) return;
            const geoJsonLayer = L.geoJSON(displayFeature, { style: { color: '#4ecdc4', weight: 2, opacity: 0.9, fillColor: '#4ecdc4', fillOpacity: 0.4 }, pointToLayer: function(feature, latlng) { return L.circleMarker(latlng, {radius: 5, color: '#4ecdc4'}); } });
            geoJsonLayer.eachLayer(layer => {
                layer.feature = layer.feature || JSON.parse(JSON.stringify(displayFeature)); layer.feature.properties = Object.assign({}, displayFeature.properties || {}, layer.feature.properties || {});
                layer.feature.properties.label = fileLabel; layer.feature.properties.class_name = fileLabel;
                if (isBoxProject) { layer.feature.properties.shape_type = 'rectangle'; layer.feature.properties.target_type = 'boxes_labels'; }
                bindAnnotationSelection(layer); drawnItems.addLayer(layer); count++;
                try { if (typeof layer.getBounds === 'function') { const b = layer.getBounds(); if (b && b.isValid()) bounds = bounds ? bounds.extend(b) : b; } } catch(e) {}
            });
        } catch(e) {}
    });
    isAnnotationLoaded = true; loadedAnnotationFeatures = features; updateObjectCount();
    if (bounds && bounds.isValid() && map) map.fitBounds(bounds, {padding:[50,50], maxZoom:18});
    const st = document.getElementById('annotationStatus'); if (st) { st.textContent = '✅ Loaded ' + count + ' ' + (isBoxAnnotationModel(annotationModelType) ? 'boxes' : 'polygons') + ' (' + fileLabel + ')'; st.style.color = '#34d399'; }
    if (count > 0 && typeof previewLabelClusterAOIs === 'function') setTimeout(() => previewLabelClusterAOIs(), 600);
    closeAnnotationModal(); currentLabel = fileLabel; document.getElementById('activeLabelDisplay').textContent = fileLabel; document.getElementById('objectLabel').value = fileLabel;
}

async function mergeAllLabels() {
    if (!currentProjectId) { addLog('warning','Select a project'); return; }
    const st = document.getElementById('annotationStatus'); if (st) { st.textContent = '⏳ Merging...'; st.style.color = '#fbbf24'; }
    try {
        const r = await fetch('/api/annotations/merge', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({project_id: currentProjectId, label: currentLabel || 'st'}) });
        const d = await r.json(); if (!d.success) { addLog('error','Merge failed: ' + (d.error||'')); return; }
        addLog('success', 'Merged ' + d.num_features + ' features'); if (st) { st.textContent = '✅ ' + d.num_features + ' features merged'; st.style.color = '#34d399'; }
        showMergeExportPanel(d);
    } catch(e) { addLog('error', 'Merge error: ' + e.message); }
}
function showMergeExportPanel(d) {
    let panel = document.getElementById('mergeExportPanel');
    if (!panel) { panel = document.createElement('div'); panel.id = 'mergeExportPanel'; panel.style.cssText = 'background:#0a0e17;border:1px solid #1e293b;border-left:3px solid #34d399;border-radius:4px;padding:8px;margin-top:6px;font-size:9px;'; const st = document.getElementById('annotationStatus'), sec = st ? st.closest('.config-section') : null; if (sec && sec.parentNode) sec.parentNode.insertBefore(panel, sec.nextSibling); }
    let h = `<div style="color:#34d399;font-weight:700;margin-bottom:4px;">✅ Merged Layer (${d.num_features} features)</div><div style="display:flex;gap:4px;flex-wrap:wrap;">`;
    if (d.download_geojson_url) h += `<button class="btn btn-success btn-xs" onclick="downloadFile('${d.download_geojson_url}')"><i class="fas fa-file-code"></i> GeoJSON</button>`;
    if (d.download_shp_url) h += `<button class="btn btn-info btn-xs" onclick="downloadFile('${d.download_shp_url}')"><i class="fas fa-file-shape"></i> Shapefile</button>`;
    h += `<button class="btn btn-secondary btn-xs" onclick="document.getElementById('mergeExportPanel').remove()">✕</button></div>`; panel.innerHTML = h;
}
function downloadFile(url) { if (!url) return; const a = document.createElement('a'); a.href = url; a.download = ''; document.body.appendChild(a); a.click(); document.body.removeChild(a); addLog('success', 'Downloaded: ' + url.split('/').pop()); }

async function buildDataset() {
    if (!currentProjectId) { addLog('warning', 'Select a project'); return; }
    if (!drawnItems || drawnItems.getLayers().length === 0) { addLog('warning', 'No shapes drawn.'); return; }
    const tile = parseInt(document.getElementById('tileSize')?.value || '64', 10), overlap = parseFloat(document.getElementById('tileOverlap')?.value || '0.2'), status = document.getElementById('datasetStatus'), btn = document.getElementById('btnBuildDataset');
    if (status) { status.textContent = '⏳ Preparing...'; status.style.color = '#fbbf24'; }
    if (btn) btn.disabled = true;
    const activeLabelName = getActiveAnnotationLabel();
    const layers = drawnItems.getLayers().slice(), features = [];
    layers.forEach(layer => {
        const existingProps = (layer.feature && layer.feature.properties) || {}, featureLabel = String(existingProps.class_name || existingProps.label || existingProps.class || activeLabelName).trim();
        if (featureLabel.toLowerCase() !== activeLabelName.toLowerCase()) return;
        const feature = layerToAnnotationFeature(layer, Object.assign({}, existingProps, { project_id: currentProjectId, label: activeLabelName, class_name: activeLabelName }), getSelectedProjectAnnotationModelType());
        if (feature) features.push(feature);
    });
    if (!features.length) {
        addLog('error', 'No valid annotation geometries available.');
        if (status) status.textContent = '❌ No valid annotations';
        if (btn) btn.disabled = false;
        return;
    }
    try {
        // Multi-AOI path
        if (userAois && userAois.length > 0) {
            if (status) { status.textContent = `⏳ Building across ${userAois.length} AOIs...`; status.style.color = '#fbbf24'; }
            addLog('info', `Building dataset across ${userAois.length} independent AOIs (tile=${tile}, overlap=${overlap})...`);
            const aoisPayload = userAois.map(a => ({
                id: a.id,
                name: a.name,
                bounds: a.bounds,
                geometry: a.geometry,
                annotation_count: a.annotation_count
            }));
            const buildResponse = await fetch('/api/dataset/build', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    project_id: currentProjectId,
                    model_type: getCurrentModelType(),
                    model_type_key: normalizeAnnotationModelType(getCurrentModelType()),
                    tile_size: tile,
                    overlap: overlap,
                    label: activeLabelName,
                    features: features,
                    aois: aoisPayload,
                    service_id: document.getElementById('mapServiceSelect')?.value,
                    capture_zoom: Math.max(17, Number(trainingZoom) || 0)
                })
            });
            const result = await buildResponse.json();
            if (!buildResponse.ok || !result.success) throw new Error(result.error || 'Dataset build failed');
            addLog('success', `🎉 Dataset built: ${result.total_chips} chips from ${result.num_aois || userAois.length} AOIs`);
            if (status) { status.textContent = `✅ ${result.total_chips} chips (${result.num_aois || userAois.length} AOIs)`; status.style.color = '#34d399'; }
            const chipCount = document.getElementById('projectChipCount'); if (chipCount) chipCount.textContent = result.total_chips || 0;
            if (result.chip_previews) displayChipPreviews(result.chip_previews);
            updateStatus(); loadProjects();
            return;
        }

        // Single AOI fallback
        let activeAOI = window.activeLabelAOI;
        if (!activeAOI || String(activeAOI.active_label || '').toLowerCase() !== activeLabelName.toLowerCase()) activeAOI = await previewLabelClusterAOIs();
        if (!activeAOI || !activeAOI.bounds) { addLog('error', 'Cannot build chips: draw an AOI or click Cluster AOI first.'); if (status) status.textContent = '❌ AOI boundary unavailable'; return; }
        const aoiBounds = activeAOI.bounds;
        const captureResponse = await fetch('/api/map/capture', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ bounds: { north: Number(aoiBounds.north), south: Number(aoiBounds.south), east: Number(aoiBounds.east), west: Number(aoiBounds.west) }, zoom: Math.max(17, Number(trainingZoom) || 0), service_id: document.getElementById('mapServiceSelect')?.value, project_id: currentProjectId }) });
        if (!captureResponse.ok) { const err = await captureResponse.json().catch(() => ({})); throw new Error(err.error || 'Image capture failed'); }
        const capture = await captureResponse.json(); if (!capture.success || !capture.filepath || !capture.bounds) throw new Error(capture.error || 'Capture response missing data');
        const bounds = Object.assign({crs: 'EPSG:4326'}, capture.bounds);
        const buildResponse = await fetch('/api/dataset/build', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ project_id: currentProjectId, model_type: getCurrentModelType(), model_type_key: normalizeAnnotationModelType(getCurrentModelType()), tile_size: tile, overlap: overlap, label: activeLabelName, features: features, aoi_geometry: activeAOI.geometry, aoi_bounds: aoiBounds, bounds: bounds, image_path: capture.filepath, image_width: capture.width, image_height: capture.height, crs: 'EPSG:4326', is_aoi: true }) });
        const result = await buildResponse.json(); if (!buildResponse.ok || !result.success) throw new Error(result.error || 'Dataset build failed');
        addLog('success', 'Dataset built: ' + result.total_chips + ' chips'); if (status) { status.textContent = '✅ ' + result.total_chips + ' chips'; status.style.color = '#34d399'; }
        const chipCount = document.getElementById('projectChipCount'); if (chipCount) chipCount.textContent = result.total_chips || 0;
        if (result.chip_previews) displayChipPreviews(result.chip_previews);
        updateStatus(); loadProjects();
    } catch (error) { addLog('error', 'Dataset build error: ' + error.message); if (status) { status.textContent = '❌ ' + error.message; status.style.color = '#f87171'; } }
    finally { if (btn) btn.disabled = false; }
}

function displayChipPreviews(chips) {
    let c = document.getElementById('chipPreviewContainer'); if (!c) return;
    c.innerHTML = ''; if (!chips || !chips.length) return;
    const best = chips.filter(x => x.has_foreground).slice(0, 2), show = best.length ? best : chips.slice(0, 2), grid = document.createElement('div');
    grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px;margin-top:6px;';
    show.forEach(ch => { const card = document.createElement('div'); card.style.cssText = 'background:#0a0e17;border:1px solid #1e293b;border-radius:4px;padding:4px;'; card.innerHTML = `<div style="display:flex;gap:4px;"><img src="data:image/png;base64,${ch.image}" style="width:50%;height:80px;object-fit:cover;border-radius:3px;"><img src="data:image/png;base64,${ch.mask}" style="width:50%;height:80px;object-fit:cover;border-radius:3px;"></div>`; grid.appendChild(card); });
    c.appendChild(grid);
}

function startTrainingTimer() { trainingStartTime = Date.now(); if (trainingTimerInterval) clearInterval(trainingTimerInterval); trainingTimerInterval = setInterval(function() { if (!trainingStartTime) return; const el = Date.now() - trainingStartTime, h = String(Math.floor(el/3600000)).padStart(2,'0'), m = String(Math.floor((el%3600000)/60000)).padStart(2,'0'), s = String(Math.floor((el%60000)/1000)).padStart(2,'0'); const e = document.getElementById('runTime'); if (e) e.textContent = `⏱️ ${h}:${m}:${s}`; }, 1000); }
function stopTrainingTimer() { if (trainingTimerInterval) { clearInterval(trainingTimerInterval); trainingTimerInterval = null; } trainingStartTime = null; }

function startTraining() {
    if (!currentProjectId) { addLog('warning','Select a project'); return; }
    if (isTraining) { addLog('warning','Already running'); return; }
    const chipCount = parseInt(document.getElementById('projectChipCount')?.textContent || '0', 10);
    if (chipCount === 0) { addLog('warning','Build dataset first'); return; }
    if (!currentLabel || currentLabel === 'unlabeled') { addLog('warning', 'Pick a real label'); return; }
    const epochs = parseInt(document.getElementById('epochsInput')?.value || '100', 10), batch = parseInt(document.getElementById('batchSize')?.value || '2', 10), lr = parseFloat(document.getElementById('lrInput')?.value || '0.0001'), thr = parseFloat(document.getElementById('thresholdInput')?.value || '0.5'), mode = document.getElementById('modeSelect')?.value || 'binary', modelName = document.getElementById('modelNameInput')?.value?.trim() || '';
    const currentModelType = getCurrentModelType(), isMaskRCNN = isMaskRCNNModel(currentModelType), selectedBackbone = document.getElementById('backboneSelect')?.value || 'unet', backbone = isMaskRCNN ? 'resnet50' : selectedBackbone;
    if (!modelName) { addLog('warning','Enter a model name'); return; }
    lastTrainingModelName = modelName.replace(/\.pth$/i, '');
    trainingStats = { epochs, currentEpoch: 0, trainLoss: [], valLoss: [], iou: [], bestIou: 0, isComplete: false, isModelLoaded: false };
    const progressPanel = document.getElementById('runProgress'); if (progressPanel) progressPanel.style.display = 'block';
    updateLossChart(); document.getElementById('trainBtn').disabled = true; document.getElementById('trainBtn').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...'; startTrainingTimer();
    const fineTuneRow = document.getElementById('fineTuneExistingModelRow'), fineTuneModeActive = !!fineTuneRow && fineTuneRow.style.display !== 'none', fineTuneSelectPath = document.getElementById('fineTuneExistingModelSelect')?.value || '', fineTuneHiddenPath = document.getElementById('fineTuneModelPath')?.value || '', fineTunePath = fineTuneModeActive ? (fineTuneSelectPath || fineTuneHiddenPath).trim() : '';
    if (fineTuneModeActive && !fineTunePath) { alert('Select an existing trained model.'); document.getElementById('trainBtn').disabled = false; return; }
    const payload = { project_id: currentProjectId, epochs, batch_size: batch, learning_rate: lr, threshold: thr, label: currentLabel, backbone: backbone, mode: mode, model_name: modelName, model_type: currentModelType, fine_tune_model_path: fineTunePath, training_zoom: trainingZoom, training_service: trainingService, ...getAdvancedTrainingSettings() };
    isTraining = true; trainingStats.isComplete = false;
    fetch('/train', { method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) })
    .then(r => r.json()).then(d => {
        document.getElementById('trainBtn').disabled = false; document.getElementById('trainBtn').innerHTML = '<i class="fas fa-play"></i> TRAIN';
        if (d.success) { isTraining = true; document.getElementById('trainBtn').style.display = 'none'; document.getElementById('stopBtn').style.display = 'flex'; document.getElementById('runStatus').textContent = '⏳ Training...'; updateStatusBadge('training'); }
        else { addLog('error', 'Failed: ' + (d.error||'')); stopTrainingTimer(); }
    }).catch(e => { document.getElementById('trainBtn').disabled = false; addLog('error', 'Error: ' + e.message); stopTrainingTimer(); });
}

function stopTraining() { if (!confirm('Stop training?')) return; fetch('/api/train/stop', {method:'POST'}).then(r => r.json()).then(d => { if (d.success) addLog('info', 'Stop sent'); }); }
function runOverfitTest() { if (!currentProjectId) { addLog('warning','Select a project'); return; } const lbl = currentLabel; if (!lbl || lbl === 'unlabeled') return; fetch('/api/train/overfit_test', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ project_id: currentProjectId, label: lbl }) }).then(r => r.json()).then(d => { if (!d.success) return; addLog(d.history[d.history.length-1].iou > 0.7 ? 'success' : 'error', d.verdict); }).catch(e => addLog('error', 'Overfit error: ' + e.message)); }

function normalizeModelPath(value) { return value ? String(value).trim().replace(/\\/g, '/') : ''; }

function loadModels() {
    return fetch('/api/models/all', {cache: 'no-store'}).then(r => {
        if (!r.ok) throw new Error('Models list HTTP ' + r.status);
        return r.json();
    }).then(d => {
        const raw = Array.isArray(d.models) ? d.models : [];
        const byKey = new Map();
        raw.forEach(m => {
            const filepath = normalizeModelPath(m.filepath || m.path);
            if (!filepath) return;
            const key = m.model_id || filepath.toLowerCase(); 
            const row = { ...m, filepath, path: filepath, model_path: filepath, model_name: m.model_name || m.filename || 'model', key: key };
            const previous = byKey.get(key);
            if (!previous || (row.is_dlpk && !previous.is_dlpk) || (row.extra_info && !previous.extra_info)) byKey.set(key, row);
        });
        availableModels = Array.from(byKey.values()).sort((a, b) => {
            const active = (normalizeModelPath(b.filepath) === normalizeModelPath(loadedModelPath)) - (normalizeModelPath(a.filepath) === normalizeModelPath(loadedModelPath));
            return active || String(a.model_name).localeCompare(String(b.model_name));
        });

        const sel = document.getElementById('modelSelect');
        if (sel) {
            const currentValue = normalizeModelPath(loadedModelPath || sel.value || '');
            const newSignature = availableModels.map(m => m.key).join('|');
            const currentSignature = Array.from(sel.options).map(o => o.dataset.modelKey).filter(Boolean).join('|');

            if (newSignature !== currentSignature) {
                sel.innerHTML = '<option value="">-- Select a model to load --</option>';
                availableModels.forEach(m => {
                    const tag = m.is_dlpk ? '.dlpk' : m.source || 'project';
                    const iou = Number(m.best_iou || 0).toFixed(3);
                    const label = `${m.model_name} (IoU ${iou}) | ${m.model_type || '?'} | ${tag}`;
                    const opt = new Option(label, m.filepath);
                    opt.dataset.modelKey = m.key;
                    opt.dataset.modelIndex = String(availableModels.indexOf(m));
                    sel.appendChild(opt);
                });
            }
            if (currentValue) {
                const matchingOpt = Array.from(sel.options).find(o => normalizeModelPath(o.value) === currentValue);
                if (matchingOpt) sel.value = matchingOpt.value;
            }
        }

        const list = document.getElementById('yourModelsList');
        if (list) {
            list.innerHTML = '';
            if (availableModels.length === 0) {
                list.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:40px;color:#64748b;">No trained models yet</div>';
            } else {
                availableModels.forEach((m, idx) => {
                    const p = normalizeModelPath(m.filepath), isLoaded = loadedModelPath === p, tz = m.tile_size ? ` | tile ${m.tile_size}px` : '', sourceLabel = m.is_dlpk ? 'Imported .dlpk' : (m.source || 'project'), extra = m.extra_info || {}, description = extra.description || '';
                    const iou = Number(m.best_iou || 0);
                    const prec = Number(extra.precision || extra.average_precision || m.precision || 0);
                    const rec = Number(extra.recall || m.recall || 0);
                    const f1 = Number(extra.f1 || m.f1 || 0);
                    let metricsHtml = '';
                    if (iou > 0) metricsHtml += `<span><span class="label">📈 IoU:</span> ${(iou * 100).toFixed(2)}%</span>`;
                    else if (prec > 0) metricsHtml += `<span><span class="label">🎯 Prec:</span> ${(prec * 100).toFixed(2)}%</span>`; // Fallback to precision if IoU is missing
                    if (rec > 0) metricsHtml += `<span><span class="label">🔍 Rec:</span> ${(rec * 100).toFixed(2)}%</span>`;
                    if (f1 > 0) metricsHtml += `<span><span class="label">⚖️ F1:</span> ${(f1 * 100).toFixed(2)}%</span>`;
                    if (!metricsHtml) metricsHtml = `<span><span class="label">📈 IoU:</span> 0.00%</span>`;

                    let borderColor = '#1e293b';
                    let badgeText = 'NATIVE';
                    let badgeColor = '#38bdf8';

                    if (m.is_dlpk) { borderColor = '#a78bfa'; badgeText = 'IMPORTED .DLPK'; badgeColor = '#a78bfa'; }
                    else if (m.is_finetuned) { borderColor = '#fbbf24'; badgeText = 'FINE-TUNED'; badgeColor = '#fbbf24'; }
                    else if (isLoaded) { borderColor = '#4ecdc4'; }

                    list.innerHTML += `
                        <div class="model-card" style="border-color:${borderColor}; border-width: 2px;">
                            <div class="model-header">
                                <h4><i class="fas fa-robot"></i> ${m.model_name}</h4>
                                <div style="display:flex; gap:6px; align-items:center;">
                                    <span class="status-badge" style="background:${badgeColor}22; color:${badgeColor}; border:1px solid ${badgeColor}55; font-size:9px;">${badgeText}</span>
                                    <span class="status-badge ${isLoaded ? 'completed' : 'idle'}">${isLoaded ? 'ACTIVE' : 'READY'}</span>
                                </div>
                            </div>
                            <div class="model-details">
                                ${metricsHtml}
                                <span><span class="label">🧠 Arch:</span> ${m.architecture || m.model_type || '?'}</span>
                                <span><span class="label">📦 Source:</span> ${sourceLabel}</span>
                                <span><span class="label">🏷️ Label:</span> ${m.label || '—'}</span>
                                <span><span class="label">📐 Size:</span> ${m.tile_size || '?'}px${tz}</span>
                                <span><span class="label">🔢 Mode:</span> ${m.mode || 'binary'}</span>
                            </div>
                            ${description ? `<div style="font-size:10px;color:#94a3b8;margin:6px 0;line-height:1.4;">${description}</div>` : ''}
                            
                            <div class="model-actions" style="display:flex; gap:4px; align-items:center; margin-top: auto; border-top: 1px solid #1e293b; padding-top: 8px;">
                                ${isLoaded ? `<button class="btn btn-success" disabled><i class="fas fa-check-circle"></i> Loaded</button>` : `<button class="btn btn-info" onclick="loadModelByIndex(${idx})"><i class="fas fa-upload"></i> Load</button>`}
                                
                                ${geoaiCan('can_publish_model') ? `
                                    <select id="exportFormat_${idx}" style="background:#0a0e17;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:0 4px;font-size:10px; height:26px; flex:1; min-width:80px;">
                                        <option value=".pth">.pth (PyTorch)</option>
                                        <option value=".dlpk">.dlpk (ArcGIS)</option>
                                        <option value=".onnx">.onnx</option>
                                        <option value=".pt">.pt (TorchScript)</option>
                                        <option value=".ckpt">.ckpt</option>
                                        <option value=".h5">.h5 (Keras)</option>
                                        <option value=".pb">.pb (TF SavedModel)</option>
                                        <option value=".tflite">.tflite</option>
                                    </select>
                                    <button class="btn btn-success" onclick="exportModelWithFormat(${idx})" style="padding: 0 8px;"><i class="fas fa-download"></i> Export</button>
                                ` : ''}
                                
                                <button class="btn btn-danger" onclick="deleteModel('${p.replace(/'/g, "\\'")}')" style="padding: 0 8px;"><i class="fas fa-trash"></i></button>
                            </div>
                        </div>`;
                });
            }
        }
        const n = availableModels.length;
        ['modelsTotal', 'modelsTrained', 'modelBadge2', 'dashModels', 'dashModels2'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = n; });
    }).catch(e => { console.error('loadModels error:', e); addLog('error', 'Could not list models: ' + e.message); });
}

async function exportModelWithFormat(index) {
    const m = availableModels[index];
    if (!m) return;

    const formatSelect = document.getElementById(`exportFormat_${index}`);
    const format = formatSelect ? formatSelect.value : '.pth';
    const path = m.filepath;
    const name = m.model_name || 'exported_model';

    addLog('info', `Exporting model as ${format}...`);

    const exportBtn = formatSelect && formatSelect.nextElementSibling;
    let originalBtnHTML = '';
    
    if (exportBtn) {
        originalBtnHTML = exportBtn.innerHTML;
        exportBtn.disabled = true;
        exportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';
    }

    try {
        const response = await fetch('/api/models/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: path,
                model_name: name,
                format: format,
                save_location: './models/saved',
                include_torchscript: true
            })
        });

        const d = await response.json();
        if (!response.ok || d.success === false) {
            throw new Error(d.error || 'Export failed');
        }

        addLog('success', `Exported successfully to ${d.model_path}`);

        if (format !== '.dlpk') {
            downloadModel(d.model_path);
        } else {
            const verdict = dlpkVerdictText(d.arcgis_runnable, d.arcgis_runnable_reason);
            if (confirm(`Exported .dlpk (${verdict.head}). Download now?`)) {
                downloadModel(d.dlpk_path);
            }
            loadModels();
        }
    } catch (e) {
        alert('Export failed: ' + e.message);
    } finally {
        if (exportBtn) {
            exportBtn.disabled = false;
            exportBtn.innerHTML = originalBtnHTML;
        }
    }
}

function loadModelByIndex(index) { const m = availableModels[Number(index)]; if (!m) return; loadModel(m.filepath); }
function loadModelFromModalDropdown() {
    const sel = document.getElementById('modelSelect');
    const btn = document.getElementById('modalLoadModelBtn');
    if (!sel) return;

    const chosenPath = (sel.value || '').trim();
    if (!chosenPath) {
        addLog('warning', 'Pick a model in the dropdown first.');
        return;
    }

    if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading…'; }

    const modal = document.getElementById('modelSelectModal');
    const reenable = function() {
        if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-upload"></i> Load Selected Model'; }
    };
    if (modal && typeof MutationObserver !== 'undefined') {
        const observer = new MutationObserver(function(muts, obs) {
            if (!modal.classList.contains('active')) {
                obs.disconnect();
                reenable();
            }
        });
        observer.observe(modal, { attributes: true, attributeFilter: ['class'] });
        setTimeout(function() { observer.disconnect(); reenable(); }, 15000);
    } else {
        setTimeout(reenable, 15000);
    }

    loadModel(chosenPath);
}

function refreshEvaluateModelStatus() {
    const statusEl = document.getElementById('evalStatus'), modelEl  = document.getElementById('evalModelPath');
    if (!loadedModelPath) { if (statusEl) { statusEl.textContent = '⚠️ No model loaded — click "Load Model"'; statusEl.style.color = '#fbbf24'; } if (modelEl) modelEl.textContent = '—'; return; }
    const shortName = loadedModelPath.split('/').pop(), meta = window.loadedModelMeta || {}, iou = meta.best_iou !== undefined ? Number(meta.best_iou).toFixed(4) : '—', bkb = meta.backbone || currentBackbone || '—', mdl = meta.model_type || 'unet';
    if (statusEl) { statusEl.textContent = `✅ Model loaded — ${shortName} | IoU=${iou} | ${mdl}/${bkb}`; statusEl.style.color = '#34d399'; }
    if (modelEl) modelEl.textContent = shortName;
}

function loadModel(path) {
    const cleanPath = normalizeModelPath(path || document.getElementById('modelSelect')?.value || '');
    if (!cleanPath) { addLog('warning', 'No model selected'); alert('Please select a model from the dropdown first.'); return; }
    addLog('info', 'Loading model: ' + cleanPath.split('/').pop());
    const modalBtn = document.querySelector('#modelSelectModal .btn-create'), originalBtnText = modalBtn ? modalBtn.innerHTML : '';
    if (modalBtn) { modalBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...'; modalBtn.disabled = true; }
    const es = document.getElementById('evalStatus'); if (es) { es.textContent = '⏳ Loading model...'; es.style.color = '#fbbf24'; }
    const ps = document.getElementById('predictStatus'); if (ps) { ps.textContent = '⏳ Loading model...'; ps.style.color = '#fbbf24'; }
    fetch('/api/models/load', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ model_path: cleanPath, label: currentLabel || 'default' }) })
    .then(r => r.json()).then(d => {
        if (!d.success) { const msg = d.error || 'Load failed'; if (es) { es.textContent = '❌ ' + msg; es.style.color = '#f87171'; } if (ps) { ps.textContent = '❌ ' + msg; ps.style.color = '#f87171'; } if (modalBtn) { modalBtn.innerHTML = originalBtnText; modalBtn.disabled = false; } return; }
        loadedModelPath = normalizeModelPath(d.model_path || cleanPath); trainingStats.isModelLoaded = true; trainingStats.bestIou = Number(d.best_iou || 0);
        if (d.class_names) { classNames = d.class_names; allLabels = [...classNames]; } if (d.backbone) currentBackbone = d.backbone; if (d.mode) currentMode = d.mode;

        if (d.tile_size) { trainingZoom = undefined; const tileInput = document.getElementById('tileSize'); if (tileInput) tileInput.value = d.tile_size; }

        window.loadedModelMeta = d;
        const loadedClasses = (d.class_names || []).filter(n => String(n).toLowerCase() !== 'background');
        const modelSummary = `${d.model_name || cleanPath.split('/').pop()} | ${d.model_type || 'model'}${loadedClasses.length ? ' | classes: ' + loadedClasses.join(', ') : ''}`;
        if (es) { es.textContent = '✅ Loaded: ' + modelSummary; es.style.color = '#34d399'; } if (ps) { ps.textContent = '✅ Loaded: ' + modelSummary; ps.style.color = '#34d399'; }
        const ep = document.getElementById('evalModelPath'); if (ep) ep.textContent = loadedModelPath.split('/').pop();
        const btn = document.getElementById('predictShpBtn'); if (btn) btn.style.display = 'inline-flex';
        updateStatusBadge('completed');
        const extraInfoPanel = document.getElementById('modelExtraInfo');
        if (extraInfoPanel) {
            const extra = d.extra_info || {}, keys = Object.keys(extra).filter(k => extra[k] !== null && extra[k] !== undefined && extra[k] !== '');
            if (keys.length > 0) {
                const fmt = v => typeof v === 'object' ? JSON.stringify(v) : String(v);
                let html = '<div class="section-title">ℹ️ MODEL CARD (from .dlpk / .emd)</div><div style="font-size:11px;color:#94a3b8;line-height:1.8;">';
                for (const k of keys) { const label = k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); html += `<div style="margin-bottom:4px;"><strong style="color:#cbd5e1;">${label}:</strong> ${fmt(extra[k])}</div>`; }
                html += '</div>'; extraInfoPanel.innerHTML = html; extraInfoPanel.style.display = 'block';
            } else extraInfoPanel.style.display = 'none';
        }
        const modal = document.getElementById('modelSelectModal'); if (modal) modal.classList.remove('active'); if (modalBtn) { modalBtn.innerHTML = originalBtnText; modalBtn.disabled = false; }
        loadModels(); refreshEvaluateModelStatus();
    }).catch(e => { addLog('error', 'Load error: ' + e.message); if (es) { es.textContent = '❌ ' + e.message; es.style.color = '#f87171'; } if (ps) { ps.textContent = '❌ ' + e.message; ps.style.color = '#f87171'; } if (modalBtn) { modalBtn.innerHTML = originalBtnText; modalBtn.disabled = false; } });
}

function dlpkVerdictText(verdict, reason) { if (verdict === 'unverified') return { cls: 'warning', head: 'Packaged — not yet proven on ArcGIS', body: (reason || '') }; if (verdict === 'structurally_valid') return { cls: 'warning', head: 'Packaged — readable by ArcGIS, but NOT runnable', body: (reason || '') }; return { cls: 'success', head: 'Packaged for ArcGIS', body: reason || '' }; }

function exportDlpk(path) {
    addLog('info', 'Packaging .dlpk…');
    fetch('/api/models/' + encodeURIComponent(path).replace(/%2F/g, '/') + '/export_dlpk', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) }).then(async r => {
        const j = await r.json().catch(() => ({}));
        if (r.status === 403) { alert('Not permitted.'); return; }
        if (!r.ok || j.success === false) { alert('Export failed:\n\n' + (j.error || j.detail || ('HTTP ' + r.status))); return; }
        const v = dlpkVerdictText(j.arcgis_runnable, j.arcgis_runnable_reason), name = (j.dlpk_path || '').split(/[\\/]/).pop();
        if (confirm('Exported ' + name + ' (' + v.head + '). Download now?')) downloadModel(j.dlpk_path);
        loadModels();
    }).catch(e => alert('Export failed: ' + e.message));
}

function importDlpk(input) {
    const f = input && input.files && input.files[0];
    if (!f) return;
    input.value = '';

    const importBtn = document.querySelector('button[onclick*="dlpkImportInput"]');
    let originalBtnHTML = '';

    if (importBtn) {
        originalBtnHTML = importBtn.innerHTML;
        importBtn.disabled = true;
        importBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 0%';
    }

    const fd = new FormData();
    fd.append('file', f);

    // Use XMLHttpRequest to track upload progress percentage
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/models/import_dlpk', true);

    xhr.upload.onprogress = function(e) {
        if (e.lengthComputable && importBtn) {
            const percent = Math.round((e.loaded / e.total) * 100);
            importBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${percent}%`;
        }
    };

    xhr.onload = function() {
        if (importBtn) {
            importBtn.disabled = false;
            importBtn.innerHTML = originalBtnHTML;
        }
        
        try {
            const j = JSON.parse(xhr.responseText);
            if (xhr.status < 200 || xhr.status >= 300 || j.success === false) {
                alert('Import failed:\n\n' + (j.error || j.detail || ('HTTP ' + xhr.status)));
                return;
            }
            
            addLog('success', 'Imported .dlpk -> ' + (j.model_path || j.extract_dir || ''));
            const p = loadModels();
            if (p && typeof p.then === 'function') {
                p.then(() => {
                    if (!j.model_path) return;
                    const sel = document.getElementById('modelSelect');
                    if (!sel) return;
                    const norm = String(j.model_path).replace(/\\/g, '/');
                    for (const opt of sel.options) {
                        if (String(opt.value).replace(/\\/g, '/') === norm) {
                            sel.value = opt.value;
                            break;
                        }
                    }
                }).catch(e => console.error('loadModels failed after import:', e));
            }
        } catch (err) {
            alert('Import failed: Invalid server response.');
        }
    };

    xhr.onerror = function() {
        if (importBtn) {
            importBtn.disabled = false;
            importBtn.innerHTML = originalBtnHTML;
        }
        alert('Import failed: Network error.');
    };

    xhr.send(fd);
}

function downloadModel(path) { if (!path) return; const name = String(path).split(/[\\/]/).pop(); fetch('/api/models/download?path=' + encodeURIComponent(path)).then(async r => { if (!r.ok) { alert('Download failed'); return null; } return r.blob(); }).then(blob => { if (!blob) return; const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000); }); }
function deleteModel(path) { if (!confirm('Delete model?')) return; fetch('/api/models/' + encodeURIComponent(path.replace(/\\/g, '/')), {method:'DELETE'}).then(r => r.json()).then(d => { if (d.success) { if (loadedModelPath === path.replace(/\\/g, '/')) { loadedModelPath = null; trainingStats.isModelLoaded = false; } loadModels(); } }); }
function showLoadModelDialog() { const m = document.getElementById('modelSelectModal'); if (m) m.classList.add('active'); loadModelsForModal(); setTimeout(refreshEvaluateModelStatus, 300); }

function loadModelsForModal() {
    const sel = document.getElementById('modelSelect');
    if (!sel) return;

    const previousValue = normalizeModelPath(sel.value);
    const currentSignature = Array.from(sel.options)
        .map(o => o.dataset.modelKey)
        .filter(Boolean)
        .join('|');

    fetch('/api/models/all', {cache: 'no-store'})
        .then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(d => {
            const rawRows = (d && (d.models || d.rows)) || [];
            const seenKeys = new Set();
            const rows = rawRows.filter(m => {
                const filepath = normalizeModelPath(m.filepath || m.path);
                const key = m.model_id || filepath.toLowerCase();
                if (!key || seenKeys.has(key)) return false;
                seenKeys.add(key); 
                m.key = key;
                return true;
            });

            const newSignature = rows.map(m => m.key).sort().join('|');

            if (newSignature === currentSignature && sel.options.length > 1) {
                if (previousValue) {
                    const match = Array.from(sel.options).find(o => normalizeModelPath(o.value) === previousValue);
                    if (match) sel.value = match.value;
                }
                return;
            }

            sel.innerHTML = '<option value="">-- Select a model to load --</option>';
            if (rows.length === 0) {
                sel.innerHTML = '<option value="">-- No trained models found --</option>';
                return;
            }
            rows.forEach(m => {
                const path = normalizeModelPath(m.filepath || m.path);
                if (!path) return;
                const classNames = Array.isArray(m.class_names) ? m.class_names.filter(n => String(n).toLowerCase() !== 'background').join(', ') : '';
                const iouText = Number(m.best_iou || 0) > 0 ? ` | IoU ${(Number(m.best_iou) * 100).toFixed(1)}%` : '';
                const sourceText = m.is_dlpk || m.source === 'dlpk' ? 'DLPK' : (m.source || 'native');
                const optionText = `${m.model_name} | ${m.model_type || 'unknown'} | ${sourceText}${classNames ? ' | ' + classNames : ''}${iouText}`;
                
                const opt = new Option(optionText, path);
                opt.dataset.modelKey = m.key;
                sel.appendChild(opt);
            });

            if (previousValue) {
                const match = Array.from(sel.options).find(o => normalizeModelPath(o.value) === previousValue);
                if (match) sel.value = match.value;
            }
        })
        .catch(e => {
            sel.innerHTML = '<option value="">-- Error loading models --</option>';
            console.error('Modal models error:', e);
        });
}

function showSaveModelSection(path, metrics) { trainedModelPath = path; trainedModelMetrics = metrics; const s = document.getElementById('saveModelSection'); if (!s) return; s.style.display = 'block'; const inp = document.getElementById('modelNameInputSave'); if (inp) { inp.value = (lastTrainingModelName || currentLabel || 'model').replace(/\.pth$/i, ''); inp.focus(); inp.select(); } selectedSaveFolder = './models/saved'; document.getElementById('selectedFolderPath').textContent = selectedSaveFolder; document.getElementById('selectedFolderDisplay').style.display = 'block'; document.getElementById('saveModelStatus').textContent = '✅ Training complete — choose a name and location, then Save'; }
function openFolderBrowser() { const current = selectedSaveFolder || './models/saved', entered = window.prompt('Enter the folder where the trained model should be saved.\n\nExamples:\n  ./models/saved\n  D:\\GeoAI\\MyModels\n\nFolder path:', current); if (entered === null) return; const cleaned = entered.trim(); if (!cleaned) return; selectedSaveFolder = cleaned; document.getElementById('selectedFolderPath').textContent = selectedSaveFolder; document.getElementById('selectedFolderDisplay').style.display = 'block'; }
function saveTrainedModel() { const nmRaw = document.getElementById('modelNameInputSave')?.value?.trim(); if (!nmRaw) return; if (!trainedModelPath) return; const nm = nmRaw.replace(/\.pth$/i, ''), location = (selectedSaveFolder || './models/saved').trim(), status = document.getElementById('saveModelStatus'); if (status) status.textContent = '⏳ Saving model...'; fetch('/api/models/save', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ model_path: trainedModelPath, model_name: nm, label: currentLabel, metrics: trainedModelMetrics, save_location: location }) }).then(async r => { const d = await r.json(); if (!r.ok || !d.success) throw new Error(d.error || `Save failed (${r.status})`); return d; }).then(d => { if (status) status.textContent = `✅ Model saved: ${d.saved_path}`; addLog('success', `💾 Saved '${d.filename}' → ${d.saved_path}`); loadModels(); }).catch(e => { if (status) status.textContent = `❌ ${e.message}`; }); }

function evaluateModel() {
    if (!loadedModelPath) { addLog('warning', 'Load a model first'); return; }
    if (!currentProjectId) { addLog('warning', 'Select a project'); return; }
    const es = document.getElementById('evalStatus'); if (es) { es.textContent = '⏳ Evaluating TEST chips...'; es.style.color = '#fbbf24'; }
    if (!evalMap || !evalMapInitialized) initEvalMap();
    const aoiPayload = (evalAoiBounds && evalAoiBounds.isValid()) ? { north: evalAoiBounds.getNorth(), south: evalAoiBounds.getSouth(), east: evalAoiBounds.getEast(), west: evalAoiBounds.getWest() } : null;
    const thr = parseFloat(document.getElementById('thresholdInput')?.value || '0.5'), evalSplit = (document.getElementById('evalSplitSelect')?.value || 'test').toLowerCase();
    fetch('/api/evaluate_geojson', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ project_id: currentProjectId, model_path: loadedModelPath, label: currentLabel, threshold: thr, min_area: 8, eval_split: evalSplit, aoi_bounds: aoiPayload }) })
    .then(r => r.json()).then(d => {
        if (!d.success) { addLog('error', 'Eval failed: ' + (d.error || '')); if (es) { es.textContent = '❌ ' + (d.error || 'failed'); es.style.color = '#f87171'; } return; }
        const m = d.metrics || {}, split = (d.split || m.evaluation_split || evalSplit).toUpperCase();
        const setv = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = Number(v || 0).toFixed(3); };
        setv('evalMap', m.mean_iou); setv('evalF1', m.mean_f1); setv('evalRecall', m.mean_recall); setv('evalPrecision', m.mean_precision);
        const sampleCount = m.num_samples || d.processed_chip_count || 0; document.getElementById('evalSamples').textContent = sampleCount; document.getElementById('evalSamples2').textContent = sampleCount;
        if (es) { es.textContent = `✅ ${split} evaluation complete`; es.style.color = '#34d399'; }
        updateStatusBadge('evaluated'); addLog('success', `📊 ${split} eval: IoU=${Number(m.mean_iou || 0).toFixed(4)} P=${Number(m.mean_precision || 0).toFixed(4)} R=${Number(m.mean_recall || 0).toFixed(4)} F1=${Number(m.mean_f1 || 0).toFixed(4)}`);
        drawEvaluationPolygons(d);
    }).catch(e => { addLog('error', 'Eval error: ' + e.message); if (es) { es.textContent = '❌ ' + e.message; es.style.color = '#f87171'; } });
}

function drawEvaluationPolygons(d) {
    if (!evalMap) { switchWorkflow('evaluate'); setTimeout(() => drawEvaluationPolygons(d), 900); return; }
    if (window.__evalPolyLayer) { try { evalMap.removeLayer(window.__evalPolyLayer); } catch(e) {} }
    window.__evalPolyLayer = L.layerGroup().addTo(evalMap);
    const gtFeats = (d.gt_geojson && d.gt_geojson.features) || [], predFeats = (d.pred_geojson && d.pred_geojson.features) || []; let allBounds = null;
    function addFeature(feat, isGT) {
        try {
            if (!feat || !feat.geometry) return false; const geom = feat.geometry; if (geom.type !== 'Polygon' && geom.type !== 'MultiPolygon') return false;
            let rings = geom.type === 'Polygon' ? [geom.coordinates[0]] : geom.coordinates.map(poly => poly[0]);
            const style = isGT ? { color:'#60a5fa', weight:2, opacity:0.95, fillColor:'#60a5fa', fillOpacity:0.35 } : { color:'#f87171', weight:2, opacity:0.95, fillColor:'#f87171', fillOpacity:0.35 };
            rings.forEach(ring => { if (!ring || ring.length < 3) return; const latlngs = ring.map(pt => [Number(pt[1]), Number(pt[0])]).filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1])); if (latlngs.length < 3) return; const poly = L.polygon(latlngs, style); poly.addTo(window.__evalPolyLayer); try { const b = poly.getBounds(); if (b && b.isValid()) allBounds = allBounds ? allBounds.extend(b) : b; } catch(e) {} });
            return true;
        } catch (err) { return false; }
    }
    gtFeats.forEach(f => { addFeature(f, true); }); predFeats.forEach(f => { addFeature(f, false); });
    if (allBounds && allBounds.isValid()) { try { evalMap.fitBounds(allBounds, { padding: [40, 40], maxZoom: 18 }); } catch(e) {} }
    setTimeout(() => { try { evalMap.invalidateSize(); } catch(e) {} }, 200);
}

function clearEvaluationMetrics() {
    ['evalMap','evalF1','evalRecall','evalPrecision'].forEach(id => { const e = document.getElementById(id); if (e) e.textContent = '—'; });
    document.getElementById('evalSamples').textContent = '0'; document.getElementById('evalSamples2').textContent = '—';
    if (window.__evalPolyLayer && evalMap) { try { evalMap.removeLayer(window.__evalPolyLayer); } catch(e) {} window.__evalPolyLayer = null; }
    if (evalDrawnItems) { try { evalDrawnItems.clearLayers(); } catch(e) {} } evalAoiBounds = null;
    document.getElementById('evalMetrics').innerHTML = '<p style="color:#64748b;">Run evaluation to see metrics</p>'; updateStatusBadge('idle');
}

function runPrediction() {
    const predictionStartedAt = Date.now();
    if (!loadedModelPath) { if (availableModels.length > 0) { loadModel(availableModels[0].path); } else { addLog('warning', 'Load a model first'); } return; }
    if (!mapPredict) { addLog('error','Predict map not ready'); return; }

    let bounds;
    const modelMeta = window.loadedModelMeta || {};
    const modelZoom = (typeof trainingZoom === 'number' && trainingZoom > 0) ? trainingZoom : null;
    const modelTileSize = modelMeta.tile_size || 64;
    const impliedZoom = modelTileSize > 256 ? 18 : 17;
    if (aoiBounds && aoiBounds.isValid()) { bounds = aoiBounds; } else { const current = mapPredict.getZoom(); if (modelZoom && current !== modelZoom) { mapPredict.setZoom(modelZoom, {animate:false}); } else if (modelTileSize > 256 && current < impliedZoom) { mapPredict.setZoom(impliedZoom, {animate:false}); } bounds = mapPredict.getBounds(); }

    const thr = parseFloat(document.getElementById('predictThreshold')?.value || '0.5'), minA = parseInt(document.getElementById('predictMinArea')?.value || '20', 10), smoothSigma = parseFloat(document.getElementById('predictSmoothSigma')?.value || '0.6'), simplifyTolerance = parseFloat(document.getElementById('predictSimplifyTolerance')?.value || '0.008'), morphClose = !!document.getElementById('predictMorphClose')?.checked;
    const ps = document.getElementById('predictStatus'); if (ps) { ps.textContent = '⏳ Predicting...'; ps.style.color = '#fbbf24'; }
    const curZoom = mapPredict.getZoom(), aoiBoundsPayload = (aoiBounds && aoiBounds.isValid()) ? { north: aoiBounds.getNorth(), south: aoiBounds.getSouth(), east: aoiBounds.getEast(), west: aoiBounds.getWest() } : null;

    let detScale = 1.0;
    if (curZoom === 17) detScale = 2.0;
    if (curZoom === 18) detScale = 1.5;
    if (curZoom === 19) detScale = 1.0;

    fetch('/api/map/capture', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ bounds: {north: bounds.getNorth(), south: bounds.getSouth(), east: bounds.getEast(), west: bounds.getWest()}, zoom: curZoom, service_id: 'satellite', project_id: currentProjectId }) })
    .then(r => r.json()).then(cd => { if (!cd.success) throw new Error(cd.error || 'capture failed'); return fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model_path: loadedModelPath, image_path: cd.filepath, bounds: cd.bounds, threshold: thr, min_area: minA, label: currentLabel, backbone: currentBackbone, zoom: curZoom, smooth_sigma: smoothSigma, simplify_tolerance: simplifyTolerance, morph_close: morphClose, aoi_bounds: aoiBoundsPayload, detection_scale: detScale }) }); })
    .then(r => r.json()).then(d => {
        if (!d.success) { addLog('error','Predict failed: ' + (d.error||'')); if (ps) { ps.textContent = '❌ Error'; ps.style.color = '#f87171'; } return; }
        lastPredictionGeojson = d.geojson; lastPredictionShp = d.shp_url || d.shp_path;
        const shpBtn = document.getElementById('predictShpBtn');
        if (shpBtn) { shpBtn.style.display = lastPredictionShp ? 'block' : 'none'; }
        const count = d.geojson ? d.geojson.features.length : 0; document.getElementById('predictObjectCount').textContent = count;
        if (ps) { ps.textContent = `✅ ${count} objects`; ps.style.color = '#34d399'; }
        const elapsedSeconds = Number.isFinite(Number(d.duration_seconds)) ? Number(d.duration_seconds) : Math.max(0, (Date.now() - predictionStartedAt) / 1000);
        addLog('success', `Prediction completed in ${elapsedSeconds.toFixed(1)}s: ${count} objects`);
        if (predictResultLayer) { mapPredict.removeLayer(predictResultLayer); predictResultLayer = null; }
        if (count > 0 && d.geojson) { predictResultLayer = L.geoJSON(d.geojson, { style: function(f) { const c = Number(f.properties?.confidence || 0.5); return { color: '#22c55e', weight: 2, opacity: 0.95, fillColor: '#22c55e', fillOpacity: 0.15 + c * 0.35 }; }, onEachFeature: function(f, layer) { const p = f.properties || {}, conf = ((p.confidence || 0) * 100).toFixed(1), area = Math.round(p.area || 0), cls = p.class_name || 'object', iid = p.instance_id !== undefined ? p.instance_id : '—'; layer.bindPopup(`<div style="font-family:'Segoe UI',monospace;font-size:12px;color:#e2e8f0;background:#0f172a;padding:6px 10px;border-radius:4px;"><div style="font-weight:700;color:#22c55e;margin-bottom:2px;">${cls}</div><div>Instance: <b>#${iid}</b></div><div>Confidence: <b style="color:#34d399;">${conf}%</b></div><div>Area: <b>${area}</b> px</div></div>`); layer.on('mouseover', function() { this.openPopup(); }); } }).addTo(mapPredict); try { const b = predictResultLayer.getBounds(); if (b.isValid()) mapPredict.fitBounds(b, {padding:[40,40]}); } catch(e) {} }
        renderPredictResults(d.geojson);
    }).catch(e => { addLog('error','Error: ' + e.message); if (ps) { ps.textContent = '❌ ' + e.message; ps.style.color = '#f87171'; } });
}

(function wirePredictPostprocessControls() { const slider = document.getElementById('predictSmoothSigma'), value = document.getElementById('predictSmoothSigmaVal'); if (slider && value) slider.addEventListener('input', () => { value.textContent = Number(slider.value).toFixed(1); }); })();

function renderPredictResults(geojson) {
    const panel = document.getElementById('predictResults'); if (!panel) return;
    if (!geojson || !geojson.features || geojson.features.length === 0) { panel.innerHTML = '<p style="color:#64748b;font-size:11px;">No objects detected</p>'; window.__lastPredFeatures = null; return; }
    const feats = [...geojson.features].sort((a, b) => (b.properties?.confidence || 0) - (a.properties?.confidence || 0));
    let html = `<div style="font-size:10px;color:#64748b;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid #1e293b;">${feats.length} object${feats.length === 1 ? '' : 's'} detected</div>`;
    feats.forEach((f, i) => { const p = f.properties || {}, conf = ((p.confidence || 0) * 100).toFixed(1), area = Math.round(p.area || 0), cls = p.class_name || 'default', confColor = conf >= 70 ? '#34d399' : conf >= 50 ? '#fbbf24' : '#f87171'; html += `<div class="predict-result-row" style="display:flex;align-items:center;gap:6px;padding:5px 6px;border-bottom:1px solid #1e293b;font-size:10px;color:#e2e8f0;cursor:pointer;transition:background 0.15s;" onmouseover="this.style.background='#1e293b'" onmouseout="this.style.background='transparent'" onclick="zoomToPrediction(${i})"><span style="display:inline-flex;align-items:center;justify-content:center;min-width:18px;height:18px;border-radius:3px;background:#1e293b;color:#4ecdc4;font-size:9px;font-weight:700;">${i + 1}</span><span style="flex:1;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${cls}</span><span style="color:${confColor};font-weight:700;">${conf}%</span><span style="color:#64748b;font-size:9px;">${area}px</span></div>`; });
    panel.innerHTML = html; window.__lastPredFeatures = feats;
}

function zoomToPrediction(idx) { const feats = window.__lastPredFeatures; if (!feats || !feats[idx] || !predictResultLayer || !mapPredict) return; const target = feats[idx], tp = target.properties || {}; let found = false; predictResultLayer.eachLayer(function(layer) { if (found) return; const lp = layer.feature?.properties; if (!lp) return; if (lp.confidence === tp.confidence && lp.area === tp.area && lp.class_name === tp.class_name) { try { const b = layer.getBounds(); if (b.isValid()) mapPredict.fitBounds(b, {padding:[120,120], maxZoom:18}); layer.openPopup(); found = true; } catch(e) {} } }); }

function clearPrediction() { if (predictResultLayer) { mapPredict.removeLayer(predictResultLayer); predictResultLayer = null; } if (lastPredictMaskLayer) { mapPredict.removeLayer(lastPredictMaskLayer); lastPredictMaskLayer = null; } document.getElementById('predictObjectCount').textContent = '0'; document.getElementById('predictResults').innerHTML = '<p style="color:#64748b;font-size:11px;">No predictions yet</p>'; document.getElementById('predictStatusText').textContent = 'Pan to any region, or draw an AOI'; window.__lastPredFeatures = null; }
function downloadShapefile() {
    if (lastPredictionShp) {
        const a = document.createElement('a'); a.href = lastPredictionShp;
        a.download = lastPredictionShp.split('/').pop() || 'prediction.zip';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
    } else if (lastPredictionGeojson && lastPredictionGeojson.features && lastPredictionGeojson.features.length) {
        fetch('/api/export/shapefile', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ label: currentLabel || 'prediction', features: lastPredictionGeojson.features })
        }).then(r => r.json()).then(d => {
            if (d.success && d.filename) {
                const safe = (currentLabel || 'prediction').toLowerCase().replace(' ', '_');
                const rel = `/labels/${safe}/evaluation/shp/${d.filename}`;
                const a = document.createElement('a'); a.href = rel; a.download = d.filename;
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
            }
        });
    }
}
function downloadGeoJSON() {
    if (!lastPredictionGeojson) return;
    const cleanGeojson = {
        type: 'FeatureCollection',
        features: (lastPredictionGeojson.features || []).map(f => {
            const p = Object.assign({}, f.properties || {});
            delete p.bbox; // Remove array bbox from attributes for ArcGIS Pro compatibility
            return {
                type: 'Feature',
                bbox: f.bbox || undefined,
                geometry: f.geometry,
                properties: p
            };
        })
    };
    const blob = new Blob([JSON.stringify(cleanGeojson, null, 2)], {type:'application/geo+json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_' + Date.now() + '.geojson';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
function exportGeoJSON() { downloadGeoJSON(); }
function exportShapefile() { downloadShapefile(); }

function runBatchPredict() { if (!loadedModelPath) { addLog('warning', 'Load a model first'); return; } const folder = prompt("Enter folder path with images:"); if (!folder) return; const thr = parseFloat(document.getElementById('predictThreshold')?.value || '0.5'), minA = parseInt(document.getElementById('predictMinArea')?.value || '20', 10); const ps = document.getElementById('predictStatus'); if (ps) { ps.textContent = '⏳ Batch predicting...'; ps.style.color = '#fbbf24'; } fetch('/api/predict_folder', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder, label: currentLabel, model_path: loadedModelPath, threshold: thr, min_area: minA }) }).then(r => r.json()).then(d => { if (!d.success) { addLog('error', 'Batch failed: ' + (d.error || '')); if (ps) { ps.textContent = '❌ ' + (d.error || 'failed'); ps.style.color = '#f87171'; } return; } addLog('success', `Batch: ${d.n_files} files → ${d.total_objects} objects`); if (ps) { ps.textContent = `✅ ${d.total_objects} objects across ${d.n_files} files`; ps.style.color = '#34d399'; } document.getElementById('predictObjectCount').textContent = d.total_objects; }).catch(e => { addLog('error', 'Batch error: ' + e.message); if (ps) { ps.textContent = '❌ ' + e.message; ps.style.color = '#f87171'; } }); }

function refreshPublicDatasets() { const tbody = document.getElementById('publicDatasetsBody'); if (!tbody) return; tbody.innerHTML = ''; let idx = 1; for (const [key, s] of Object.entries(MAP_SERVICES)) { tbody.innerHTML += `<tr><td>${idx}</td><td><strong style="color:#e2e8f0;">${s.name}</strong><span style="font-size:9px;padding:1px 5px;border-radius:3px;margin-left:6px;background:${s.internal ? '#064e3b' : '#7f1d1d'};color:${s.internal ? '#6ee7b7' : '#fecaca'};">${s.internal ? 'IGA' : 'PUBLIC'}</span></td><td>${s.resolution || '—'}</td><td>${s.type || 'Raster'}</td><td>Z ${s.max_zoom || '—'}</td><td style="font-size:10px;">${s.attribution || '—'}</td><td><button class="btn btn-info btn-xs" onclick="useDataset('${key}')"><i class="fas fa-external-link-alt"></i></button></td></tr>`; idx++; } document.getElementById('publicDatasetsCount').textContent = Object.keys(MAP_SERVICES).length + ' public datasets'; document.getElementById('publicBadge').textContent = Object.keys(MAP_SERVICES).length; }
function useDataset(key) { const m = document.getElementById('mapServiceSelect'); if (m) m.value = key; changeMapService(); switchWorkflow('train'); }
function loadUploadedDatasets() { fetch('/api/uploaded_datasets').then(r => r.json()).then(d => { const tbody = document.getElementById('uploadedDatasetsBody'); if (!tbody) return; tbody.innerHTML = ''; if (d.success && d.sessions && d.sessions.length) { d.sessions.forEach((s, i) => { tbody.innerHTML += `<tr><td>${i+1}</td><td>${s.session_id}</td><td>${s.upload_date ? new Date(s.upload_date).toLocaleString() : '—'}</td><td>${s.total_size_mb || 0} MB</td><td>${s.file_count || 0}</td><td>—</td><td><button class="btn btn-danger btn-xs" onclick="deleteDataset('${s.session_id}')">Delete</button></td></tr>`; }); } }); }
function deleteDataset(sid) { if (!confirm('Delete session?')) return; fetch('/api/uploaded_datasets/' + sid, {method:'DELETE'}).then(r => r.json()).then(d => { if (d.success) loadUploadedDatasets(); }); }

window.toggleDrawAoiMode = toggleDrawAoiMode; window.handleAoiCreated = handleAoiCreated; window.zoomToAoi = zoomToAoi; window.toggleEditAoi = toggleEditAoi; window.deleteUserAoi = deleteUserAoi; window.clearAllUserAois = clearAllUserAois; window.toggleClusterOptions = toggleClusterOptions; window.executeClusterAOIs = executeClusterAOIs;
window.selectProject = selectProject; window.deleteProject = deleteProject; window.showCreateProject = showCreateProject; window.closeCreateProject = closeCreateProject; window.createProjectFromNewModal = createProjectFromNewModal; window.createNewLabel = createNewLabel; window.selectLabel = selectLabel; window.startTraining = startTraining; window.stopTraining = stopTraining; window.runOverfitTest = runOverfitTest; window.loadModel = loadModel; window.deleteModel = deleteModel; window.showLoadModelDialog = showLoadModelDialog; window.evaluateModel = evaluateModel; window.clearEvaluationMetrics = clearEvaluationMetrics; window.drawEvaluationPolygons = drawEvaluationPolygons; window.runPrediction = runPrediction; window.runBatchPredict = runBatchPredict; window.clearPrediction = clearPrediction; window.downloadShapefile = downloadShapefile; window.downloadGeoJSON = downloadGeoJSON; window.exportGeoJSON = exportGeoJSON; window.exportShapefile = exportShapefile; window.mergeAllLabels = mergeAllLabels; window.downloadFile = downloadFile; window.saveAnnotations = saveAnnotations; window.loadAnnotations = loadAnnotations; window.clearAll = clearAll; window.deleteSelectedShape = deleteSelectedShape; window.previewLabelClusterAOIs = previewLabelClusterAOIs; window.clearLabelClusterAOIs = clearLabelClusterAOIs; window.setAOI = setAOI; window.captureMap = captureMap; window.lockTrainingScale = lockTrainingScale; window.buildDataset = buildDataset; window.saveTrainedModel = saveTrainedModel; window.openFolderBrowser = openFolderBrowser; window.showDashboard = showDashboard; window.showProjects = showProjects; window.showModels = showModels; window.showPublicDatasets = showPublicDatasets; window.showUploadedDatasets = showUploadedDatasets; window.showPurchasedDatasets = showPurchasedDatasets; window.showDownload = showDownload; window.switchWorkflow = switchWorkflow; window.searchLocation = searchLocation; window.searchPredictLocation = searchPredictLocation; window.searchEvalLocation = searchEvalLocation; window.refreshPublicDatasets = refreshPublicDatasets; window.refreshAnnotationModalList = refreshAnnotationModalList; window.loadUploadedDatasets = loadUploadedDatasets; window.loadProjects = loadProjects; window.loadModels = loadModels; window.loadLabels = loadLabels; window.renderPredictResults = renderPredictResults; window.zoomToPrediction = zoomToPrediction; window.getCurrentModelType = getCurrentModelType; window.isMaskRCNNModel = isMaskRCNNModel; window.toggleAdvanced = toggleAdvanced; window.getAdvancedTrainingSettings = getAdvancedTrainingSettings; window.loadModelFromModalDropdown = loadModelFromModalDropdown; window.refreshEvaluateModelStatus = refreshEvaluateModelStatus; window.layerToPolygonRing = layerToPolygonRing; window.layerToPolygonFeature = layerToPolygonFeature; window.ringToPolygonGeometry = ringToPolygonGeometry; window.featureToPolygonRing = featureToPolygonRing; window.importDlpk = importDlpk; window.exportDlpk = exportDlpk; window.downloadModel = downloadModel; window.loadFineTuneModelChoices = loadFineTuneModelChoices; window.selectFineTuneExistingModel = selectFineTuneExistingModel; window.importFineTuneModel = importFineTuneModel; window.startTrainingOrFineTune = startTrainingOrFineTune; window.startFineTune = startFineTune; window.exportModelWithFormat = exportModelWithFormat;