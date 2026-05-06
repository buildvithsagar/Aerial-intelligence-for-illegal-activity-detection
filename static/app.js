const state = {
    mode: "detection",
    running: false,
    forecastEnabled: true,
    uploadedFile: "",
    zones: [],
    frameWidth: 0,
    frameHeight: 0,
    map: null,
    coverage: null,
    marker: null,
    chartTick: 0,
    nlpDebounceTimer: null,
    lastNlpPrompt: "",
};

// Utility: Debounce with leading execution prevention
function debounceNlp(fn, delayMs = 300) {
    return function(...args) {
        clearTimeout(state.nlpDebounceTimer);
        state.nlpDebounceTimer = setTimeout(() => fn(...args), delayMs);
    };
}

const DEFAULT_LAT = 28.6765;
const DEFAULT_LON = 77.5005;
const DEFAULT_WIDTH_METERS = 100;
const DEFAULT_ZOOM = 16;

const els = {
    sourceType: document.getElementById("sourceType"),
    videoSelect: document.getElementById("videoSelect"),
    videoUpload: document.getElementById("videoUpload"),
    cameraIndex: document.getElementById("cameraIndex"),
    rtspUrl: document.getElementById("rtspUrl"),
    lightMode: document.getElementById("lightMode"),
    zoneEnabled: document.getElementById("zoneEnabled"),
    maxPeople: document.getElementById("maxPeople"),
    overcrowd: document.getElementById("overcrowd"),
    startBtn: document.getElementById("startBtn"),
    stopBtn: document.getElementById("stopBtn"),
    clearZonesBtn: document.getElementById("clearZonesBtn"),
    resetBtn: document.getElementById("resetBtn"),
    mainStream: document.getElementById("mainStream"),
    streamContainer: document.getElementById("streamContainer"),
    zoneCanvas: document.getElementById("zoneCanvas"),
    statusBadge: document.getElementById("statusBadge"),
    countValue: document.getElementById("countValue"),
    trafficValue: document.getElementById("trafficValue"),
    fpsValue: document.getElementById("fpsValue"),
    peakValue: document.getElementById("peakValue"),
    alertsValue: document.getElementById("alertsValue"),
    zoneCountValue: document.getElementById("zoneCountValue"),
    thumbNormal: document.getElementById("thumbNormal"),
    thumbDetection: document.getElementById("thumbDetection"),
    thumbHeatmap: document.getElementById("thumbHeatmap"),
    latInput: document.getElementById("latInput"),
    lonInput: document.getElementById("lonInput"),
    widthInput: document.getElementById("widthInput"),
    detectLocationBtn: document.getElementById("detectLocationBtn"),
    updateMapBtn: document.getElementById("updateMapBtn"),
    locationMeta: document.getElementById("locationMeta"),
    fileRow: document.getElementById("fileRow"),
    uploadRow: document.getElementById("uploadRow"),
    cameraRow: document.getElementById("cameraRow"),
    rtspRow: document.getElementById("rtspRow"),
    nlpProvider: document.getElementById("nlpProvider"),
    nlpModel: document.getElementById("nlpModel"),
    nlpInput: document.getElementById("nlpInput"),
    nlpSendBtn: document.getElementById("nlpSendBtn"),
    nlpReply: document.getElementById("nlpReply"),
    realtimeClock: document.getElementById("realtimeClock"),
    forecastToggle: document.getElementById("forecastToggle"),
    forecastCard: document.getElementById("forecastCard"),
    forecastChart: document.getElementById("forecastChart"),
    forecastMethod: document.getElementById("forecastMethod"),
    refreshForecast: document.getElementById("refreshForecast"),
    forecastLoading: document.getElementById("forecastLoading"),
    forecastMeta: document.getElementById("forecastMeta"),
};

const chartCtx = document.getElementById("countChart").getContext("2d");
const countChart = new Chart(chartCtx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "People",
            data: [],
            borderColor: "#19b394",
            backgroundColor: "rgba(25,179,148,0.2)",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
        }],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { ticks: { color: "#9eb1c8" }, grid: { color: "#2f3947" } },
            y: { ticks: { color: "#9eb1c8" }, grid: { color: "#2f3947" }, beginAtZero: true },
        },
        plugins: {
            legend: { labels: { color: "#dce6f3" } },
        },
    },
});

// Forecast Chart
const forecastCtx = els.forecastChart?.getContext("2d");
const forecastChart = forecastCtx ? new Chart(forecastCtx, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            {
                label: "Forecast",
                data: [],
                borderColor: "#43b3ff",
                backgroundColor: "rgba(67,179,255,0.1)",
                tension: 0.3,
                pointRadius: 2,
                borderWidth: 2,
                borderDash: [5, 5],
            },
            {
                label: "Upper CI (95%)",
                data: [],
                borderColor: "rgba(67,179,255,0.3)",
                backgroundColor: "rgba(67,179,255,0.05)",
                fill: false,
                borderWidth: 1,
                pointRadius: 0,
                borderDash: [2, 2],
            },
            {
                label: "Lower CI (95%)",
                data: [],
                borderColor: "rgba(67,179,255,0.3)",
                backgroundColor: "rgba(67,179,255,0.05)",
                fill: false,
                borderWidth: 1,
                pointRadius: 0,
                borderDash: [2, 2],
            }
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { ticks: { color: "#9eb1c8" }, grid: { color: "#2f3947" } },
            y: { ticks: { color: "#9eb1c8" }, grid: { color: "#2f3947" }, beginAtZero: true },
        },
        plugins: {
            legend: { labels: { color: "#dce6f3" }, display: true },
        },
        interaction: {
            mode: 'index',
            intersect: false,
        },
    },
}) : null;

function setBadge(live) {
    const dot = els.statusBadge.querySelector('.status-dot');
    const txt = els.statusBadge.querySelector('.status-text');
    if (txt) txt.textContent = live ? "LIVE" : "IDLE";
    els.statusBadge.classList.toggle("live", live);
    document.body.classList.toggle("video-running", live);
}

function updateClock() {
    els.realtimeClock.textContent = new Date().toLocaleTimeString();
}

function setForecastVisibility(enabled) {
    state.forecastEnabled = !!enabled;

    if (els.forecastCard) {
        els.forecastCard.classList.toggle("hidden", !state.forecastEnabled);
    }

    if (!state.forecastEnabled) {
        if (els.forecastLoading) {
            els.forecastLoading.style.display = "none";
        }
        if (els.forecastMeta) {
            els.forecastMeta.textContent = "";
        }
        if (forecastChart) {
            forecastChart.data.labels = [];
            forecastChart.data.datasets[0].data = [];
            forecastChart.data.datasets[1].data = [];
            forecastChart.data.datasets[2].data = [];
            forecastChart.update("none");
        }
    }
}

function currentModeUrl() {
    return `/stream?mode=${state.mode}&t=${Date.now()}`;
}

function refreshMainStream() {
    els.mainStream.src = currentModeUrl();
}

function setMode(mode, refreshStream = true) {
    state.mode = mode;
    document.querySelectorAll(".mode-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    [els.thumbNormal, els.thumbDetection, els.thumbHeatmap].forEach((t) => t.classList.remove("selected"));
    if (mode === "normal") els.thumbNormal.classList.add("selected");
    if (mode === "detection") els.thumbDetection.classList.add("selected");
    if (mode === "heatmap") els.thumbHeatmap.classList.add("selected");
    // Also update .view-tab buttons if present
    document.querySelectorAll('.view-tab').forEach(t => t.classList.toggle('selected', t.dataset.mode === mode));
    if (refreshStream) {
        refreshMainStream();
    }
}

function resetDashboardView() {
    state.mode = "detection";
    state.running = false;
    state.uploadedFile = "";
    state.zones = [];
    state.frameWidth = 0;
    state.frameHeight = 0;
    state.chartTick = 0;

    els.sourceType.value = "file";
    els.videoUpload.value = "";
    els.cameraIndex.value = 0;
    els.rtspUrl.value = "";
    els.lightMode.checked = false;
    els.zoneEnabled.checked = false;
    els.forecastToggle.checked = true;
    els.maxPeople.value = 350;
    els.overcrowd.value = 80;
    els.latInput.value = DEFAULT_LAT;
    els.lonInput.value = DEFAULT_LON;
    els.widthInput.value = DEFAULT_WIDTH_METERS;

    updateSourceRows();
    setForecastVisibility(true);
    setMode("detection", false);
    setBadge(false);

    els.locationMeta.textContent = "Tip: click map or drag marker to fine-tune location.";

    if (state.coverage && state.map) {
        state.map.removeLayer(state.coverage);
        state.coverage = null;
    }

    if (state.marker && state.map) {
        state.marker.setLatLng([DEFAULT_LAT, DEFAULT_LON]);
    }

    if (state.map) {
        state.map.setView([DEFAULT_LAT, DEFAULT_LON], DEFAULT_ZOOM);
    }

    if (els.mainStream) {
        els.mainStream.removeAttribute("src");
    }

    const canvas = els.zoneCanvas;
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    countChart.data.labels = [];
    countChart.data.datasets[0].data = [];
    countChart.update("none");

    if (forecastChart) {
        forecastChart.data.labels = [];
        forecastChart.data.datasets[0].data = [];
        forecastChart.data.datasets[1].data = [];
        forecastChart.data.datasets[2].data = [];
        forecastChart.update("none");
    }

    els.countValue.textContent = "0";
    els.trafficValue.textContent = "0.00%";
    els.fpsValue.textContent = "0.00";
    els.peakValue.textContent = "0.00%";
    els.alertsValue.textContent = "0";
    els.zoneCountValue.textContent = "0";

    sendZones().catch(() => {});
    pushConfig().catch(() => {});
    els.mainStream.removeAttribute("src");
    drawZones();
}

function updateSourceRows() {
    const type = els.sourceType.value;
    els.fileRow.classList.toggle("hidden", type !== "file");
    els.uploadRow.classList.toggle("hidden", type !== "upload");
    els.cameraRow.classList.toggle("hidden", type !== "camera");
    els.rtspRow.classList.toggle("hidden", type !== "rtsp");
}

async function loadVideos() {
    const res = await fetch("/api/videos");
    const data = await res.json();
    els.videoSelect.innerHTML = "";
    data.videos.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        els.videoSelect.appendChild(opt);
    });
}

async function uploadVideoIfNeeded() {
    if (els.sourceType.value !== "upload") return "";
    const file = els.videoUpload.files[0];
    if (!file) throw new Error("Select a video to upload");

    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    state.uploadedFile = data.filename;
    return state.uploadedFile;
}

async function startProcessing() {
    const sourceType = els.sourceType.value;
    let fileName = "";
    if (sourceType === "upload") {
        fileName = await uploadVideoIfNeeded();
    }

    const payload = {
        source_type: sourceType,
        file_name: sourceType === "file" ? els.videoSelect.value : fileName,
        camera_index: Number(els.cameraIndex.value || 0),
        rtsp_url: els.rtspUrl.value.trim(),
        light_mode: els.lightMode.checked,
        zone_counting_enabled: els.zoneEnabled.checked,
        max_people: Number(els.maxPeople.value || 350),
        overcrowd_thresh: Number(els.overcrowd.value || 80),
    };

    const res = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to start");

    state.running = true;
    setBadge(true);
    refreshMainStream();
}

async function stopProcessing() {
    await fetch("/api/stop", { method: "POST" });
    state.running = false;
    setBadge(false);
}

async function resetBackendState() {
    await fetch("/api/reset", { method: "POST" });
}

async function pushConfig() {
    const payload = {
        zone_counting_enabled: els.zoneEnabled.checked,
        max_people: Number(els.maxPeople.value || 350),
        overcrowd_thresh: Number(els.overcrowd.value || 80),
    };
    await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
}

async function sendZones() {
    await fetch("/api/zones", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zones: state.zones }),
    });
}

function setupThumbs() {
    const thumbs = [
        { el: els.thumbNormal, mode: "normal" },
        { el: els.thumbDetection, mode: "detection" },
        { el: els.thumbHeatmap, mode: "heatmap" },
    ];

    thumbs.forEach(({ el, mode }) => {
        el.addEventListener("click", () => setMode(mode));
    });

    // Also bind view-tab buttons with data-mode
    document.querySelectorAll('.view-tab[data-mode]').forEach(btn => {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });
}

function initMap() {
    state.map = L.map("map", { attributionControl: false }).setView([DEFAULT_LAT, DEFAULT_LON], DEFAULT_ZOOM);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(state.map);

    state.marker = L.marker([DEFAULT_LAT, DEFAULT_LON], { draggable: true }).addTo(state.map);
    state.marker.on("dragend", () => {
        const pos = state.marker.getLatLng();
        els.latInput.value = pos.lat.toFixed(6);
        els.lonInput.value = pos.lng.toFixed(6);
        updateMapCoverage();
        els.locationMeta.textContent = "Location refined from marker drag.";
    });

    state.map.on("click", (e) => {
        const { lat, lng } = e.latlng;
        els.latInput.value = lat.toFixed(6);
        els.lonInput.value = lng.toFixed(6);
        updateMapCoverage();
        els.locationMeta.textContent = "Location refined from map click.";
    });

    updateMapCoverage();
}

function updateMapCoverage() {
    const lat = Number(els.latInput.value || DEFAULT_LAT);
    const lon = Number(els.lonInput.value || DEFAULT_LON);
    const widthMeters = Number(els.widthInput.value || DEFAULT_WIDTH_METERS);

    state.map.setView([lat, lon], Math.max(state.map.getZoom(), DEFAULT_ZOOM));
    if (state.marker) {
        state.marker.setLatLng([lat, lon]);
    }
    if (state.coverage) state.map.removeLayer(state.coverage);
    state.coverage = L.circle([lat, lon], {
        radius: widthMeters / 2,
        color: "#ff6861",
        fillColor: "#ff6861",
        fillOpacity: 0.25,
        weight: 2,
    }).addTo(state.map);
}

function detectLiveLocation() {
    if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser.");
        return;
    }

    els.locationMeta.textContent = "Detecting location... collecting best GPS fix.";

    const samples = [];
    const started = Date.now();
    const maxDurationMs = 8000;
    const maxSamples = 4;

    const finalize = () => {
        if (samples.length === 0) {
            alert("Unable to detect location. Check browser location permissions and GPS.");
            els.locationMeta.textContent = "Location detection failed.";
            return;
        }

        samples.sort((a, b) => a.accuracy - b.accuracy);
        const best = samples[0];

        els.latInput.value = best.lat.toFixed(6);
        els.lonInput.value = best.lon.toFixed(6);
        state.map.setView([best.lat, best.lon], best.accuracy <= 30 ? 18 : 16);
        updateMapCoverage();

        const accText = Number.isFinite(best.accuracy) ? `${Math.round(best.accuracy)}m` : "unknown";
        els.locationMeta.textContent = `Detected location accuracy: ${accText}. Drag marker or click map to refine.`;
    };

    const watchId = navigator.geolocation.watchPosition(
        (pos) => {
            samples.push({
                lat: pos.coords.latitude,
                lon: pos.coords.longitude,
                accuracy: pos.coords.accuracy ?? Number.POSITIVE_INFINITY,
            });

            const doneBySamples = samples.length >= maxSamples;
            const doneByTime = Date.now() - started >= maxDurationMs;
            if (doneBySamples || doneByTime) {
                navigator.geolocation.clearWatch(watchId);
                finalize();
            }
        },
        () => {
            navigator.geolocation.clearWatch(watchId);
            alert("Unable to detect location. Check browser location permissions and GPS.");
            els.locationMeta.textContent = "Location detection failed.";
        },
        {
            enableHighAccuracy: true,
            timeout: 7000,
            maximumAge: 0,
        }
    );

    setTimeout(() => {
        navigator.geolocation.clearWatch(watchId);
        if (samples.length > 0) {
            finalize();
        }
    }, maxDurationMs + 200);
}

function drawZones() {
    const canvas = els.zoneCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#4da3ff";
    ctx.fillStyle = "rgba(30, 74, 120, 0.2)";

    state.zones.forEach((z) => {
        const xScale = canvas.width / (state.frameWidth || canvas.width);
        const yScale = canvas.height / (state.frameHeight || canvas.height);
        const x = z[0] * xScale;
        const y = z[1] * yScale;
        const w = (z[2] - z[0]) * xScale;
        const h = (z[3] - z[1]) * yScale;
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
    });
}

function setupZoneCanvas() {
    const canvas = els.zoneCanvas;
    const container = els.streamContainer;
    let start = null;

    function resizeCanvasToImage() {
        canvas.width = container.clientWidth || 640;
        canvas.height = container.clientHeight || 360;
        drawZones();
    }

    const resizeObserver = new ResizeObserver(resizeCanvasToImage);
    resizeObserver.observe(container);
    window.addEventListener("resize", resizeCanvasToImage);

    canvas.addEventListener("mousedown", (e) => {
        if (!els.zoneEnabled.checked) return;
        start = { x: e.offsetX, y: e.offsetY };
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!start) return;
        drawZones();
        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "#90d8ff";
        ctx.setLineDash([6, 4]);
        const w = e.offsetX - start.x;
        const h = e.offsetY - start.y;
        ctx.strokeRect(start.x, start.y, w, h);
        ctx.setLineDash([]);
    });

    canvas.addEventListener("mouseup", async (e) => {
        if (!start) return;

        const x1 = Math.min(start.x, e.offsetX);
        const y1 = Math.min(start.y, e.offsetY);
        const x2 = Math.max(start.x, e.offsetX);
        const y2 = Math.max(start.y, e.offsetY);
        start = null;

        if (Math.abs(x2 - x1) < 8 || Math.abs(y2 - y1) < 8) {
            drawZones();
            return;
        }

        const xScale = (state.frameWidth || canvas.width) / canvas.width;
        const yScale = (state.frameHeight || canvas.height) / canvas.height;
        state.zones.push([
            Math.round(x1 * xScale),
            Math.round(y1 * yScale),
            Math.round(x2 * xScale),
            Math.round(y2 * yScale),
        ]);

        drawZones();
        await sendZones();
    });

    resizeCanvasToImage();
}

async function refreshStats() {
    const res = await fetch("/api/stats");
    const s = await res.json();

    state.running = !!s.running;
    state.frameWidth = s.frame_width || state.frameWidth;
    state.frameHeight = s.frame_height || state.frameHeight;

    if (!state.running) {
        els.countValue.textContent = "0";
        els.trafficValue.textContent = "0.00%";
        els.fpsValue.textContent = "0.00";
        els.peakValue.textContent = "0.00%";
        els.alertsValue.textContent = "0";
        els.zoneCountValue.textContent = "0";
        setBadge(false);
        return;
    }

    els.countValue.textContent = String(s.count ?? 0);
    els.trafficValue.textContent = `${(s.traffic ?? 0).toFixed(2)}%`;
    els.fpsValue.textContent = (s.fps ?? 0).toFixed(2);
    els.peakValue.textContent = `${(s.peak_density ?? 0).toFixed(2)}%`;
    els.alertsValue.textContent = String(s.alerts_triggered ?? 0);
    els.zoneCountValue.textContent = String(s.people_in_zones ?? 0);

    setBadge(state.running);

    const now = Date.now() / 1000;
    const points = (s.series || []).filter((p) => now - p.t <= 10);
    state.chartTick += 1;
    if (points.length > 0 && state.chartTick % 2 === 0) {
        const t0 = points[0].t;
        countChart.data.labels = points.map((p) => (p.t - t0).toFixed(1));
        countChart.data.datasets[0].data = points.map((p) => p.count);
        countChart.update("none");
    }

    drawZones();
}

async function refreshForecast() {
    if (!state.forecastEnabled || !state.running || !forecastChart) return;
    
    try {
        els.forecastLoading.style.display = 'block';
        const method = els.forecastMethod?.value || 'hybrid';
        const res = await fetch("/api/forecast", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ method, steps: 120 }),
        });
        const data = await res.json();

        if (!data.forecast_density) {
            els.forecastLoading.style.display = 'none';
            return;
        }

        // Format timestamps for display (minutes:seconds)
        const timestamps = data.timestamps || [];
        const labels = timestamps.map((t, i) => {
            const seconds = Math.round(i * 0.5);
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        });

        // Update forecast chart
        forecastChart.data.labels = labels;
        forecastChart.data.datasets[0].data = data.forecast_count || [];
        forecastChart.data.datasets[1].data = data.confidence_upper || [];
        forecastChart.data.datasets[2].data = data.confidence_lower || [];
        forecastChart.update("none");

        // Update forecast metadata
        const avgForecast = data.forecast_density ? data.forecast_density.reduce((a, b) => a + b, 0) / data.forecast_density.length : 0;
        const maxForecast = Math.max(...(data.forecast_density || [0]));
        els.forecastMeta.textContent = `Method: ${data.method} | Avg: ${avgForecast.toFixed(1)}% | Peak: ${maxForecast.toFixed(1)}% | Current: ${data.current_density.toFixed(1)}%`;
    } catch (err) {
        console.error("Forecast error:", err);
        els.forecastMeta.textContent = "Forecast unavailable";
    } finally {
        els.forecastLoading.style.display = 'none';
    }
}

async function handleNlpCommand() {
    const prompt = els.nlpInput.value.trim();
    if (!prompt) return;

    // Input validation on client side
    if (prompt.length > 500) {
        els.nlpReply.textContent = "Prompt too long (max 500 characters).";
        return;
    }

    els.nlpReply.textContent = "Processing...";
    els.nlpSendBtn.disabled = true;

    const payload = {
        prompt,
        provider: els.nlpProvider.value,
        model: els.nlpModel.value.trim(),
    };

    try {
        const res = await fetch("/api/nlp", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        let reply = data.reply || "No reply from model.";
        if (data.warning) {
            reply = `${reply} (⚠️ ${data.warning})`;
        }
        els.nlpReply.textContent = reply;

        if (data.action && data.action.name !== "noop") {
            try {
                executeNlpAction(data.action);
            } catch (actionErr) {
                els.nlpReply.textContent = `${reply} (Action failed: ${actionErr.message})`;
            }
        }
    } catch (err) {
        els.nlpReply.textContent = `Error: ${err.message}`;
    } finally {
        els.nlpInput.value = "";
        els.nlpSendBtn.disabled = false;
    }
}

function executeNlpAction(action) {
    const p = action.params || {};

    switch (action.name) {
        case "start":
            if (p.source_type) {
                els.sourceType.value = p.source_type;
                if (p.source_type === "camera" && Number.isFinite(p.camera_index)) {
                    els.cameraIndex.value = p.camera_index;
                }
                updateSourceRows();
            }
            els.startBtn.click();
            break;
        case "stop":
            els.stopBtn.click();
            break;
        case "reset":
            els.resetBtn.click();
            break;
        case "set_mode":
            if (p.mode && ["normal", "detection", "heatmap"].includes(p.mode)) {
                setMode(p.mode);
            }
            break;
        case "toggle_zone":
            els.zoneEnabled.checked = !!p.enabled;
            pushConfig();
            break;
        case "toggle_light":
            els.lightMode.checked = !!p.enabled;
            break;
        case "set_max_people":
            if (Number.isFinite(p.value) && p.value > 0) {
                els.maxPeople.value = p.value;
                pushConfig();
            }
            break;
        case "set_overcrowd":
            if (Number.isFinite(p.value) && p.value > 0 && p.value <= 100) {
                els.overcrowd.value = p.value;
                pushConfig();
            }
            break;
    }
}

function bindEvents() {
    els.sourceType.addEventListener("change", updateSourceRows);

    els.nlpSendBtn.addEventListener("click", handleNlpCommand);

        // Forecast controls
        if (els.refreshForecast) {
            els.refreshForecast.addEventListener("click", refreshForecast);
        }
        if (els.forecastMethod) {
            els.forecastMethod.addEventListener("change", refreshForecast);
        }
    
    // Immediate send on Enter, debounced on change for live suggestions
    els.nlpInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleNlpCommand();  // Immediate execution, not debounced
        }
    });

    els.startBtn.addEventListener("click", async () => {
        try {
            await pushConfig();
            await startProcessing();
        } catch (err) {
            alert(err.message || "Failed to start processing");
        }
    });

    els.stopBtn.addEventListener("click", async () => {
        await stopProcessing();
    });

    els.clearZonesBtn.addEventListener("click", async () => {
        state.zones = [];
        drawZones();
        await sendZones();
    });

    els.resetBtn.addEventListener("click", async () => {
        try {
            await resetBackendState();
        } finally {
            resetDashboardView();
        }
    });

    els.zoneEnabled.addEventListener("change", pushConfig);
    if (els.forecastToggle) {
        els.forecastToggle.addEventListener("change", () => {
            setForecastVisibility(els.forecastToggle.checked);
        });
    }
    els.maxPeople.addEventListener("change", pushConfig);
    els.overcrowd.addEventListener("change", pushConfig);

    document.querySelectorAll(".mode-btn").forEach((btn) => {
        btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });

    els.nlpProvider.addEventListener("change", () => {
        const provider = els.nlpProvider.value;
        els.nlpModel.innerHTML = "";
        const modelGroup = document.getElementById('nlpModelGroup');
        
        if (provider === "fallback") {
            els.nlpModel.classList.add("hidden");
            if (modelGroup) modelGroup.classList.add("hidden");
        } else {
            els.nlpModel.classList.remove("hidden");
            if (modelGroup) modelGroup.classList.remove("hidden");
            
            if (provider === "gemini") {
                els.nlpModel.innerHTML = `
                    <option value="gemini-3.1-pro-preview">Gemini 3.1 Pro (Preview)</option>
                    <option value="gemini-3-flash">Gemini 3 Flash (Fast)</option>
                    <option value="gemini-2.5-pro">Gemini 2.5 Pro (Stable)</option>
                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                    <option value="gemini-1.5-flash">Gemini 1.5 Flash (Legacy Fast)</option>
                    <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                `;
            } else if (provider === "openai") {
                els.nlpModel.innerHTML = `
                    <option value="gpt-4o-mini">GPT-4o Mini</option>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                `;
            } else if (provider === "ollama") {
                els.nlpModel.innerHTML = `
                    <option value="llama3.1:8b">Llama 3.1 (8B)</option>
                    <option value="llama3:8b">Llama 3 (8B)</option>
                    <option value="mistral">Mistral</option>
                `;
            }
        }
    });

    els.detectLocationBtn.addEventListener("click", detectLiveLocation);
    els.updateMapBtn.addEventListener("click", updateMapCoverage);
}

async function init() {
    bindEvents();
    setupThumbs();
    setupZoneCanvas();
    updateSourceRows();
    await loadVideos();
    initMap();
    setMode("detection");
    setForecastVisibility(!!els.forecastToggle?.checked);

    els.nlpProvider.dispatchEvent(new Event("change"));

    updateClock();
    setInterval(updateClock, 1000);
    setInterval(() => {
        refreshStats().catch(() => {});
    }, 600);
    setInterval(() => {
        refreshForecast().catch(() => {});
    }, 3000);
}

init();
