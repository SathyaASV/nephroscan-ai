/**
 * NephroScan AI — Software Thermal Proxy Dashboard
 * Faithful browser port of the supplied Python thermal algorithm.
 * Uses navigator.mediaDevices.getUserMedia() — no server camera, no cv2.
 * Educational prototype only. Not an infrared measurement.
 */
(function () {
  'use strict';

  /* ===================== DOM REFS ===================== */
  var startBtn       = document.getElementById('presenceStartBtn');
  var stopBtn        = document.getElementById('presenceStopBtn');
  var clearBtn       = document.getElementById('presenceClearBtn');
  var video          = document.getElementById('presenceVideo');
  var rgbCanvas      = document.getElementById('presenceRgbCanvas');
  var thermCanvas    = document.getElementById('presenceThermalCanvas');
  var placeholder    = document.getElementById('presencePlaceholder');
  var statusEl       = document.getElementById('presenceStatus');
  var logBody        = document.getElementById('presenceLogBody');
  var indexValEl     = document.getElementById('thermalIndexVal');
  var chartCanvas    = document.getElementById('thermalChartCanvas');
  var fpsEl          = document.getElementById('thermalFpsVal');
  var latencyEl      = document.getElementById('thermalLatencyVal');
  var reportEl       = document.getElementById('presenceReport');

  if (!startBtn || !stopBtn || !video || !rgbCanvas || !thermCanvas) {
    console.warn('Expo presence: required DOM elements not found.');
    return;
  }

  var APP_VERSION = (typeof window.APP_VERSION === 'string') ? window.APP_VERSION : '2.1.0';

  /* ===================== INFERNO COLORMAP (256 entries) ===================== */
  var INFERNO_STOPS = [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 67], [245, 125, 21], [250, 186, 47], [252, 255, 164]
  ];
  var INFERNO_LUT = new Uint8Array(256 * 3);
  (function buildLut() {
    var n = INFERNO_STOPS.length;
    for (var i = 0; i < 256; i++) {
      var t = i / 255;
      var idx = t * (n - 1);
      var lo = Math.floor(idx);
      var hi = Math.min(lo + 1, n - 1);
      var f = idx - lo;
      INFERNO_LUT[i * 3]     = Math.round(INFERNO_STOPS[lo][0] + (INFERNO_STOPS[hi][0] - INFERNO_STOPS[lo][0]) * f);
      INFERNO_LUT[i * 3 + 1] = Math.round(INFERNO_STOPS[lo][1] + (INFERNO_STOPS[hi][1] - INFERNO_STOPS[lo][1]) * f);
      INFERNO_LUT[i * 3 + 2] = Math.round(INFERNO_STOPS[lo][2] + (INFERNO_STOPS[hi][2] - INFERNO_STOPS[lo][2]) * f);
    }
  })();

  /* ===================== STATE ===================== */
  var cameraStream    = null;
  var animFrame       = null;
  var rgbCtx          = null;
  var thermCtx        = null;
  var presenceCounter = 1;
  var lastLogTime     = 0;
  var sessionStart    = null;
  var sessionLog      = [];
  var chartHistory    = [];
  var CHART_MAX       = 60;
  var frameCount      = 0;
  var lastFpsTime     = performance.now();

  var LOG_THROTTLE_MS = 2000;
  var BOX_W = 240;
  var BOX_H = 280;

  /* ===================== HELPERS ===================== */

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = 'STATUS: ' + msg;
  }

  function resetLog() {
    if (logBody) {
      logBody.innerHTML = '<tr><td colspan="7" class="expo-empty-log">No data yet. Start the camera to begin.</td></tr>';
    }
    presenceCounter = 1;
    sessionLog = [];
    chartHistory = [];
    if (indexValEl) indexValEl.textContent = '\u2014';
    updateReport(null);
  }

  /* ===================== THERMAL CANVAS — INFERNO COLORMAP ===================== */
  function applyInferno(srcCanvas, tgtCanvas) {
    if (!thermCtx) thermCtx = tgtCanvas.getContext('2d');
    thermCtx.drawImage(srcCanvas, 0, 0, tgtCanvas.width, tgtCanvas.height);
    var imageData = thermCtx.getImageData(0, 0, tgtCanvas.width, tgtCanvas.height);
    var d = imageData.data;
    for (var i = 0; i < d.length; i += 4) {
      var gray = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
      var inverted = 255 - gray;
      var lutIdx = inverted * 3;
      d[i]     = INFERNO_LUT[lutIdx];
      d[i + 1] = INFERNO_LUT[lutIdx + 1];
      d[i + 2] = INFERNO_LUT[lutIdx + 2];
      d[i + 3] = 255;
    }
    thermCtx.putImageData(imageData, 0, 0);
  }

  /* ===================== 240x280 TARGETING BRACKETS ===================== */
  function drawBrackets(ctx, w, h) {
    var cx = w / 2, cy = h / 2;
    var x = Math.round(cx - BOX_W / 2);
    var y = Math.round(cy - BOX_H / 2);
    var bl = 25;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,0,0.85)';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(255,255,0,0.4)';
    ctx.shadowBlur = 4;

    ctx.beginPath();
    ctx.moveTo(x, y + bl); ctx.lineTo(x, y); ctx.lineTo(x + bl, y);
    ctx.moveTo(x + BOX_W - bl, y); ctx.lineTo(x + BOX_W, y); ctx.lineTo(x + BOX_W, y + bl);
    ctx.moveTo(x, y + BOX_H - bl); ctx.lineTo(x, y + BOX_H); ctx.lineTo(x + bl, y + BOX_H);
    ctx.moveTo(x + BOX_W - bl, y + BOX_H); ctx.lineTo(x + BOX_W, y + BOX_H); ctx.lineTo(x + BOX_W, y + BOX_H - bl);
    ctx.stroke();
    ctx.restore();
  }

  /* ===================== BOTTOM SIGNAL PANEL ===================== */
  function drawSignalPanel(ctx, w, h) {
    var px = 25, py = h - 65, pw = 150, ph = 40;
    ctx.save();
    ctx.fillStyle = 'rgba(20,20,30,0.8)';
    ctx.fillRect(px, py, pw, ph);
    ctx.strokeStyle = 'rgba(255,255,0,0.3)';
    ctx.strokeRect(px, py, pw, ph);
    ctx.fillStyle = '#ff6f3c';
    ctx.font = '10px monospace';
    ctx.fillText('SOFTWARE THERMAL PROXY', px + 6, py + 14);
    ctx.fillStyle = '#0ff';
    ctx.fillText('PROTOTYPE v' + APP_VERSION, px + 6, py + 28);
    ctx.restore();
  }

  /* ===================== ROLLING CHART ===================== */
  function drawChart() {
    if (!chartCanvas) return;
    var ctx = chartCanvas.getContext('2d');
    var W = chartCanvas.width, H = chartCanvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, W, H);
    if (chartHistory.length < 2) return;
    var step = W / (CHART_MAX - 1);
    ctx.beginPath();
    ctx.strokeStyle = '#ff6f3c';
    ctx.lineWidth = 1.5;
    for (var i = 0; i < chartHistory.length; i++) {
      var x = i * step;
      var y = H - chartHistory[i] * (H - 8) - 4;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineTo((chartHistory.length - 1) * step, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,111,60,0.1)';
    ctx.fill();
  }

  /* ===================== ROI BRIGHTNESS + RELATIVE INDEX ===================== */
  var temperatureHistory = [];
  var HIST_MAX = 15;

  function computeROI(grayImageData, w, h) {
    var x0 = Math.round((w - BOX_W) / 2);
    var y0 = Math.round((h - BOX_H) / 2);
    var data = grayImageData;
    var sum = 0, count = 0;
    for (var row = y0; row < y0 + BOX_H && row < h; row++) {
      for (var col = x0; col < x0 + BOX_W && col < w; col++) {
        var idx = (row * w + col) * 4;
        sum += data[idx];
        count++;
      }
    }
    return count > 0 ? sum / count : 128;
  }

  function computeGrayData(rgbData) {
    var gray = new Float32Array(rgbData.length / 4);
    for (var i = 0; i < gray.length; i++) {
      gray[i] = 0.299 * rgbData[i * 4] + 0.587 * rgbData[i * 4 + 1] + 0.114 * rgbData[i * 4 + 2];
    }
    return gray;
  }

  function processROI(avgBrightness, w, h) {
    var rawIndex, smoothed, statusText, subjectDetected;

    if (avgBrightness < 220) {
      subjectDetected = true;
      rawIndex = 35.2 + (1.0 - (avgBrightness / 255.0)) * 3.3;
      if (rawIndex < 35.8) rawIndex = 36.2;
      if (rawIndex > 38.2) rawIndex = 36.8;
      temperatureHistory.push(rawIndex);
      if (temperatureHistory.length > HIST_MAX) temperatureHistory.shift();
      smoothed = 0;
      for (var i = 0; i < temperatureHistory.length; i++) smoothed += temperatureHistory[i];
      smoothed = Math.round(smoothed / temperatureHistory.length * 10) / 10;

      if (smoothed >= 37.4) {
        statusText = 'ELEVATED PROXY INDEX';
      } else {
        statusText = 'NORMAL PROXY RANGE';
      }
    } else {
      subjectDetected = false;
      temperatureHistory = [];
      smoothed = 0;
      rawIndex = 0;
      statusText = 'ALIGN SUBJECT IN TARGET BRACKETS';
    }

    var relativeIndex = smoothed > 0 ? Math.round((smoothed - 35.2) / (38.2 - 35.2) * 100) / 100 : 0;
    relativeIndex = Math.max(0, Math.min(1, relativeIndex));

    return {
      avgBrightness: Math.round(avgBrightness * 10) / 10,
      rawIndex: rawIndex,
      smoothed: smoothed,
      relativeIndex: relativeIndex,
      statusText: statusText,
      subjectDetected: subjectDetected,
      sampleCount: temperatureHistory.length
    };
  }

  /* ===================== LIVE REPORT ===================== */
  function updateReport(data) {
    if (!reportEl) return;
    if (!data) {
      reportEl.innerHTML = '<div style="font-size:.72rem;color:#9aa7b2;">Camera inactive. Press Start Camera to begin.</div>';
      return;
    }
    var elapsed = sessionStart ? Math.round((Date.now() - sessionStart.getTime()) / 1000) : 0;
    var mins = Math.floor(elapsed / 60);
    var secs = elapsed % 60;
    var timeStr = mins + 'm ' + secs + 's';
    var trend = chartHistory.length >= 2
      ? (chartHistory[chartHistory.length - 1] > chartHistory[chartHistory.length - 2] ? 'Rising' :
         chartHistory[chartHistory.length - 1] < chartHistory[chartHistory.length - 2] ? 'Falling' : 'Stable')
      : '\u2014';

    reportEl.innerHTML =
      '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.5rem;font-size:.7rem;">' +
      '<div><b style="color:#1a2d3d;">Camera Status</b><br><span style="color:#168a62;">' + (cameraStream ? 'ACTIVE' : 'INACTIVE') + '</span></div>' +
      '<div><b style="color:#1a2d3d;">Presence State</b><br><span style="color:#1a2d3d;">' + (data.subjectDetected ? 'Subject Detected' : 'No Subject') + '</span></div>' +
      '<div><b style="color:#1a2d3d;">ROI Brightness</b><br><span style="color:#1a2d3d;">' + data.avgBrightness + ' lx</span></div>' +
      '<div><b style="color:#1a2d3d;">Relative Index</b><br><span style="color:#1a2d3d;">' + data.relativeIndex.toFixed(2) + '</span></div>' +
      '<div><b style="color:#1a2d3d;">Smoothed Trend</b><br><span style="color:#1a2d3d;">' + (data.smoothed > 0 ? data.smoothed : '\u2014') + '</span></div>' +
      '<div><b style="color:#1a2d3d;">Trend Direction</b><br><span style="color:#1a2d3d;">' + trend + '</span></div>' +
      '<div><b style="color:#1a2d3d;">Sample Count</b><br><span style="color:#1a2d3d;">' + data.sampleCount + ' / ' + HIST_MAX + '</span></div>' +
      '<div><b style="color:#1a2d3d;">FPS</b><br><span style="color:#1a2d3d;">' + (fpsEl ? fpsEl.textContent : '\u2014') + '</span></div>' +
      '<div><b style="color:#1a2d3d;">Frame Latency</b><br><span style="color:#1a2d3d;">' + (latencyEl ? latencyEl.textContent : '\u2014') + ' ms</span></div>' +
      '<div><b style="color:#1a2d3d;">Session Uptime</b><br><span style="color:#1a2d3d;">' + timeStr + '</span></div>' +
      '<div><b style="color:#1a2d3d;">Status</b><br><span style="color:#1a2d3d;">' + data.statusText + '</span></div>' +
      '</div>' +
      '<div style="margin-top:.5rem;font-size:.6rem;color:#9aa7b2;border-top:1px solid #edf2f5;padding-top:.4rem;">' +
      'Disclosure: Calculated from ordinary webcam brightness; not an infrared or medical temperature reading.' +
      '</div>';
  }

  /* ===================== DETECTION LOOP ===================== */
  function detectionLoop() {
    if (!cameraStream) return;
    var loopStart = performance.now();
    if (video.readyState >= 2) {
      if (!rgbCtx) {
        rgbCanvas.width = video.videoWidth;
        rgbCanvas.height = video.videoHeight;
        thermCanvas.width = video.videoWidth;
        thermCanvas.height = video.videoHeight;
        rgbCtx = rgbCanvas.getContext('2d');
        thermCtx = thermCanvas.getContext('2d');
      }

      /* 1. Mirrored frame */
      rgbCtx.save();
      rgbCtx.translate(rgbCanvas.width, 0);
      rgbCtx.scale(-1, 1);
      rgbCtx.drawImage(video, 0, 0, rgbCanvas.width, rgbCanvas.height);
      rgbCtx.restore();

      /* 2. Grayscale + inversion + Inferno colormap */
      applyInferno(rgbCanvas, thermCanvas);

      /* 3. Grayscale data for ROI */
      var rgbData = rgbCtx.getImageData(0, 0, rgbCanvas.width, rgbCanvas.height).data;
      var grayData = computeGrayData(rgbData);

      /* 4. ROI brightness */
      var avgBrightness = computeROI(grayData, rgbCanvas.width, rgbCanvas.height);

      /* 5. Process ROI → relative index, smoothing, status */
      var roi = processROI(avgBrightness, rgbCanvas.width, rgbCanvas.height);

      /* 6. Draw brackets + signal panel on thermal canvas */
      drawBrackets(thermCtx, thermCanvas.width, thermCanvas.height);
      drawSignalPanel(thermCtx, thermCanvas.width, thermCanvas.height);

      /* 7. HUD overlays on thermal canvas */
      thermCtx.save();
      thermCtx.font = '12px monospace';
      thermCtx.fillStyle = '#0ff';
      thermCtx.fillText('LUMINANCE: ' + roi.avgBrightness + ' lx', 30, 30);
      thermCtx.fillStyle = roi.subjectDetected ? '#0f0' : '#ff0';
      thermCtx.fillText(roi.subjectDetected ? roi.statusText : roi.statusText, 30, 50);
      thermCtx.fillStyle = '#0ff';
      thermCtx.fillText('RELATIVE INDEX: ' + roi.relativeIndex.toFixed(2), 30, 70);
      thermCtx.restore();

      /* 8. Update emulated index display */
      if (indexValEl) indexValEl.textContent = roi.relativeIndex.toFixed(2);

      /* 9. Rolling chart */
      chartHistory.push(roi.relativeIndex);
      if (chartHistory.length > CHART_MAX) chartHistory.shift();
      drawChart();

      /* 10. Live report */
      updateReport(roi);

      /* 11. Log entry */
      var now = Date.now();
      if (now - lastLogTime > LOG_THROTTLE_MS) {
        lastLogTime = now;
        var trend = chartHistory.length >= 2
          ? (chartHistory[chartHistory.length - 1] > chartHistory[chartHistory.length - 2] ? 'Rising' :
             chartHistory[chartHistory.length - 1] < chartHistory[chartHistory.length - 2] ? 'Falling' : 'Stable')
          : '\u2014';
        var ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        var id = 'T-' + String(presenceCounter++).padStart(3, '0');
        sessionLog.push({
          time: ts, id: id,
          brightness: roi.avgBrightness,
          relativeIndex: roi.relativeIndex.toFixed(2),
          smoothed: roi.smoothed,
          trend: trend,
          status: roi.statusText,
          samples: roi.sampleCount
        });
        if (logBody) {
          if (logBody.querySelector('.expo-empty-log')) logBody.innerHTML = '';
          var row = document.createElement('tr');
          row.innerHTML = '<td>' + ts + '</td><td>' + id + '</td><td>' + roi.avgBrightness + '</td><td>' + roi.relativeIndex.toFixed(2) + '</td><td>' + roi.smoothed + '</td><td>' + trend + '</td><td>' + roi.statusText + '</td>';
          logBody.prepend(row);
          while (logBody.rows.length > 25) logBody.deleteRow(-1);
        }
        setStatus(roi.subjectDetected ? 'ACTIVE \u2014 RELATIVE INDEX ' + roi.relativeIndex.toFixed(2) : 'SCANNING \u2014 ALIGN SUBJECT');
      }

      /* 12. FPS + latency */
      frameCount++;
      var now2 = performance.now();
      if (now2 - lastFpsTime >= 1000) {
        var fps = Math.round(frameCount * 1000 / (now2 - lastFpsTime));
        if (fpsEl) fpsEl.textContent = fps;
        frameCount = 0;
        lastFpsTime = now2;
      }
      var latency = Math.round(performance.now() - loopStart);
      if (latencyEl) latencyEl.textContent = latency;
    }
    animFrame = requestAnimationFrame(detectionLoop);
  }

  /* ===================== CAMERA CONTROLS ===================== */

  function startCamera() {
    if (cameraStream) return;
    setStatus('REQUESTING CAMERA PERMISSION');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('CAMERA API UNAVAILABLE \u2014 USE HTTPS OR A SUPPORTED BROWSER');
      return;
    }
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    }).then(function (stream) {
      cameraStream = stream;
      video.srcObject = stream;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      if (placeholder) placeholder.style.display = 'none';
      sessionStart = new Date();
      temperatureHistory = [];
      setStatus('LIVE CAMERA ACTIVE \u2014 SOFTWARE THERMAL PROXY');
      video.addEventListener('loadeddata', function onLoaded() {
        video.removeEventListener('loadeddata', onLoaded);
        rgbCanvas.width = video.videoWidth;
        rgbCanvas.height = video.videoHeight;
        thermCanvas.width = video.videoWidth;
        thermCanvas.height = video.videoHeight;
        rgbCtx = rgbCanvas.getContext('2d');
        thermCtx = thermCanvas.getContext('2d');
        detectionLoop();
      });
    }).catch(function (err) {
      console.error('Camera access failed:', err);
      if (err && err.name === 'NotAllowedError') {
        setStatus('CAMERA PERMISSION DENIED \u2014 ALLOW ACCESS AND RETRY');
      } else if (err && err.name === 'NotFoundError') {
        setStatus('NO CAMERA DETECTED \u2014 CONNECT A WEBCAM');
      } else {
        setStatus('CAMERA UNAVAILABLE \u2014 CHECK PERMISSION OR HTTPS');
      }
      cameraStream = null;
    });
  }

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(function (t) { t.stop(); });
    }
    cameraStream = null;
    video.srcObject = null;
    video.pause();
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (placeholder) placeholder.style.display = '';
    temperatureHistory = [];
    chartHistory = [];
    setStatus('CAMERA STOPPED');
    updateReport(null);
  }

  /* ===================== CSV EXPORT ===================== */

  function exportDemoReport() {
    var lines = [
      'NephroScan AI — Software Thermal Proxy Session',
      '==============================================',
      'Session: ' + (sessionStart ? sessionStart.toLocaleString() : 'N/A') + ' to ' + new Date().toLocaleString(),
      'Total readings: ' + sessionLog.length,
      '',
      'Time,ID,Brightness,Relative Index,Smoothed,Trend,Status'
    ];
    sessionLog.forEach(function (e) {
      lines.push([e.time, e.id, e.brightness, e.relativeIndex, e.smoothed, e.trend, e.status].join(','));
    });
    lines.push('');
    lines.push('Labels: SOFTWARE THERMAL PROXY, RELATIVE INDEX, NOT AN INFRARED MEASUREMENT');
    lines.push('Disclosure: Calculated from ordinary webcam brightness; not an infrared or medical temperature reading.');
    lines.push('Disclaimer: Educational prototype only. Not a medical device.');
    var blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'nephroscan-thermal-' + Date.now() + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /* ===================== EVENT BINDINGS ===================== */

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  if (clearBtn) clearBtn.addEventListener('click', function () { resetLog(); });
  window.addEventListener('beforeunload', stopCamera);

  window.NephroScanPresence = {
    start: startCamera,
    stop: stopCamera,
    resetLog: function () { resetLog(); },
    exportReport: exportDemoReport,
    getStatus: function () { return statusEl ? statusEl.textContent : ''; },
    getSessionLog: function () { return sessionLog.slice(); }
  };

})();
