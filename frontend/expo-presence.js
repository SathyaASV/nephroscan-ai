/**
 * NephroScan AI — Software Thermal Proxy Dashboard
 *
 * Browser-local camera pipeline with Inferno-style thermal colormap,
 * temporal smoothing, rolling chart, ROI reticle, and tracking table.
 * Educational prototype only. Not a medical device.
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

  if (!startBtn || !stopBtn || !video || !rgbCanvas || !thermCanvas) {
    console.warn('Expo presence: required DOM elements not found — module disabled.');
    return;
  }

  /* ===================== INFERNO COLORMAP ===================== */
  var INFERNO = [
    [0,0,4],[28,16,68],[76,12,107],[120,28,109],[162,44,96],
    [194,69,76],[222,108,46],[243,155,10],[247,209,58],[252,255,164]
  ];

  function infernoColor(t) {
    t = Math.max(0, Math.min(1, t));
    var idx = t * (INFERNO.length - 1);
    var lo = Math.floor(idx);
    var hi = Math.min(lo + 1, INFERNO.length - 1);
    var f = idx - lo;
    return [
      Math.round(INFERNO[lo][0] + (INFERNO[hi][0] - INFERNO[lo][0]) * f),
      Math.round(INFERNO[lo][1] + (INFERNO[hi][1] - INFERNO[lo][1]) * f),
      Math.round(INFERNO[lo][2] + (INFERNO[hi][2] - INFERNO[lo][2]) * f)
    ];
  }

  /* ===================== STATE ===================== */
  var cameraStream    = null;
  var animFrame       = null;
  var rgbCtx          = null;
  var thermCtx        = null;
  var presenceCounter = 1;
  var lastLogTime     = 0;
  var prevFrameData   = null;
  var smoothedIntensity = 0.5;
  var sessionStart    = null;
  var sessionLog      = [];
  var chartHistory    = [];
  var CHART_MAX       = 60;
  var fpsEl           = document.getElementById('thermalFpsVal');
  var latencyEl       = document.getElementById('thermalLatencyVal');
  var frameCount      = 0;
  var lastFpsTime     = performance.now();

  var LOG_THROTTLE_MS = 2000;
  var SMOOTH_ALPHA    = 0.15;
  var MOTION_STEP     = 10;

  /* ===================== HELPERS ===================== */

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = 'STATUS: ' + msg;
  }

  function resetLog() {
    if (logBody) {
      logBody.innerHTML = '<tr><td colspan="5" class="expo-empty-log">No data yet. Start the camera to begin.</td></tr>';
    }
    presenceCounter = 1;
    sessionLog = [];
    chartHistory = [];
    if (indexValEl) indexValEl.textContent = '\u2014';
  }

  /* ===================== THERMAL CANVAS (INFERNO COLORMAP) ===================== */

  function drawThermal(srcCanvas, tgtCanvas) {
    if (!thermCtx) thermCtx = tgtCanvas.getContext('2d');
    thermCtx.drawImage(srcCanvas, 0, 0, tgtCanvas.width, tgtCanvas.height);
    var imageData = thermCtx.getImageData(0, 0, tgtCanvas.width, tgtCanvas.height);
    var d = imageData.data;
    for (var i = 0; i < d.length; i += 4) {
      var lum = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      var c = infernoColor(lum);
      d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255;
    }
    thermCtx.putImageData(imageData, 0, 0);
  }

  /* ===================== ROI RETICLE ===================== */

  function drawReticle(ctx, w, h) {
    var cx = w / 2, cy = h / 2;
    var r = Math.min(w, h) * 0.18;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,111,60,0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(cx - r - 8, cy); ctx.lineTo(cx + r + 8, cy);
    ctx.moveTo(cx, cy - r - 8); ctx.lineTo(cx, cy + r + 8);
    ctx.stroke();
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

  /* ===================== MOTION / INTENSITY ===================== */

  function computeIntensity(frameData) {
    if (!prevFrameData) {
      prevFrameData = new Uint8ClampedArray(frameData);
      return 0.5;
    }
    var diffSum = 0, count = 0;
    for (var i = 0; i < frameData.length; i += MOTION_STEP * 4) {
      var dr = Math.abs(frameData[i] - prevFrameData[i]);
      var dg = Math.abs(frameData[i + 1] - prevFrameData[i + 1]);
      var db = Math.abs(frameData[i + 2] - prevFrameData[i + 2]);
      diffSum += (dr + dg + db) / 3;
      count++;
    }
    for (var j = 0; j < frameData.length; j++) prevFrameData[j] = frameData[j];
    var motion = count > 0 ? Math.min(1, (diffSum / count) / 40) : 0;
    var brightness = 0;
    for (var k = 0; k < frameData.length; k += MOTION_STEP * 4) {
      brightness += (frameData[k] + frameData[k + 1] + frameData[k + 2]) / 3;
    }
    brightness = count > 0 ? brightness / count / 255 : 0.5;
    var combined = motion * 0.6 + brightness * 0.4;
    smoothedIntensity = smoothedIntensity * (1 - SMOOTH_ALPHA) + combined * SMOOTH_ALPHA;
    return smoothedIntensity;
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

      /* Mirrored optical view */
      rgbCtx.save();
      rgbCtx.translate(rgbCanvas.width, 0);
      rgbCtx.scale(-1, 1);
      rgbCtx.drawImage(video, 0, 0, rgbCanvas.width, rgbCanvas.height);
      rgbCtx.restore();

      /* Thermal proxy */
      drawThermal(rgbCanvas, thermCanvas);
      drawReticle(thermCtx, thermCanvas.width, thermCanvas.height);

      /* Intensity calculation */
      var frameData = rgbCtx.getImageData(0, 0, rgbCanvas.width, rgbCanvas.height).data;
      var intensity = computeIntensity(frameData);

      /* Update emulated index display */
      var displayIndex = (intensity * 100).toFixed(1);
      if (indexValEl) indexValEl.textContent = displayIndex;

      /* Rolling chart */
      chartHistory.push(intensity);
      if (chartHistory.length > CHART_MAX) chartHistory.shift();
      drawChart();

      /* Log entry */
      var now = Date.now();
      if (now - lastLogTime > LOG_THROTTLE_MS) {
        lastLogTime = now;
        var trend = chartHistory.length >= 2 ?
          (chartHistory[chartHistory.length - 1] > chartHistory[chartHistory.length - 2] ? 'Rising' :
           chartHistory[chartHistory.length - 1] < chartHistory[chartHistory.length - 2] ? 'Falling' : 'Stable') : '\u2014';
        var status = intensity > 0.6 ? 'Active Region' : intensity > 0.3 ? 'Moderate' : 'Low';
        var ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        var id = 'T-' + String(presenceCounter++).padStart(3, '0');
        sessionLog.push({ time: ts, id: id, index: displayIndex, trend: trend, status: status });
        if (logBody) {
          if (logBody.querySelector('.expo-empty-log')) logBody.innerHTML = '';
          var row = document.createElement('tr');
          row.innerHTML = '<td>' + ts + '</td><td>' + id + '</td><td>' + displayIndex + '</td><td>' + trend + '</td><td>' + status + '</td>';
          logBody.prepend(row);
          while (logBody.rows.length > 25) logBody.deleteRow(-1);
        }
        setStatus('THERMAL PROXY ACTIVE \u2014 EMULATED INDEX ' + displayIndex);
      }

      /* FPS and latency tracking */
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
      setStatus('CAMERA ACTIVE \u2014 SOFTWARE THERMAL PROXY RUNNING');
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
    prevFrameData = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (placeholder) placeholder.style.display = '';
    setStatus('CAMERA STOPPED');
  }

  /* ===================== EXPORT ===================== */

  function exportDemoReport() {
    var lines = [
      'NephroScan AI — Software Thermal Proxy Session',
      '==============================================',
      'Session: ' + (sessionStart ? sessionStart.toLocaleString() : 'N/A') + ' to ' + new Date().toLocaleString(),
      'Total readings: ' + sessionLog.length,
      '',
      'Time,ID,Proxy Index,Trend,Status'
    ];
    sessionLog.forEach(function (e) {
      lines.push([e.time, e.id, e.index, e.trend, e.status].join(','));
    });
    lines.push('', 'Labels: SOFTWARE THERMAL PROXY, EMULATED INDEX, NOT AN INFRARED MEASUREMENT');
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
  if (clearBtn) clearBtn.addEventListener('click', resetLog);
  window.addEventListener('beforeunload', stopCamera);

  window.NephroScanPresence = {
    start: startCamera,
    stop: stopCamera,
    resetLog: resetLog,
    exportReport: exportDemoReport,
    getStatus: function () { return statusEl ? statusEl.textContent : ''; },
    getSessionLog: function () { return sessionLog.slice(); }
  };

})();
