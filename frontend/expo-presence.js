/**
 * NephroScan AI — Software Thermal Proxy Dashboard
 * Motion-tracking thermal visualization ported from the supplied Python algorithm.
 * Uses navigator.mediaDevices.getUserMedia() — no Python, no OpenCV, no server.
 * Educational prototype only. Not an infrared measurement.
 */
(function () {
  'use strict';

  /* ===================== DOM REFS ===================== */
  var startBtn    = document.getElementById('presenceStartBtn');
  var stopBtn     = document.getElementById('presenceStopBtn');
  var clearBtn    = document.getElementById('presenceClearBtn');
  var video       = document.getElementById('presenceVideo');
  var rgbCanvas   = document.getElementById('presenceRgbCanvas');
  var thermCanvas = document.getElementById('presenceThermalCanvas');
  var placeholder = document.getElementById('presencePlaceholder');
  var statusEl    = document.getElementById('presenceStatus');
  var logBody     = document.getElementById('presenceLogBody');
  var indexValEl  = document.getElementById('thermalIndexVal');
  var chartCanvas = document.getElementById('thermalChartCanvas');
  var fpsEl       = document.getElementById('thermalFpsVal');
  var latencyEl   = document.getElementById('thermalLatencyVal');

  /* Live report spans — must match index.html IDs exactly */
  var reportProxyEl = document.getElementById('reportProxyIndex');
  var reportLumaEl  = document.getElementById('reportLuma');
  var reportStatEl  = document.getElementById('reportStatus');
  var reportMotEl   = document.getElementById('reportMotion');
  var reportSampEl  = document.getElementById('reportSamples');
  var reportUpEl    = document.getElementById('reportUptime');

  if (!startBtn || !stopBtn || !video || !rgbCanvas || !thermCanvas) {
    console.warn('Expo presence: required DOM elements not found.');
    return;
  }

  var APP_VERSION = (typeof window.APP_VERSION === 'string') ? window.APP_VERSION : '2.1.0';

  /* ===================== INFERNO COLORMAP LUT ===================== */
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

  /* ===================== MOTION TRACKING STATE ===================== */
  var prevGrayData = null;
  var lastX = 200, lastY = 110, lastW = 240, lastH = 260;
  var startTime      = 0;
  var bootComplete   = false;

  /* ===================== 15-SAMPLE HISTORY ===================== */
  var readingHistory = [];
  var HIST_MAX       = 15;
  var THRESHOLD      = 35.0;

  /* ===================== GRAYSCALE HELPERS ===================== */
  function toGrayscale(rgbData, w, h) {
    var gray = new Uint8Array(w * h);
    for (var i = 0; i < gray.length; i++) {
      var j = i * 4;
      gray[i] = (0.299 * rgbData[j] + 0.587 * rgbData[j + 1] + 0.114 * rgbData[j + 2]) | 0;
    }
    return gray;
  }

  function boxBlur21(src, w, h) {
    var out = new Uint8Array(w * h);
    var r = 10;
    for (var y = 0; y < h; y++) {
      var yTop = Math.max(0, y - r);
      var yBot = Math.min(h - 1, y + r);
      for (var x = 0; x < w; x++) {
        var xLeft = Math.max(0, x - r);
        var xRight = Math.min(w - 1, x + r);
        var sum = 0, cnt = 0;
        for (var yy = yTop; yy <= yBot; yy++) {
          var rowOff = yy * w;
          for (var xx = xLeft; xx <= xRight; xx++) {
            sum += src[rowOff + xx];
            cnt++;
          }
        }
        out[y * w + x] = (sum / cnt + 0.5) | 0;
      }
    }
    return out;
  }

  function absDiff(a, b) {
    var out = new Uint8Array(a.length);
    for (var i = 0; i < a.length; i++) {
      var d = a[i] - b[i];
      out[i] = d < 0 ? -d : d;
    }
    return out;
  }

  function thresholdBinary(src, t) {
    var out = new Uint8Array(src.length);
    for (var i = 0; i < src.length; i++) {
      out[i] = src[i] > t ? 255 : 0;
    }
    return out;
  }

  function dilate(src, w, h, iters) {
    var out = new Uint8Array(src);
    for (var it = 0; it < iters; it++) {
      var tmp = new Uint8Array(w * h);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          var off = y * w + x;
          if (out[off]) {
            tmp[off] = 255;
            if (x > 0) tmp[off - 1] = 255;
            if (x < w - 1) tmp[off + 1] = 255;
            if (y > 0) tmp[off - w] = 255;
            if (y < h - 1) tmp[off + w] = 255;
          }
        }
      }
      out = tmp;
    }
    return out;
  }

  function findMotionBBox(mask, w, h) {
    var minX = w, minY = h, maxX = 0, maxY = 0, count = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (mask[y * w + x]) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
          count++;
        }
      }
    }
    if (count < 2000) return null;
    return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
  }

  /* ===================== INFERNO COLORMAP ON CANVAS ===================== */
  function applyInfernoGray(grayData, ctx, w, h) {
    var imageData = ctx.createImageData(w, h);
    var d = imageData.data;
    for (var i = 0; i < grayData.length; i++) {
      var inverted = 255 - grayData[i];
      var lutIdx = inverted * 3;
      var j = i * 4;
      d[j]     = INFERNO_LUT[lutIdx];
      d[j + 1] = INFERNO_LUT[lutIdx + 1];
      d[j + 2] = INFERNO_LUT[lutIdx + 2];
      d[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  /* ===================== CANVAS DRAWING ===================== */
  function drawTrackingBox(ctx, x, y, w, h, isAlarm) {
    ctx.save();
    ctx.strokeStyle = isAlarm ? 'rgba(255,0,0,0.9)' : 'rgba(0,255,255,0.85)';
    ctx.lineWidth = isAlarm ? 3 : 2;
    ctx.strokeRect(x, y, w, h);
    if (isAlarm) { ctx.shadowColor = 'rgba(255,0,0,0.4)'; ctx.shadowBlur = 6; ctx.strokeRect(x, y, w, h); }
    var cX = (x + w / 2) | 0, cY = (y + h / 2) | 0;
    ctx.strokeStyle = 'rgba(255,0,0,0.8)'; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cX - 15, cY); ctx.lineTo(cX + 15, cY);
    ctx.moveTo(cX, cY - 15); ctx.lineTo(cX, cY + 15);
    ctx.stroke();
    ctx.restore();
  }

  function drawHUD(ctx, w, h, statusText, statusColor, elapsed, avgBrightness) {
    ctx.save();
    ctx.fillStyle = 'rgba(25,25,25,0.85)';
    ctx.fillRect(15, 20, 245, 95);
    ctx.strokeStyle = 'rgba(50,50,50,0.8)';
    ctx.strokeRect(15, 20, 245, 95);
    ctx.font = '11px monospace';
    ctx.fillStyle = statusColor;
    ctx.fillText('HUD: ' + statusText, 25, 40);
    ctx.fillStyle = '#b4b4b4';
    ctx.fillText('UPTIME: ' + elapsed + 's', 25, 58);
    ctx.fillText('RAW LUMA: ' + avgBrightness.toFixed(1), 25, 76);
    ctx.fillStyle = '#00ff00';
    ctx.fillText('DATA INTEGRATION PIPE: OK', 25, 94);
    ctx.restore();
  }

  function drawBottomBar(ctx, w, h) {
    ctx.save();
    ctx.fillStyle = 'rgba(15,15,15,0.9)';
    ctx.fillRect(0, h - 24, w, 24);
    ctx.font = '11px monospace';
    ctx.fillStyle = '#969696';
    ctx.fillText('SOFTWARE THERMAL PROXY v' + APP_VERSION + ' // MOTION/PRESENCE VISUALIZATION', 15, h - 8);
    ctx.restore();
  }

  function drawBootBar(ctx, w, h, elapsed) {
    ctx.save();
    var barW = ((elapsed / 3.0) * (w - 200)) | 0;
    var midY = (h / 2) | 0;
    ctx.fillStyle = 'rgba(0,255,255,0.6)';
    ctx.fillRect(100, midY - 10, barW, 20);
    ctx.strokeStyle = '#ffffff';
    ctx.strokeRect(100, midY - 10, w - 100, 20);
    ctx.restore();
  }

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
      var px = i * step;
      var py = H - chartHistory[i] * (H - 8) - 4;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.lineTo((chartHistory.length - 1) * step, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,111,60,0.1)';
    ctx.fill();
  }

  /* ===================== HELPERS ===================== */
  function setStatus(msg) {
    if (statusEl) statusEl.textContent = 'STATUS: ' + msg;
  }

  function resetState() {
    prevGrayData = null;
    lastX = 200; lastY = 110; lastW = 240; lastH = 260;
    bootComplete = false;
    readingHistory = [];
    chartHistory = [];
    presenceCounter = 1;
    sessionLog = [];
    lastLogTime = 0;
    frameCount = 0;
    lastFpsTime = performance.now();
    if (indexValEl) indexValEl.textContent = '\u2014';
    updateReportFields('\u2014', '\u2014', '\u2014', '\u2014', '\u2014 / 15', '\u2014');
    if (logBody) {
      logBody.innerHTML = '<tr><td colspan="5" class="expo-empty-log">No data yet. Start the camera to begin.</td></tr>';
    }
  }

  /* ===================== UPDATE REPORT FIELDS — textContent every frame ===================== */
  function updateReportFields(proxyVal, lumaVal, statusVal, motionVal, samplesVal, uptimeVal) {
    if (reportProxyEl) reportProxyEl.textContent = proxyVal;
    if (reportLumaEl)  reportLumaEl.textContent  = lumaVal;
    if (reportStatEl)  reportStatEl.textContent   = statusVal;
    if (reportMotEl)   reportMotEl.textContent    = motionVal;
    if (reportSampEl)  reportSampEl.textContent   = samplesVal;
    if (reportUpEl)    reportUpEl.textContent     = uptimeVal;
  }

  /* ===================== DETECTION LOOP ===================== */
  function detectionLoop() {
    if (!cameraStream) return;
    var loopStart = performance.now();

    if (video.readyState >= 2) {
      var w = video.videoWidth;
      var h = video.videoHeight;

      if (!rgbCtx) {
        rgbCanvas.width = w;  rgbCanvas.height = h;
        thermCanvas.width = w; thermCanvas.height = h;
        rgbCtx = rgbCanvas.getContext('2d');
        thermCtx = thermCanvas.getContext('2d');
      }

      /* 1. Mirrored frame */
      rgbCtx.save();
      rgbCtx.translate(w, 0);
      rgbCtx.scale(-1, 1);
      rgbCtx.drawImage(video, 0, 0, w, h);
      rgbCtx.restore();

      /* 2. Grayscale */
      var rgbData = rgbCtx.getImageData(0, 0, w, h).data;
      var grayData = toGrayscale(rgbData, w, h);

      /* 3. Apply Inferno colormap */
      applyInfernoGray(grayData, thermCtx, w, h);

      /* 4. Motion tracking pipeline */
      var elapsed = (performance.now() - startTime) / 1000;
      var avgBrightness = 128.0;
      var liveCalc = 0;
      var smoothed = 0;
      var motionDetected = false;
      var targetMoving = false;
      var statusText, statusColor, isAlarm = false;

      if (elapsed < 3.0) {
        statusText = 'COMPUTING MOTION DELTA MAPS (' + ((3.0 - elapsed) | 0) + 's)...';
        statusColor = '#00a5ff';
        drawBootBar(thermCtx, w, h, elapsed);
      } else {
        if (!bootComplete) { prevGrayData = null; bootComplete = true; }

        var currentBlur = boxBlur21(grayData, w, h);

        if (prevGrayData) {
          var frameDelta = absDiff(prevGrayData, currentBlur);
          var motionMask = dilate(thresholdBinary(frameDelta, 15), w, h, 2);
          var bbox = findMotionBBox(motionMask, w, h);

          if (bbox) {
            lastX = (lastX * 0.8 + bbox.x * 0.2) | 0;
            lastY = (lastY * 0.8 + bbox.y * 0.2) | 0;
            lastW = (lastW * 0.8 + bbox.w * 0.2) | 0;
            lastH = (lastH * 0.8 + bbox.h * 0.2) | 0;
            if (lastW < 180) lastW = 200;
            if (lastH < 180) lastH = 220;
            motionDetected = true;
            targetMoving = true;
          }
        }

        prevGrayData = currentBlur;

        /* Clamp box to screen */
        var bx = Math.max(0, Math.min(w - 50, lastX));
        var by = Math.max(0, Math.min(h - 50, lastY));
        var bw = Math.min(w - bx, lastW);
        var bh = Math.min(h - by, lastH);

        /* 5. ROI brightness */
        var sum = 0, cnt = 0;
        for (var row = by; row < by + bh && row < h; row++) {
          var rowOff = row * w;
          for (var col = bx; col < bx + bw && col < w; col++) {
            sum += grayData[rowOff + col];
            cnt++;
          }
        }
        avgBrightness = cnt > 0 ? sum / cnt : 128.0;

        /* 6. SUPPLIED FORMULA: live_calc = 15.0 + (avg_brightness / 255.0) * 30.0 */
        liveCalc = 15.0 + (avgBrightness / 255.0) * 30.0;

        readingHistory.push(liveCalc);
        if (readingHistory.length > HIST_MAX) readingHistory.shift();
        smoothed = readingHistory.reduce(function (a, b) { return a + b; }, 0) / readingHistory.length;
        smoothed = Math.round(smoothed * 10) / 10;

        /* 7. Threshold — supplied value 35.0 */
        if (smoothed >= THRESHOLD) {
          statusText = 'ALARM: HIGH HEAT PROFILE';
          statusColor = '#ff0000';
          isAlarm = true;
        } else {
          statusText = 'MOTION LOCKED // LIVE STREAM';
          statusColor = '#00ff00';
        }

        /* 8. Draw tracking box + crosshair */
        drawTrackingBox(thermCtx, bx, by, bw, bh, isAlarm);

        /* 9. HUD + bottom bar */
        drawHUD(thermCtx, w, h, statusText, statusColor, elapsed.toFixed(1), avgBrightness);
        drawBottomBar(thermCtx, w, h);

        /* 10. HUD text on tracking box */
        thermCtx.save();
        thermCtx.font = '12px monospace';
        thermCtx.fillStyle = '#00ff00';
        thermCtx.fillText('MOTION_LOCK', bx, by - 26);
        thermCtx.fillStyle = '#00ffff';
        thermCtx.fillText('EMULATED INDEX: ' + smoothed.toFixed(1), bx, by - 8);
        thermCtx.restore();
      }

      /* ===================== UPDATE ALL DISPLAYS EVERY FRAME ===================== */

      /* Big index card */
      if (indexValEl) indexValEl.textContent = smoothed > 0 ? smoothed.toFixed(1) : '\u2014';

      /* Live report card — textContent, every frame */
      updateReportFields(
        smoothed > 0 ? smoothed.toFixed(1) : '\u2014',
        avgBrightness.toFixed(1),
        statusText || 'BOOTING...',
        targetMoving ? 'MOTION LOCKED' : 'SEARCHING',
        readingHistory.length + ' / 15',
        elapsed.toFixed(1) + 's'
      );

      /* Rolling chart */
      chartHistory.push(smoothed);
      if (chartHistory.length > CHART_MAX) chartHistory.shift();
      drawChart();

      /* Throttled log entry */
      var now = Date.now();
      if (bootComplete && now - lastLogTime > LOG_THROTTLE_MS) {
        lastLogTime = now;
        var trend = chartHistory.length >= 2
          ? (chartHistory[chartHistory.length - 1] > chartHistory[chartHistory.length - 2] ? 'Rising' :
             chartHistory[chartHistory.length - 1] < chartHistory[chartHistory.length - 2] ? 'Falling' : 'Stable')
          : '\u2014';
        var ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        var id = 'T-' + String(presenceCounter++).padStart(3, '0');
        sessionLog.push({ time: ts, id: id, smoothed: smoothed, trend: trend, status: statusText || '' });
        if (logBody) {
          if (logBody.querySelector('.expo-empty-log')) logBody.innerHTML = '';
          var row = document.createElement('tr');
          row.innerHTML = '<td>' + ts + '</td><td>' + id + '</td><td>' + smoothed.toFixed(1) + '</td><td>' + trend + '</td><td>' + (statusText || '') + '</td>';
          logBody.prepend(row);
          while (logBody.rows.length > 25) logBody.deleteRow(-1);
        }
        setStatus(motionDetected ? 'ACTIVE \u2014 EMULATED INDEX ' + smoothed.toFixed(1) : 'SCANNING \u2014 AWAITING MOTION');
      }

      /* FPS + latency */
      frameCount++;
      var now2 = performance.now();
      if (now2 - lastFpsTime >= 1000) {
        if (fpsEl) fpsEl.textContent = Math.round(frameCount * 1000 / (now2 - lastFpsTime));
        frameCount = 0;
        lastFpsTime = now2;
      }
      if (latencyEl) latencyEl.textContent = Math.round(performance.now() - loopStart);
    }

    animFrame = requestAnimationFrame(detectionLoop);
  }

  /* ===================== CAMERA CONTROLS ===================== */

  function startCamera() {
    if (cameraStream) return;
    setStatus('REQUESTING CAMERA PERMISSION');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('CAMERA API UNAVAILABLE \u2014 USE HTTPS');
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
      resetState();
      startTime = performance.now();
      sessionStart = new Date();
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
      if (err && err.name === 'NotAllowedError') setStatus('CAMERA PERMISSION DENIED');
      else if (err && err.name === 'NotFoundError') setStatus('NO CAMERA DETECTED');
      else setStatus('CAMERA UNAVAILABLE');
      cameraStream = null;
    });
  }

  function stopCamera() {
    if (cameraStream) cameraStream.getTracks().forEach(function (t) { t.stop(); });
    cameraStream = null;
    video.srcObject = null;
    video.pause();
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    prevGrayData = null;
    readingHistory = [];
    chartHistory = [];
    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (placeholder) placeholder.style.display = '';
    setStatus('CAMERA STOPPED');
    updateReportFields('\u2014', '\u2014', '\u2014', '\u2014', '\u2014 / 15', '\u2014');
    if (indexValEl) indexValEl.textContent = '\u2014';
  }

  /* ===================== CSV EXPORT ===================== */
  function exportDemoReport() {
    var lines = [
      'NephroScan AI \u2014 Software Thermal Proxy Session',
      'Session: ' + (sessionStart ? sessionStart.toLocaleString() : 'N/A') + ' to ' + new Date().toLocaleString(),
      'Total readings: ' + sessionLog.length,
      '', 'Time,ID,Smoothed,Trend,Status'
    ];
    sessionLog.forEach(function (e) { lines.push([e.time, e.id, e.smoothed, e.trend, e.status].join(',')); });
    lines.push('');
    lines.push('Disclosure: Calculated from ordinary webcam brightness; not an infrared or medical temperature reading.');
    lines.push('Disclaimer: Educational prototype only. Not a medical device.');
    var blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a'); a.href = url;
    a.download = 'nephroscan-thermal-' + Date.now() + '.csv';
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /* ===================== EVENT BINDINGS ===================== */
  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  if (clearBtn) clearBtn.addEventListener('click', function () { resetState(); });
  window.addEventListener('beforeunload', stopCamera);

  window.NephroScanPresence = {
    start: startCamera,
    stop: stopCamera,
    resetLog: function () { resetState(); },
    exportReport: exportDemoReport,
    getStatus: function () { return statusEl ? statusEl.textContent : ''; },
    getSessionLog: function () { return sessionLog.slice(); }
  };

})();
