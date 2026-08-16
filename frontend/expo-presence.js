/**
 * NephroScan AI — Expo Human Presence Module
 *
 * Browser-local camera pipeline with canvas thermal proxy, presence detection,
 * Judge Mode overlays, diagnostics, session log, and export.
 * Educational prototype only. Not a medical device.
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

  if (!startBtn || !stopBtn || !video || !rgbCanvas || !thermCanvas) {
    console.warn('Expo presence: required DOM elements not found — module disabled.');
    return;
  }

  /* ===================== STATE ===================== */
  var cameraStream    = null;
  var animFrame       = null;
  var rgbCtx          = null;
  var thermCtx        = null;
  var presenceCounter = 1;
  var lastLogTime     = 0;
  var prevFrameData   = null;
  var sessionStart    = null;
  var sessionLog      = [];

  var LOG_THROTTLE_MS = 2500;
  var MOTION_STEP     = 12;

  /* ===================== HELPERS ===================== */

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = 'STATUS: ' + msg;
  }

  function resetLog() {
    if (logBody) {
      logBody.innerHTML =
        '<tr><td colspan="6" class="expo-empty-log">No presence events yet.</td></tr>';
    }
    presenceCounter = 1;
    sessionLog = [];
  }

  function addPresenceLogEntry(confidence, tempLabel, channel) {
    var now = new Date();
    var ts = now.toLocaleTimeString('en-US', { hour12: false });
    var id = 'P-' + String(presenceCounter++).padStart(3, '0');
    var row = document.createElement('tr');
    row.innerHTML =
      '<td>' + ts + '</td>' +
      '<td>' + id + '</td>' +
      '<td><span style="color:#168a5b">MOTION DETECTED</span></td>' +
      '<td>' + confidence + '%</td>' +
      '<td>' + tempLabel + '</td>' +
      '<td>' + channel + '</td>';
    if (logBody) {
      if (logBody.querySelector('.expo-empty-log')) logBody.innerHTML = '';
      logBody.prepend(row);
      while (logBody.rows.length > 20) logBody.deleteRow(-1);
    }
    sessionLog.push({ time: ts, id: id, confidence: confidence, temp: tempLabel, channel: channel });
  }

  /* ===================== THERMAL PROXY ===================== */

  function drawThermalProxy(srcCanvas, tgtCanvas) {
    if (!thermCtx) thermCtx = tgtCanvas.getContext('2d');
    thermCtx.drawImage(srcCanvas, 0, 0, tgtCanvas.width, tgtCanvas.height);
    var imageData = thermCtx.getImageData(0, 0, tgtCanvas.width, tgtCanvas.height);
    var d = imageData.data;
    for (var i = 0; i < d.length; i += 4) {
      var lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      var n = lum / 255;
      d[i]     = Math.min(255, Math.floor(n * 320));
      d[i + 1] = Math.max(0, Math.floor((1 - n) * 200));
      d[i + 2] = Math.floor((1 - n) * 180);
      d[i + 3] = 255;
    }
    thermCtx.putImageData(imageData, 0, 0);
  }

  /* ===================== MOTION / PRESENCE HEURISTIC ===================== */

  function computeMotion(frameData, width, height) {
    if (!prevFrameData) {
      prevFrameData = new Uint8ClampedArray(frameData);
      return 0;
    }
    var diffSum = 0;
    var count = 0;
    for (var i = 0; i < frameData.length; i += MOTION_STEP * 4) {
      var dr = Math.abs(frameData[i] - prevFrameData[i]);
      var dg = Math.abs(frameData[i + 1] - prevFrameData[i + 1]);
      var db = Math.abs(frameData[i + 2] - prevFrameData[i + 2]);
      diffSum += (dr + dg + db) / 3;
      count++;
    }
    for (var j = 0; j < frameData.length; j++) prevFrameData[j] = frameData[j];
    if (count === 0) return 0;
    var avgDiff = diffSum / count;
    return Math.min(100, Math.round(avgDiff * 2.5));
  }

  /* ===================== JUDGE MODE (bounding box overlay) ===================== */

  function drawJudgeOverlay(ctx, w, h, motionPct) {
    if (motionPct < 15) return;
    var cx = w / 2, cy = h / 2;
    var bw = w * 0.45, bh = h * 0.65;
    var bx = cx - bw / 2, by = cy - bh / 2;
    ctx.save();
    ctx.strokeStyle = '#ff9900';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(bx, by, bw, bh);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0,0,0,0.72)';
    ctx.fillRect(bx, by - 22, 260, 18);
    ctx.fillStyle = '#ff9900';
    ctx.font = 'bold 10px JetBrains Mono, monospace';
    ctx.fillText('JUDGE MODE: MOTION DETECTED — ' + motionPct + '%', bx + 6, by - 8);
    ctx.restore();
  }

  /* ===================== DETECTION LOOP ===================== */

  function detectionLoop() {
    if (!cameraStream) return;
    if (video.readyState >= 2) {
      if (!rgbCtx) {
        rgbCanvas.width = video.videoWidth;
        rgbCanvas.height = video.videoHeight;
        thermCanvas.width = video.videoWidth;
        thermCanvas.height = video.videoHeight;
        rgbCtx = rgbCanvas.getContext('2d');
        thermCtx = thermCanvas.getContext('2d');
      }

      rgbCtx.drawImage(video, 0, 0, rgbCanvas.width, rgbCanvas.height);
      drawThermalProxy(rgbCanvas, thermCanvas);

      var frameData = rgbCtx.getImageData(0, 0, rgbCanvas.width, rgbCanvas.height).data;
      var motionPct = computeMotion(frameData, rgbCanvas.width, rgbCanvas.height);

      drawJudgeOverlay(rgbCtx, rgbCanvas.width, rgbCanvas.height, motionPct);

      var now = Date.now();
      if (motionPct > 12 && now - lastLogTime > LOG_THROTTLE_MS) {
        lastLogTime = now;
        var tempVal = (35.2 + Math.random() * 1.6).toFixed(1);
        addPresenceLogEntry(Math.min(98, 40 + motionPct), 'Software Thermal Proxy — Not an Infrared Measurement', 'OPTICAL SIM');
        setStatus('MOTION / PRESENCE DETECTED \u2014 CONFIDENCE ' + Math.min(98, 40 + motionPct) + '%');
      } else if (motionPct <= 12) {
        setStatus('SCANNING \u2014 NO PRESENCE DETECTED');
      }
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
      setStatus('CAMERA ACTIVE \u2014 LOCAL HUMAN-PRESENCE DEMO');
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

  /* ===================== EXPORT DEMO REPORT ===================== */

  function exportDemoReport() {
    var lines = [
      'NephroScan AI — Expo Presence Session Report',
      '=============================================',
      'Session start: ' + (sessionStart ? sessionStart.toLocaleString() : 'N/A'),
      'Session end:   ' + new Date().toLocaleString(),
      'Total events:  ' + sessionLog.length,
      '',
      'Time,Tracking ID,Status,Confidence,Temperature,Thermal Channel'
    ];
    sessionLog.forEach(function (e) {
      lines.push([e.time, e.id, 'MOTION DETECTED', e.confidence + '%', e.temp, e.channel].join(','));
    });
    lines.push('', 'Disclaimer: Educational prototype only. Not a medical device.');
    lines.push('Thermal values are software-generated proxies, not infrared measurements.');

    var blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'nephroscan-presence-' + Date.now() + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  /* ===================== DIAGNOSTICS ===================== */

  function runDiagnostics() {
    var info = { camera: false, secure: location.protocol === 'https:' || location.hostname === 'localhost', userAgent: navigator.userAgent };
    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then(function (devices) {
        info.devices = devices.filter(function (d) { return d.kind === 'videoinput'; }).map(function (d) { return d.label || d.deviceId; });
        info.camera = info.devices.length > 0;
        console.log('[NephroScan Diagnostics]', info);
        setStatus('DIAGNOSTICS: ' + (info.camera ? info.devices.length + ' camera(s) found' : 'NO CAMERA') + ' | ' + (info.secure ? 'SECURE' : 'INSECURE'));
      });
    } else {
      console.log('[NephroScan Diagnostics]', info);
      setStatus('DIAGNOSTICS: enumerateDevices not available');
    }
  }

  /* ===================== EVENT BINDINGS ===================== */

  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  if (clearBtn) clearBtn.addEventListener('click', resetLog);
  window.addEventListener('beforeunload', stopCamera);

  /* Expose for external callers (export button, diagnostics, etc.) */
  window.NephroScanPresence = {
    start: startCamera,
    stop: stopCamera,
    resetLog: resetLog,
    exportReport: exportDemoReport,
    diagnostics: runDiagnostics,
    getStatus: function () { return statusEl ? statusEl.textContent : ''; },
    getSessionLog: function () { return sessionLog.slice(); }
  };

})();
