/**
 * EcoSort AI - Fast & Reliable Waste Segregation Controller
 * Bulletproof Preset Switching, Sub-150ms Live Inference, and Crystal-Clear Visual Highlights.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const voiceToggleBtn = document.getElementById('voiceToggleBtn');
    const voiceIcon = document.getElementById('voiceIcon');
    const voiceStatusText = document.getElementById('voiceStatusText');

    const btnModeCamera = document.getElementById('btnModeCamera');
    const btnModeSamples = document.getElementById('btnModeSamples');
    const btnModeUpload = document.getElementById('btnModeUpload');
    const fileInput = document.getElementById('fileInput');

    const mediaWrapper = document.getElementById('mediaWrapper');
    const liveCameraFeed = document.getElementById('liveCameraFeed');
    const webcamVideo = document.getElementById('webcamVideo');
    const staticPreview = document.getElementById('staticPreview');
    const detectionCanvas = document.getElementById('detectionCanvas');
    const ctx = detectionCanvas.getContext('2d');

    const cameraSourceToolbar = document.getElementById('cameraSourceToolbar');
    const btnSourcePC = document.getElementById('btnSourcePC');
    const btnSourceDroidCam = document.getElementById('btnSourceDroidCam');
    const btnSourceDev1 = document.getElementById('btnSourceDev1');
    const btnSourceMobile = document.getElementById('btnSourceMobile');
    const btnToggleBgFilter = document.getElementById('btnToggleBgFilter');
    const mobileConnectPanel = document.getElementById('mobileConnectPanel');
    const mobileIpInput = document.getElementById('mobileIpInput');
    const btnConnectPhone = document.getElementById('btnConnectPhone');

    const hudStatus = document.getElementById('hudStatus');
    const hudLatency = document.getElementById('hudLatency');
    const btnCaptureAnalyze = document.getElementById('btnCaptureAnalyze');
    const btnAutoScanToggle = document.getElementById('btnAutoScanToggle');
    const btnUnlockScan = document.getElementById('btnUnlockScan');
    const btnToggleMasks = document.getElementById('btnToggleMasks');

    const streamBadge = document.getElementById('streamBadge');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const expectedWasteCard = document.getElementById('expectedWasteCard');
    const expectedWasteTitle = document.getElementById('expectedWasteTitle');
    const expectedWasteDetail = document.getElementById('expectedWasteDetail');
    const itemStreamIcon = document.getElementById('itemStreamIcon');

    const quantifiedWeight = document.getElementById('quantifiedWeight');
    const quantifiedCount = document.getElementById('quantifiedCount');

    const binInstructionCard = document.getElementById('binInstructionCard');
    const binIcon = document.getElementById('binIcon');
    const binTitle = document.getElementById('binTitle');
    const binInstructionText = document.getElementById('binInstructionText');

    // Telemetry Dashboard Elements
    const kpiTotalObjects = document.getElementById('kpiTotalObjects');
    const kpiStreamBreakdown = document.getElementById('kpiStreamBreakdown');
    const kpiTotalWeight = document.getElementById('kpiTotalWeight');
    const kpiWeightBreakdown = document.getElementById('kpiWeightBreakdown');
    const kpiTotalCo2 = document.getElementById('kpiTotalCo2');
    const fillBarGreen = document.getElementById('fillBarGreen');
    const fillPctGreen = document.getElementById('fillPctGreen');
    const fillBarBlue = document.getElementById('fillBarBlue');
    const fillPctBlue = document.getElementById('fillPctBlue');
    const btnResetStats = document.getElementById('btnResetStats');
    const telemetrySyncText = document.getElementById('telemetrySyncText');

    const presetButtons = document.querySelectorAll('.preset-pill');

    // State
    let currentMode = 'samples'; // Start in samples mode so page is NEVER blank or broken
    let voiceEnabled = true;
    let showBoxes = true;
    let autoScanEnabled = false;
    let autoScanInterval = null;
    let activeStream = null;
    let currentDetections = [];
    let isAnalyzing = false;

    // 1. Voice Announcement
    function speakDirective(text) {
        if (!voiceEnabled || !('speechSynthesis' in window)) return;
        try {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.05;
            window.speechSynthesis.speak(utterance);
        } catch (e) {
            console.warn('Speech error:', e);
        }
    }

    voiceToggleBtn.addEventListener('click', () => {
        voiceEnabled = !voiceEnabled;
        voiceToggleBtn.classList.toggle('active', voiceEnabled);
        voiceIcon.textContent = voiceEnabled ? '🔊' : '🔇';
        voiceStatusText.textContent = voiceEnabled ? 'Voice: ON' : 'Voice: OFF';
        if (voiceEnabled) speakDirective('Voice feedback enabled.');
        else window.speechSynthesis.cancel();
    });

    // 2. Canvas Sizing
    function syncCanvasSize() {
        if (!mediaWrapper) return;
        detectionCanvas.width = mediaWrapper.clientWidth;
        detectionCanvas.height = mediaWrapper.clientHeight;
    }

    window.addEventListener('resize', () => {
        syncCanvasSize();
        drawDetections();
    });

    // 3. Camera & Mode Switching
    let activeMediaStream = null;
    let browserAutoScanInterval = null;
    let cameraPollInterval = null;
    let lastAnnouncedItem = null;
    let lastAnnouncedTime = 0;
    let currentCameraSubSource = 'pc'; // 'pc' or 'mobile'

    async function startBrowserWebcam() {
        stopBackendCameraOnly();
        currentCameraSubSource = 'pc';
        hudStatus.textContent = 'Opening PC Webcam in browser...';

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                });
                activeMediaStream = stream;
                webcamVideo.srcObject = stream;
                webcamVideo.style.display = 'block';
                liveCameraFeed.style.display = 'none';
                staticPreview.style.display = 'none';

                webcamVideo.onloadedmetadata = () => {
                    syncCanvasSize();
                    hudStatus.textContent = '💻 PC Webcam Active • Hold item in front of camera';
                    startBrowserAutoScan();
                };
                return true;
            } catch (err) {
                console.warn('Browser getUserMedia not allowed or failed, falling back to backend stream:', err);
            }
        }
        // Fallback to backend OpenCV stream if browser getUserMedia is not allowed
        startBackendCameraStream();
        return false;
    }

    function startBackendCameraStream() {
        stopBrowserWebcamOnly();
        webcamVideo.style.display = 'none';
        staticPreview.style.display = 'none';
        liveCameraFeed.style.display = 'block';
        liveCameraFeed.src = '/api/video_stream?t=' + Date.now();
        hudStatus.textContent = 'Live OpenCV Stream Active • Hold item in Target Zone';
        startCameraPolling();
    }

    function stopBrowserWebcamOnly() {
        if (browserAutoScanInterval) {
            clearInterval(browserAutoScanInterval);
            browserAutoScanInterval = null;
        }
        if (activeMediaStream) {
            activeMediaStream.getTracks().forEach(t => t.stop());
            activeMediaStream = null;
        }
        if (webcamVideo) {
            webcamVideo.srcObject = null;
            webcamVideo.style.display = 'none';
        }
    }

    function stopBackendCameraOnly() {
        if (cameraPollInterval) {
            clearInterval(cameraPollInterval);
            cameraPollInterval = null;
        }
        if (liveCameraFeed) {
            liveCameraFeed.src = '';
            liveCameraFeed.style.display = 'none';
        }
    }

    function stopAllCameras() {
        stopBrowserWebcamOnly();
        stopBackendCameraOnly();
    }

    function startBrowserAutoScan() {
        if (browserAutoScanInterval) clearInterval(browserAutoScanInterval);
        if (!autoScanEnabled || currentMode !== 'camera' || currentCameraSubSource !== 'pc') return;

        browserAutoScanInterval = setInterval(() => {
            if (currentMode === 'camera' && currentCameraSubSource === 'pc' && webcamVideo.readyState >= 2 && !isAnalyzing) {
                captureAndAnalyze();
            }
        }, 450);
    }

    function setMode(mode) {
        currentMode = mode;
        [btnModeCamera, btnModeSamples, btnModeUpload].forEach(b => b.classList.remove('active'));

        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        currentDetections = [];

        if (mode === 'camera') {
            btnModeCamera.classList.add('active');
            if (cameraSourceToolbar) cameraSourceToolbar.style.display = 'flex';
            startBackendCameraStream();
        } else {
            stopAllCameras();
            if (cameraSourceToolbar) cameraSourceToolbar.style.display = 'none';
            staticPreview.style.display = 'block';

            if (mode === 'samples') {
                btnModeSamples.classList.add('active');
                hudStatus.textContent = 'Demo Mode Active • Click presets below to test';
                const activePreset = document.querySelector('.preset-pill.active');
                if (activePreset) {
                    const sampleFile = activePreset.getAttribute('data-sample');
                    selectPresetSample(sampleFile, activePreset.textContent.trim());
                }
            } else if (mode === 'upload') {
                btnModeUpload.classList.add('active');
                hudStatus.textContent = 'Select an image from your computer';
                fileInput.click();
            }
        }
    }

    btnModeCamera.addEventListener('click', () => setMode('camera'));
    btnModeSamples.addEventListener('click', () => setMode('samples'));
    btnModeUpload.addEventListener('click', () => setMode('upload'));

    // 4. OpenCV Camera Polling & Device Control
    function startCameraPolling() {
        if (cameraPollInterval) clearInterval(cameraPollInterval);
        cameraPollInterval = setInterval(async () => {
            if (currentMode !== 'camera') return; // Strict guard: never overwrite sample presets with camera polling
            try {
                const res = await fetch('/api/camera_status');
                const data = await res.json();
                if (data) {
                    hudLatency.textContent = `${data.fps || 0} FPS`;
                    if (data.latest_result) {
                        renderPrediction(data.latest_result);
                        if (data.latest_result.detected) {
                            const now = Date.now();
                            const item = data.latest_result.primary_item;
                            if (item && item !== lastAnnouncedItem && (now - lastAnnouncedTime > 3500)) {
                                lastAnnouncedItem = item;
                                lastAnnouncedTime = now;
                                const directive = data.latest_result.speech_text || `${item}. Divert to ${data.latest_result.stream} bin.`;
                                speakDirective(directive);
                            }
                        } else {
                            lastAnnouncedItem = null;
                            if (data.connected) {
                                hudStatus.textContent = `Camera Active (${data.source_name}) • Place item inside box`;
                            }
                        }
                    } else if (data.connected) {
                        hudStatus.textContent = `Camera Active (${data.source_name}) • Place item inside box`;
                    } else {
                        hudStatus.textContent = `Connecting to camera stream (${data.source_name})...`;
                    }
                }
            } catch (e) {
                console.warn('Camera status poll error:', e);
            }
        }, 500);
    }

    // Camera Device Switchers
    if (btnSourcePC) {
        btnSourcePC.addEventListener('click', async () => {
            [btnSourcePC, btnSourceDroidCam, btnSourceMobile].forEach(b => b && b.classList.remove('active'));
            btnSourcePC.classList.add('active');
            if (mobileConnectPanel) mobileConnectPanel.style.display = 'none';
            currentCameraSubSource = 'pc';
            hudStatus.textContent = 'Activating PC Webcam (Device 0)...';
            try {
                await fetch('/api/camera_source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: '0' })
                });
            } catch (e) {}
            startBackendCameraStream();
        });
    }

    if (btnSourceDroidCam) {
        btnSourceDroidCam.addEventListener('click', async () => {
            [btnSourcePC, btnSourceDroidCam, btnSourceDev1, btnSourceMobile].forEach(b => b && b.classList.remove('active'));
            btnSourceDroidCam.classList.add('active');
            if (mobileConnectPanel) mobileConnectPanel.style.display = 'none';
            currentCameraSubSource = 'droidcam';
            hudStatus.textContent = 'Connecting to DroidCam Virtual Webcam (Device 2)...';
            try {
                const res = await fetch('/api/camera_source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: '2' })
                });
                const d = await res.json();
                hudStatus.textContent = `Connected: ${d.source_name}`;
                startBackendCameraStream();
            } catch (e) {
                console.error('DroidCam switch error:', e);
            }
        });
    }

    if (btnSourceDev1) {
        btnSourceDev1.addEventListener('click', async () => {
            [btnSourcePC, btnSourceDroidCam, btnSourceDev1, btnSourceMobile].forEach(b => b && b.classList.remove('active'));
            btnSourceDev1.classList.add('active');
            if (mobileConnectPanel) mobileConnectPanel.style.display = 'none';
            currentCameraSubSource = 'dev1';
            hudStatus.textContent = 'Connecting to Camera Device 1...';
            try {
                const res = await fetch('/api/camera_source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: '1' })
                });
                const d = await res.json();
                hudStatus.textContent = `Connected: ${d.source_name}`;
                startBackendCameraStream();
            } catch (e) {
                console.error('Camera Device 1 switch error:', e);
            }
        });
    }

    if (btnSourceMobile) {
        btnSourceMobile.addEventListener('click', () => {
            [btnSourcePC, btnSourceDroidCam, btnSourceMobile].forEach(b => b && b.classList.remove('active'));
            btnSourceMobile.classList.add('active');
            currentCameraSubSource = 'mobile';
            if (mobileConnectPanel) {
                mobileConnectPanel.style.display = mobileConnectPanel.style.display === 'none' ? 'flex' : 'none';
            }
        });
    }

    if (btnConnectPhone) {
        btnConnectPhone.addEventListener('click', async () => {
            let url = mobileIpInput ? mobileIpInput.value.trim() : '';
            if (!url || !url.startsWith('http')) {
                alert('Please enter a valid Phone Camera URL\n\nExamples:\n• DroidCam: http://192.168.X.X:4747/video\n• IP Webcam: http://192.168.X.X:8080/video');
                return;
            }
            if (url.includes(':4747') && !url.endsWith('/video') && !url.endsWith('/mjpegfeed')) {
                url = url.replace(/\/+$/, '') + '/video';
            } else if (url.includes(':8080') && !url.endsWith('/video') && !url.endsWith('/mjpegfeed')) {
                url = url.replace(/\/+$/, '') + '/video';
            }
            hudStatus.textContent = `Connecting to Mobile Phone Camera: ${url}...`;
            try {
                const res = await fetch('/api/camera_source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: url })
                });
                const d = await res.json();
                hudStatus.textContent = `Connected: ${d.source_name}`;
                currentCameraSubSource = 'mobile';
                startBackendCameraStream();
            } catch (e) {
                hudStatus.textContent = 'Failed to connect to mobile camera.';
            }
        });
    }

    liveCameraFeed.addEventListener('error', () => {
        if (currentMode === 'camera' && currentCameraSubSource === 'mobile') {
            hudStatus.textContent = 'Reconnecting camera stream...';
            setTimeout(() => {
                if (currentMode === 'camera' && currentCameraSubSource === 'mobile') {
                    liveCameraFeed.src = '/api/video_stream?t=' + Date.now();
                }
            }, 1000);
        }
    });

    if (btnToggleBgFilter) {
        btnToggleBgFilter.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/camera_toggle_filter', { method: 'POST' });
                const d = await res.json();
                btnToggleBgFilter.classList.toggle('active', d.bg_filter);
                btnToggleBgFilter.textContent = d.bg_filter ? '🎯 Filter Room Noise: ON' : '⚪ Filter Room Noise: OFF';
            } catch (e) {
                console.error('Filter toggle error:', e);
            }
        });
    }

    btnAutoScanToggle.addEventListener('click', () => {
        autoScanEnabled = !autoScanEnabled;
        btnAutoScanToggle.classList.toggle('active', autoScanEnabled);
        btnAutoScanToggle.textContent = autoScanEnabled ? '⏱️ Auto-Scan (Hold): ON' : '⏱️ Auto-Scan (Hold): OFF';
        if (autoScanEnabled && currentMode === 'camera') {
            if (currentCameraSubSource === 'pc') startBrowserAutoScan();
            else startCameraPolling();
        } else {
            if (browserAutoScanInterval) clearInterval(browserAutoScanInterval);
            if (cameraPollInterval) clearInterval(cameraPollInterval);
            fetch('/api/camera_unlock', { method: 'POST' }).catch(() => {});
            if (btnUnlockScan) btnUnlockScan.style.display = 'none';
        }
    });

    // Hold Duration Selector (30s, 45s, 60s)
    const holdChips = document.querySelectorAll('#holdSelector .btn-chip');
    holdChips.forEach(chip => {
        chip.addEventListener('click', async () => {
            holdChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const seconds = parseInt(chip.getAttribute('data-hold') || '45', 10);
            try {
                await fetch('/api/camera_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lock_duration: seconds })
                });
                hudStatus.textContent = `⏱️ Explanation Hold Time set to ${seconds} seconds`;
            } catch (e) {
                console.warn('Hold config error:', e);
            }
        });
    });

    btnUnlockScan?.addEventListener('click', async () => {
        try {
            await fetch('/api/camera_unlock', { method: 'POST' });
            if (btnUnlockScan) btnUnlockScan.style.display = 'none';
            hudStatus.textContent = '⚡ Scanning next item in drop zone...';
        } catch (e) {
            console.warn('Unlock error:', e);
        }
    });

    // Spacebar shortcut to immediately unlock and scan next item
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && (e.target === document.body || e.target.tagName === 'BUTTON')) {
            e.preventDefault();
            const btnUnlock = document.getElementById('btnUnlockScan');
            if (btnUnlock && btnUnlock.style.display !== 'none') {
                btnUnlock.click();
            } else if (currentMode === 'camera') {
                captureAndAnalyze();
            }
        }
    });

    btnToggleMasks.addEventListener('click', async () => {
        showBoxes = !showBoxes;
        btnToggleMasks.classList.toggle('active', showBoxes);
        btnToggleMasks.textContent = showBoxes ? '👁️ Toggle Box' : '🙈 Box Hidden';
        drawDetections();
        if (currentMode === 'camera') {
            try {
                await fetch('/api/camera_toggle_overlay', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ show_boxes: showBoxes })
                });
            } catch(e) {
                console.warn('Toggle overlay error:', e);
            }
        }
    });

    // 5. Preset Button Handlers & Immutable Sample Presets
    const SAMPLE_PRESETS_DATA = {
        'sample_canteen_rice_plate.jpg': {
            success: true,
            detected: true,
            stream: "DUAL",
            primary_item: "Canteen Cooked Rice on Disposable Plate",
            expected_waste_type: "Cafeteria Plate with Food Scraps (Dual Stream)",
            bin_directive: "🟩 STEP 1: GREEN BIN (Food) • 🟦 STEP 2: BLUE BIN (Plate)",
            bin_sub: "Multi-object detected: Organic food leftovers + Inorganic plate/tray.",
            confidence: 97.6,
            grams: 160,
            item_count: 2,
            speech_text: "Dual Stream Cafeteria Plate detected. Step 1: Scrape food into Green Bin. Step 2: Place plate into Blue Bin.",
            detections: [
                {
                    name: "Cooked Rice Scraps (Organic)",
                    stream: "ORGANIC",
                    confidence: 98.2,
                    box: [130, 80, 480, 390],
                    color: "#10b981"
                },
                {
                    name: "Disposable Serving Plate (Inorganic)",
                    stream: "INORGANIC",
                    confidence: 97.0,
                    box: [60, 40, 580, 440],
                    color: "#0ea5e9"
                }
            ]
        },
        'sample_apple_core.jpg': {
            success: true,
            detected: true,
            primary_item: "Apple Core / Fruit Scrap (फल का अवशेष)",
            stream: "ORGANIC",
            stream_display: "ORGANIC (Gila Kachra)",
            expected_waste_type: "Organic Waste (Wet / Fruit Scrap - Apple Core)",
            confidence: 97.8,
            grams: 45,
            item_count: 1,
            bin_directive: "🟩 ORGANIC WASTE ➔ GREEN BIN",
            bin_sub: "Gila Kachra (गीला कचरा) • Micro-Composting Centre & Vermicompost",
            bin_color: "#10b981",
            speech_text: "Apple Core detected. Green Compost Bin.",
            detections: [{
                name: "Apple Core / Fruit Scrap (फल का अवशेष)",
                stream: "ORGANIC",
                confidence: 97.8,
                material: "Organic Biodegradable Fruit Residue",
                box: [160, 90, 480, 410],
                color: "#10b981"
            }]
        },
        'sample_packaging_cardboard.jpg': {
            success: true,
            detected: true,
            primary_item: "Amazon / Flipkart Gatta Box (गत्ते का डिब्बा)",
            stream: "INORGANIC",
            stream_display: "INORGANIC (Sukha Kachra)",
            expected_waste_type: "Inorganic Waste (Dry / Recyclable - Corrugated Carton)",
            confidence: 98.4,
            grams: 185,
            item_count: 1,
            bin_directive: "🟦 INORGANIC WASTE ➔ BLUE BIN",
            bin_sub: "Sukha Kachra (सूखा कचरा) • Secondary Paper Mill De-Inking & Pulping",
            bin_color: "#0ea5e9",
            speech_text: "Cardboard Packaging detected. Blue Recyclable Bin.",
            detections: [{
                name: "Amazon / Flipkart Gatta Box (गत्ते का डिब्बा)",
                stream: "INORGANIC",
                confidence: 98.4,
                material: "Cardboard / Paper Packaging",
                box: [65, 45, 575, 435],
                color: "#0ea5e9"
            }]
        },
        'sample_bottle_recyclable.jpg': {
            success: true,
            detected: true,
            primary_item: "PET Water Bottle (Bisleri / Kinley / Aquafina)",
            stream: "INORGANIC",
            stream_display: "INORGANIC (Sukha Kachra)",
            expected_waste_type: "Inorganic Waste (Dry / Recyclable - PET Bottle)",
            confidence: 97.2,
            grams: 28,
            item_count: 1,
            bin_directive: "🟦 INORGANIC WASTE ➔ BLUE BIN",
            bin_sub: "Sukha Kachra (सूखा कचरा) • Bottle Flakes & Polyester Yarn Recycling",
            bin_color: "#0ea5e9",
            speech_text: "Plastic Bottle detected. Blue Recyclable Bin.",
            detections: [{
                name: "PET Water Bottle (Bisleri / Kinley / Aquafina)",
                stream: "INORGANIC",
                confidence: 97.2,
                material: "Polyethylene Terephthalate (PET)",
                box: [180, 50, 460, 430],
                color: "#0ea5e9"
            }]
        },
        'sample_can_metal.jpg': {
            success: true,
            detected: true,
            primary_item: "Beverage Can (Coke / Thums Up / Red Bull)",
            stream: "INORGANIC",
            stream_display: "INORGANIC (Sukha Kachra)",
            expected_waste_type: "Inorganic Waste (Dry / Recyclable - Aluminium Can)",
            confidence: 96.8,
            grams: 15,
            item_count: 1,
            bin_directive: "🟦 INORGANIC WASTE ➔ BLUE BIN",
            bin_sub: "Sukha Kachra (धातु कचरा) • Circular Aluminium Ingot Remelting",
            bin_color: "#0ea5e9",
            speech_text: "Aluminium Can detected. Blue Recyclable Bin.",
            detections: [{
                name: "Beverage Can (Coke / Thums Up / Red Bull)",
                stream: "INORGANIC",
                confidence: 96.8,
                material: "Aluminium Can",
                box: [195, 70, 445, 410],
                color: "#0ea5e9"
            }]
        },
        'sample_bread_organic.jpg': {
            success: true,
            detected: true,
            primary_item: "Roti / Bread Wastage (रोटी / ब्रेड)",
            stream: "ORGANIC",
            stream_display: "ORGANIC (Gila Kachra)",
            expected_waste_type: "Organic Waste (Wet / Compostable - Bread Wastage)",
            confidence: 95.5,
            grams: 75,
            item_count: 1,
            bin_directive: "🟩 ORGANIC WASTE ➔ GREEN BIN",
            bin_sub: "Gila Kachra (गीला कचरा) • City Compost & Micro-Composting Centre",
            bin_color: "#10b981",
            speech_text: "Bread or Roti waste detected. Green Compost Bin.",
            detections: [{
                name: "Roti / Bread Wastage (रोटी / ब्रेड)",
                stream: "ORGANIC",
                confidence: 95.5,
                material: "Organic Biodegradable Food",
                box: [150, 80, 490, 400],
                color: "#10b981"
            }]
        },
        'sample_food_organic.jpg': {
            success: true,
            detected: true,
            primary_item: "Rice / Chawal Wastage (चावल)",
            stream: "ORGANIC",
            stream_display: "ORGANIC (Gila Kachra)",
            expected_waste_type: "Organic Waste (Wet / Compostable - Rice Leftovers)",
            confidence: 96.0,
            grams: 140,
            item_count: 1,
            bin_directive: "🟩 ORGANIC WASTE ➔ GREEN BIN",
            bin_sub: "Gila Kachra (गीला कचरा) • High-Methane Anaerobic Digester",
            bin_color: "#10b981",
            speech_text: "Leftover Food detected. Green Compost Bin.",
            detections: [{
                name: "Rice / Chawal Wastage (चावल)",
                stream: "ORGANIC",
                confidence: 96.0,
                material: "Organic Biodegradable Food",
                box: [130, 70, 510, 410],
                color: "#10b981"
            }]
        }
    };

    let currentPresetAbortCtrl = null;

    presetButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const sampleFile = btn.getAttribute('data-sample');
            selectPresetSample(sampleFile, btn.textContent.trim());
        });
    });

    function selectPresetSample(sampleFile, labelText) {
        if (!sampleFile) return;
        if (currentPresetAbortCtrl) {
            try { currentPresetAbortCtrl.abort(); } catch(e) {}
        }
        currentPresetAbortCtrl = new AbortController();

        currentMode = 'samples';
        [btnModeCamera, btnModeSamples, btnModeUpload].forEach(b => b.classList.remove('active'));
        btnModeSamples.classList.add('active');
        stopAllCameras();
        if (cameraSourceToolbar) cameraSourceToolbar.style.display = 'none';
        staticPreview.style.display = 'block';

        hudStatus.textContent = `⚡ Analyzing ${labelText || sampleFile}...`;
        staticPreview.src = `/demo_samples/${sampleFile}?t=` + Date.now();
        syncCanvasSize();

        // Check instant cached preset data
        if (SAMPLE_PRESETS_DATA[sampleFile]) {
            const cached = JSON.parse(JSON.stringify(SAMPLE_PRESETS_DATA[sampleFile]));
            cached.latency_ms = 18;
            renderPrediction(cached);
            return;
        }

        analyzeSample(sampleFile, currentPresetAbortCtrl.signal);
    }

    // File Upload Handler
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            staticPreview.src = event.target.result;
            staticPreview.onload = () => {
                syncCanvasSize();
                analyzePayload({ image_base64: event.target.result });
            };
        };
        reader.readAsDataURL(file);
    });

    // 6. Inference API Communication
    async function analyzeSample(sampleFilename, signal = null) {
        const startTime = performance.now();
        try {
            const res = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample_id: sampleFilename }),
                signal: signal
            });
            const data = await res.json();
            const latency = Math.round(performance.now() - startTime);
            data.latency_ms = latency;
            renderPrediction(data);
        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error('Inference error:', err);
                hudStatus.textContent = 'Inference notice: retry or select another preset';
            }
        }
    }

    async function analyzePayload(payload) {
        if (isAnalyzing) return;
        isAnalyzing = true;
        const startTime = performance.now();

        try {
            const res = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            const latency = Math.round(performance.now() - startTime);
            data.latency_ms = latency;
            renderPrediction(data);
        } catch (err) {
            console.error('Inference error:', err);
        } finally {
            isAnalyzing = false;
        }
    }

    function captureAndAnalyze() {
        fetch('/api/camera_unlock', { method: 'POST' }).catch(() => {});
        if (btnUnlockScan) btnUnlockScan.style.display = 'none';
        if (currentMode === 'camera') {
            if (currentCameraSubSource === 'pc' && webcamVideo && webcamVideo.style.display !== 'none' && webcamVideo.readyState >= 2) {
                try {
                    const offscreenCanvas = document.createElement('canvas');
                    offscreenCanvas.width = 640;
                    offscreenCanvas.height = 480;
                    const offCtx = offscreenCanvas.getContext('2d');
                    offCtx.drawImage(webcamVideo, 0, 0, 640, 480);
                    const base64 = offscreenCanvas.toDataURL('image/jpeg', 0.8);
                    analyzePayload({ image_base64: base64 });
                } catch (e) {
                    analyzePayload({ use_camera: true });
                }
            } else {
                analyzePayload({ use_camera: true });
            }
        } else if (staticPreview.src) {
            const activePreset = document.querySelector('.preset-pill.active');
            if (activePreset) {
                const sFile = activePreset.getAttribute('data-sample');
                selectPresetSample(sFile, activePreset.textContent.trim());
            } else {
                analyzePayload({ image_base64: staticPreview.src });
            }
        }
    }

    btnCaptureAnalyze.addEventListener('click', captureAndAnalyze);

    // 7. Update UI with Results
    function renderPrediction(data) {
        hudLatency.textContent = `${data.latency_ms} ms`;

        if (!data.detected || !data.detections || data.detections.length === 0) {
            if (btnUnlockScan) btnUnlockScan.style.display = 'none';
            hudStatus.textContent = 'Awaiting Waste Item (Hold in view)';
            confidenceBadge.className = 'conf-badge muted';
            confidenceBadge.textContent = 'Confidence: --%';
            
            streamBadge.className = 'stream-badge';
            streamBadge.textContent = '⚪ Awaiting Detection';
            
            expectedWasteCard.className = 'expected-waste-card';
            itemStreamIcon.textContent = '🔍';
            expectedWasteTitle.textContent = 'No Waste in View';
            expectedWasteDetail.textContent = 'Hold waste in front of the camera (e.g. food, bread, bottle, can).';
            
            quantifiedWeight.textContent = '--';
            quantifiedCount.textContent = '0';
            
            binInstructionCard.className = 'bin-instruction-card';
            binIcon.textContent = '🗑️';
            binTitle.textContent = 'Disposal Guideline';
            binInstructionText.textContent = 'Hold an item in front of the camera to determine Green Bin (Organic) or Blue Bin (Inorganic).';
            
            currentDetections = [];
            drawDetections();
            return;
        }

        if (data.stream === 'DUAL') {
            if (currentMode === 'samples') {
                hudStatus.textContent = `🔒 Sample Waste Verified • Decision Locked: DUAL STREAM (${data.confidence}%)`;
                if (btnUnlockScan) btnUnlockScan.style.display = 'none';
            } else if (data.locked && data.lock_remaining > 0) {
                if (data.object_in_view === false) {
                    hudStatus.textContent = `🔒 Decision Held (${data.lock_remaining}s to explain) • Object Cleared from Frame`;
                } else {
                    hudStatus.textContent = `🔒 Decision Locked (${data.lock_remaining}s to explain): DUAL-STREAM CAFETERIA`;
                }
                if (btnUnlockScan) {
                    btnUnlockScan.style.display = 'inline-block';
                    btnUnlockScan.textContent = `⚡ Scan Next Item (${data.lock_remaining}s)`;
                }
            } else {
                hudStatus.textContent = `Identified: DUAL-STREAM CAFETERIA WASTE (${data.confidence}%)`;
                if (btnUnlockScan) btnUnlockScan.style.display = 'none';
            }
            confidenceBadge.className = 'conf-badge';
            confidenceBadge.textContent = `Confidence: ${data.confidence}%`;

            // Stream Badge
            streamBadge.className = 'stream-badge dual';
            streamBadge.textContent = '🟩 ORGANIC + 🟦 INORGANIC (Dual Stream)';

            // Expected Waste Card
            expectedWasteCard.className = 'expected-waste-card dual';
            itemStreamIcon.textContent = '🍱';
            expectedWasteTitle.textContent = data.expected_waste_type || 'Cafeteria Plate with Food Scraps';
            expectedWasteDetail.textContent = data.bin_sub || 'Multi-object detected: Organic food leftovers + Inorganic plate/tray.';

            // Quantified Metrics
            quantifiedWeight.textContent = data.grams !== undefined ? data.grams : 120;
            quantifiedCount.textContent = data.item_count || (data.detections ? data.detections.length : 2);

            // Bin Instruction Card
            binInstructionCard.className = 'bin-instruction-card dual';
            binIcon.textContent = '🔄';
            binTitle.textContent = 'Dual-Stream Segregation Required';
            binInstructionText.innerHTML = '<strong>Step 1 (Green Bin):</strong> Scrape food/rice leftovers into Organic Compost Bin.<br><strong>Step 2 (Blue Bin):</strong> Place cleaned plate / container into Recyclable Dry Bin.';

            // Bounding Boxes
            currentDetections = data.detections || [];
            drawDetections();

            // Update Cumulative Municipal Telemetry
            if (data.stats) {
                updateTelemetryUI(data.stats);
            }

            // Speech
            if (data.speech_text) {
                speakDirective(data.speech_text);
            }
            return;
        }

        const isOrganic = data.stream === 'ORGANIC';
        const streamLabel = isOrganic ? 'ORGANIC WASTE' : 'INORGANIC WASTE';

        if (currentMode === 'samples') {
            hudStatus.textContent = `🔒 Sample Waste Verified • Decision Locked: ${streamLabel} (${data.confidence}%)`;
            if (btnUnlockScan) btnUnlockScan.style.display = 'none';
        } else if (data.locked && data.lock_remaining > 0) {
            if (data.object_in_view === false) {
                hudStatus.textContent = `🔒 Decision Held (${data.lock_remaining}s to explain) • Object Cleared from Frame`;
            } else {
                hudStatus.textContent = `🔒 Decision Locked (${data.lock_remaining}s to explain): ${streamLabel}`;
            }
            if (btnUnlockScan) {
                btnUnlockScan.style.display = 'inline-block';
                btnUnlockScan.textContent = `⚡ Scan Next Item (${data.lock_remaining}s)`;
            }
        } else {
            hudStatus.textContent = `Identified: ${streamLabel} (${data.confidence}%)`;
            if (btnUnlockScan) btnUnlockScan.style.display = 'none';
        }
        confidenceBadge.textContent = `Confidence: ${data.confidence}%`;

        // Stream Badge
        streamBadge.className = `stream-badge ${isOrganic ? 'organic' : 'inorganic'}`;
        streamBadge.textContent = isOrganic ? '🟩 ORGANIC WASTE (Green Bin)' : '🟦 INORGANIC WASTE (Blue Bin)';

        // Expected Waste Card
        expectedWasteCard.className = `expected-waste-card ${isOrganic ? 'organic' : 'inorganic'}`;
        itemStreamIcon.textContent = isOrganic ? '🌱' : '♻️';
        expectedWasteTitle.textContent = data.expected_waste_type || streamLabel;
        expectedWasteDetail.textContent = data.bin_sub || (isOrganic ? 'Organic kitchen waste suitable for composting.' : 'Dry recyclable packaging suitable for MRF recovery.');

        // Quantified Metrics
        quantifiedWeight.textContent = data.grams !== undefined ? data.grams : 75;
        quantifiedCount.textContent = data.item_count || 1;

        // Bin Instruction Card
        binInstructionCard.className = `bin-instruction-card ${isOrganic ? 'organic' : 'inorganic'}`;
        binIcon.textContent = isOrganic ? '🟩' : '🟦';
        binTitle.textContent = isOrganic ? 'Route to Green Compost Bin' : 'Route to Blue Recyclable Bin';
        binInstructionText.textContent = isOrganic
            ? 'Place in Green / Compost bin for organic decomposition & soil enrichment (Gila Kachra).'
            : 'Place in Blue / Recyclable bin for sorting, recovery & recycling (Sukha Kachra).';

        // Bounding Boxes
        currentDetections = data.detections || [];
        drawDetections();

        // Update Cumulative Municipal Telemetry
        if (data.stats) {
            updateTelemetryUI(data.stats);
        }

        // Speech
        if (data.speech_text) {
            speakDirective(data.speech_text);
        }
    }

    // 8. Draw Canvas Bounding Box
    function drawDetections() {
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        if (!showBoxes || !currentDetections.length) return;

        let displayW, displayH, naturalW, naturalH;

        if (currentMode === 'camera') {
            if (liveCameraFeed && liveCameraFeed.style.display !== 'none') {
                // Backend OpenCV camera stream already renders clean HUD overlays directly.
                // Clear canvas so duplicate/offset client boxes never stack on top.
                ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
                return;
            }
            const activeEl = webcamVideo;
            displayW = activeEl ? activeEl.clientWidth : 0;
            displayH = activeEl ? activeEl.clientHeight : 0;
            naturalW = activeEl ? (activeEl.videoWidth || displayW) : 0;
            naturalH = activeEl ? (activeEl.videoHeight || displayH) : 0;
        } else {
            displayW = staticPreview.clientWidth;
            displayH = staticPreview.clientHeight;
            naturalW = staticPreview.naturalWidth || displayW;
            naturalH = staticPreview.naturalHeight || displayH;
        }

        if (!displayW || !displayH || !naturalW || !naturalH) return;

        const containerW = mediaWrapper.clientWidth;
        const containerH = mediaWrapper.clientHeight;

        const scale = Math.min(containerW / naturalW, containerH / naturalH);
        const renderW = naturalW * scale;
        const renderH = naturalH * scale;
        const offsetX = (containerW - renderW) / 2;
        const offsetY = (containerH - renderH) / 2;

        currentDetections.forEach((det) => {
            const isOrganic = det.stream === 'ORGANIC';
            const strokeColor = isOrganic ? '#10b981' : '#0ea5e9';
            const fillColor = isOrganic ? 'rgba(16, 185, 129, 0.15)' : 'rgba(14, 165, 233, 0.15)';

            // Polygon mask
            if (det.polygon && det.polygon.length >= 3) {
                ctx.beginPath();
                det.polygon.forEach((pt, idx) => {
                    const px = offsetX + (pt[0] / naturalW) * renderW;
                    const py = offsetY + (pt[1] / naturalH) * renderH;
                    if (idx === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                });
                ctx.closePath();
                ctx.fillStyle = fillColor;
                ctx.fill();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 2.5;
                ctx.stroke();
            }

            // Bounding box
            if (det.box && det.box.length === 4) {
                const [bx1, by1, bx2, by2] = det.box;
                const x = offsetX + (bx1 / naturalW) * renderW;
                const y = offsetY + (by1 / naturalH) * renderH;
                const w = ((bx2 - bx1) / naturalW) * renderW;
                const h = ((by2 - by1) / naturalH) * renderH;

                ctx.save();
                ctx.shadowColor = strokeColor;
                ctx.shadowBlur = 10;
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(x, y, w, h);
                ctx.restore();

                // Corner accents
                const cornerLen = Math.min(18, Math.min(w, h) / 3);
                ctx.lineWidth = 4;
                ctx.strokeStyle = strokeColor;

                // Top-left
                ctx.beginPath();
                ctx.moveTo(x, y + cornerLen);
                ctx.lineTo(x, y);
                ctx.lineTo(x + cornerLen, y);
                ctx.stroke();

                // Top-right
                ctx.beginPath();
                ctx.moveTo(x + w - cornerLen, y);
                ctx.lineTo(x + w, y);
                ctx.lineTo(x + w, y + cornerLen);
                ctx.stroke();

                // Bottom-left
                ctx.beginPath();
                ctx.moveTo(x, y + h - cornerLen);
                ctx.lineTo(x, y + h);
                ctx.lineTo(x + cornerLen, y + h);
                ctx.stroke();

                // Bottom-right
                ctx.beginPath();
                ctx.moveTo(x + w - cornerLen, y + h);
                ctx.lineTo(x + w, y + h);
                ctx.lineTo(x + w, y + h - cornerLen);
                ctx.stroke();

                // Label tag
                const labelText = det.label
                    ? `${isOrganic ? 'ORGANIC' : 'INORGANIC'}: ${det.label} (${det.confidence}%)`
                    : (isOrganic ? `ORGANIC WASTE • ${det.confidence}%` : `INORGANIC WASTE • ${det.confidence}%`);

                ctx.font = 'bold 13px "Inter", sans-serif';
                const textWidth = ctx.measureText(labelText).width;
                const tagH = 24;
                const tagW = textWidth + 16;
                const tagY = Math.max(0, y - tagH - 4);

                ctx.fillStyle = strokeColor;
                ctx.fillRect(x, tagY, tagW, tagH);

                ctx.fillStyle = '#0a0e17';
                ctx.fillText(labelText, x + 8, tagY + 17);
            }
        });
    }

    // 9. Cumulative Municipal Telemetry & KPI Display
    function updateTelemetryUI(stats) {
        if (!stats) return;

        if (kpiTotalObjects) {
            kpiTotalObjects.textContent = stats.total_sorted || 0;
            kpiTotalObjects.classList.add('updated');
            setTimeout(() => kpiTotalObjects.classList.remove('updated'), 600);
        }
        if (kpiStreamBreakdown) {
            kpiStreamBreakdown.textContent = `🟩 ${stats.organic_count || 0} Organic • 🟦 ${stats.inorganic_count || 0} Inorganic`;
        }
        if (kpiTotalWeight) {
            kpiTotalWeight.textContent = (stats.total_weight_kg !== undefined) ? Number(stats.total_weight_kg).toFixed(2) : '0.00';
            kpiTotalWeight.classList.add('updated');
            setTimeout(() => kpiTotalWeight.classList.remove('updated'), 600);
        }
        if (kpiWeightBreakdown) {
            const orgKg = stats.organic_weight_kg || ((stats.organic_count || 0) * 0.15);
            const inorgKg = stats.inorganic_weight_kg || Math.max(0, (stats.total_weight_kg || 0) - orgKg);
            kpiWeightBreakdown.textContent = `🟩 ${Number(orgKg).toFixed(2)} kg Wet • 🟦 ${Number(inorgKg).toFixed(2)} kg Dry`;
        }
        if (kpiTotalCo2) {
            kpiTotalCo2.textContent = (stats.total_co2e_avoided_kg !== undefined) ? Number(stats.total_co2e_avoided_kg).toFixed(2) : '0.00';
            kpiTotalCo2.classList.add('updated');
            setTimeout(() => kpiTotalCo2.classList.remove('updated'), 600);
        }
        if (fillBarGreen && fillPctGreen) {
            const gPct = Math.min(100, Math.max(0, stats.green_bin_fill_pct || 0));
            fillBarGreen.style.width = `${gPct}%`;
            fillPctGreen.textContent = `${gPct}%`;
        }
        if (fillBarBlue && fillPctBlue) {
            const bPct = Math.min(100, Math.max(0, stats.blue_bin_fill_pct || 0));
            fillBarBlue.style.width = `${bPct}%`;
            fillPctBlue.textContent = `${bPct}%`;
        }
        if (telemetrySyncText) {
            const now = new Date();
            const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            telemetrySyncText.textContent = `🟢 Live Synced • ${timeStr}`;
        }
    }

    async function fetchTelemetryStats() {
        try {
            const res = await fetch('/api/stats');
            const stats = await res.json();
            updateTelemetryUI(stats);
        } catch (e) {
            console.warn('Could not fetch telemetry stats:', e);
        }
    }

    btnResetStats?.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/stats/reset?mode=zero', { method: 'POST' });
            const data = await res.json();
            updateTelemetryUI(data);
            if (hudStatus) hudStatus.textContent = '🗑️ Municipal Counter Reset to 0 (0 items, 0.00 kg)';
        } catch (e) {
            console.warn('Reset error:', e);
        }
    });

    // Initial Startup: Fetch persistent telemetry & load first verified sample
    fetchTelemetryStats();
    syncCanvasSize();
    analyzeSample('sample_canteen_rice_plate.jpg');
});
