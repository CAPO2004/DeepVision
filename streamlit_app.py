import os, urllib.request, tempfile, cv2, torch, torch.nn as nn, numpy as np, base64, io
from PIL import Image
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
st.set_page_config(page_title="DeepVision - AI Deepfake Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
#MainMenu, header[data-testid="stHeader"], footer {display: none !important;}
.block-container {padding-top: 0rem; padding-bottom: 1rem; max-width: 1200px;}
.stApp {background-color: #0a0f1a; font-family: 'Inter', sans-serif; color: #fff; overflow-x: hidden;}
/* Background blobs */
.stApp::before, .stApp::after {content: ''; position: fixed; border-radius: 50%; z-index: 0; pointer-events: none;}
.stApp::before {top: -10%; left: -10%; width: 50vw; height: 50vw; background: radial-gradient(circle, rgba(0,85,255,0.15) 0%, rgba(0,0,0,0) 70%);}
.stApp::after {bottom: -10%; right: -10%; width: 60vw; height: 60vw; background: radial-gradient(circle, rgba(0,240,255,0.1) 0%, rgba(0,0,0,0) 70%);}
/* Navbar */
.navbar {
    background: rgba(17, 25, 40, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    padding: 0.75rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: fixed;
    top: 0; left: 0; right: 0;
    width: 100vw;
    z-index: 50;
}
.navbar-brand {display: flex; align-items: center; gap: 0.5rem; cursor: pointer;}
.navbar .icon {color: #00f0ff; font-size: 1.5rem;}
.navbar h1 {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    margin: 0;
    color: #fff;
}
/* Hero Section */
.hero {
    text-align: center;
    padding: 2.5rem 1rem;
    position: relative;
    z-index: 1;
    animation: fadeIn 0.8s ease-out forwards;
    margin-top: 5rem;
}
.hero h2 {
    font-size: 2.25rem;
    font-weight: 800;
    margin-bottom: 1rem;
    color: #fff;
    line-height: 1.2;
}
@media (min-width: 768px) {
    .hero h2 {font-size: 3.75rem !important;}
}
.hero .gradient-text {
    background: linear-gradient(to right, #0055ff, #00f0ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero p {
    color: #9ca3af;
    font-size: 1.125rem;
    max-width: 42rem;
    margin: 0 auto 2.5rem;
}
@media (min-width: 768px) {
    .hero p {font-size: 1.25rem;}
}
/* ═══ CUSTOM GLASS BOX CONTAINER ═══ */
/* Exact selector for the inner stVerticalBlock housing the uploader */
div[data-testid="stVerticalBlock"]:has(> .element-container #uploader-marker) {
    background: rgba(17, 25, 40, 0.6) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
    border-radius: 1rem;
    padding: 2rem !important;
    max-width: 56rem !important;
    margin: 0 auto 1rem auto !important;
    position: relative;
    z-index: 1;
    animation: fadeIn 0.8s ease-out forwards;
    animation-delay: 100ms;
    opacity: 0;
}
/* Remove default file uploader box since container has it */
.stFileUploader {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.stFileUploader > div > div {background: transparent !important;}
.stFileUploader label {display: none !important;}
/* ═══ BUTTON STYLING ═══ */
/* Analyze button: gradient, matches flex-[2] in original HTML */
div.element-container:has(#analyze-btn-marker) {
    display: none !important;
}
div.element-container:has(#analyze-btn-marker) + div.element-container button {
    box-sizing: border-box !important;
    background: linear-gradient(to right, #0055ff, #00f0ff) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0 1.5rem !important;
    font-size: 1.1rem !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.5) !important;
    height: 56px !important;
    width: 100% !important;
    transition: transform 0.3s, opacity 0.3s;
    margin: 0 !important;
}
div.element-container:has(#analyze-btn-marker) + div.element-container button:hover {
    transform: scale(1.02);
}
div.element-container:has(#analyze-btn-marker) + div.element-container button p {
    display: flex !important;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    font-size: 1.1rem !important;
    margin: 0;
}
/* Add More custom button */
div.stMarkdown p:has(> .custom-add-more) {
    margin: 0 !important;
}
.custom-add-more {
    box-sizing: border-box !important;
    width: 100%;
    height: 56px !important;
    background: #374151;
    color: white;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    border: 1px solid #4b5563;
    border-radius: 0.5rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    transition: background 0.3s;
    margin: 0;
    padding: 0;
}
.custom-add-more:hover {
    background: #4b5563;
}
/* File Cards */
.result-card {
    background: rgba(17, 25, 40, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid #374151;
    border-radius: 0.75rem;
    padding: 1rem;
    margin: 1rem auto;
    max-width: 56rem;
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    animation: fadeIn 0.8s ease-out;
}
@media (min-width: 768px) {
    .result-card {flex-direction: row; align-items: center;}
}
.result-card.real {border-color: #4ade80; box-shadow: 0 0 15px rgba(74, 222, 128, 0.2);}
.result-card.fake {border-color: #ef4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);}
.result-card.error {border-color: #facc15; box-shadow: 0 0 15px rgba(250, 204, 21, 0.2);}
.media-thumbnail {
    width: 6rem;
    height: 6rem;
    flex-shrink: 0;
    background-color: #000;
    border-radius: 0.5rem;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid #374151;
}
@media (max-width: 767px) {
    .hidden-mobile { display: none !important; }
}
.file-info-container {
    flex: 1;
    min-width: 0;
    padding-right: 1rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.file-info {font-weight: 700; font-size: 1.125rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 0.25rem; color: #fff;}
.file-size {font-size: 0.875rem; color: #9ca3af; margin-bottom: 0.75rem;}
/* Status and Progress inside Card */
.status-container {width: 100%;}
.result-header {display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.25rem;}
.result-status {font-size: 1.125rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em;}
.result-status.real {color: #4ade80;} .result-status.fake {color: #ef4444;} .result-status.error {color: #facc15;}
.result-confidence {font-size: 1.5rem; font-weight: 900;}
.result-confidence.real {color: #4ade80;} .result-confidence.fake {color: #ef4444;}
.progress-bg {width: 100%; height: 0.5rem; background: #1f2937; border-radius: 999px; margin-bottom: 0.25rem;}
.progress-fill {height: 100%; border-radius: 999px; transition: width 1s;}
.progress-fill.real {background: #4ade80;} .progress-fill.fake {background: #ef4444;} .progress-fill.error {background: #facc15;}
.result-reason {color: #9ca3af; font-style: italic; font-size: 0.75rem; margin: 0;}
.result-stats {color: #00f0ff; font-family: monospace; font-size: 0.625rem; letter-spacing: 0.05em; margin-top: 0.25rem;}
/* Footer */
.custom-footer {border-top: 1px solid #1f2937; margin-top: auto; padding: 2rem 0; text-align: center; color: #9ca3af; font-weight: 500; position: relative; z-index: 1;}
@keyframes fadeIn{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
::-webkit-scrollbar{width:8px;}::-webkit-scrollbar-track{background:#0a0f1a;}::-webkit-scrollbar-thumb{background:#1e3a8a;border-radius:4px;}::-webkit-scrollbar-thumb:hover{background:#00f0ff;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <i class="fa-solid fa-shield-halved icon"></i>
        <h1>DeepVision</h1>
    </div>
</div>
<div class="hero">
    <h2>AI-Powered <span class="gradient-text">Deepfake Detection</span></h2>
    <p>Batch Analysis Mode: Upload multiple images and videos simultaneously to verify their authenticity.</p>
</div>
""", unsafe_allow_html=True)

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        scale = self.sigmoid(
            self.conv(torch.cat([avg, mx], dim=1))
        )
        return x * scale

class CBAM(nn.Module):

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))

class DeepfakeDetector(nn.Module):

    def __init__(self, num_classes=2, pretrained=False, dropout=0.4):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b4(weights=weights)
        self.backbone = base.features
        self.cbam = CBAM(1792)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(1792),
            nn.Dropout(dropout),
            nn.Linear(1792, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

@st.cache_resource(show_spinner="🔄 Loading AI Models...")

def load_models(_cache_v="v4_single"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp_path = "blaze_face_short_range.tflite"

    if not os.path.exists(mp_path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            mp_path)
    base_options = python.BaseOptions(model_asset_path=mp_path)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    face_detector = vision.FaceDetector.create_from_options(options)
    model = DeepfakeDetector(num_classes=2).to(device)
    MODEL_PATH = "checkpoints/model_images.pth"

    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "checkpoints/efficientnet_cbam_best.pth"

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    else:

        st.error(f"❌ Model weights not found!")
        st.stop()
    model.eval()
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return face_detector, model, model, test_tf, device

def get_face_crop(frame_rgb, detection, margin=0.2):
    h, w, _ = frame_rgb.shape
    bbox = detection.bounding_box
    x_m = int(bbox.width * margin)
    y_m = int(bbox.height * margin)
    x1 = max(0, int(bbox.origin_x) - x_m)
    y1 = max(0, int(bbox.origin_y) - y_m)
    x2 = min(w, int(bbox.origin_x + bbox.width) + x_m)
    y2 = min(h, int(bbox.origin_y + bbox.height) + y_m)
    return frame_rgb[y1:y2, x1:x2]

face_detector, model_video, model_image, test_transforms, device = load_models()
import streamlit.components.v1 as components

with st.container():

    st.markdown('<div id="uploader-marker"></div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'], accept_multiple_files=True)
    analyze_clicked = False

    if not uploaded_files:

        st.markdown("""
        <style>
        /* Dropzone dashed border */
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: 2px dashed #4b5563 !important;
            border-radius: 0.75rem !important;
            padding: 0 !important;
            position: relative !important;
            z-index: 10 !important;
            transition: all 0.3s;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #9ca3af !important;
        }
        /* Inner wrapper = 300px centered flex */
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type {
            position: relative !important;
            height: 300px !important;
            width: 100% !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
        }
        /* Hide native text/icons */
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type span,
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type p,
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type small,
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type svg {
            display: none !important;
        }
        /* Cloud Icon */
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type::before {
            content: "" !important;
            position: absolute !important;
            top: 2.5rem !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            width: 5rem !important;
            height: 5rem !important;
            background-color: #1f2937 !important;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 640 512' fill='%2300f0ff'%3E%3Cpath d='M144 480C64.5 480 0 415.5 0 336c0-62.8 40.2-116.2 96.2-135.9c-.1-2.7-.2-5.4-.2-8.1c0-88.4 71.6-160 160-160c59.3 0 111 32.2 138.7 80.2C409.9 102 428.3 96 448 96c53 0 96 43 96 96c0 12.2-2.3 23.8-6.4 34.6C596 238.4 640 290.1 640 352c0 70.7-57.3 128-128 128H144zm79-217c-9.4 9.4-9.4 24.6 0 33.9s24.6 9.4 33.9 0l39-39V392c0 13.3 10.7 24 24 24s24-10.7 24-24V257.9l39 39c9.4 9.4 24.6 9.4 33.9 0s9.4-24.6 0-33.9l-80-80c-9.4-9.4-24.6-9.4-33.9 0l-80 80z'/%3E%3C/svg%3E") !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            background-size: 1.875rem !important;
            border-radius: 50% !important;
            z-index: 1 !important;
        }
        /* Title text */
        .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-of-type::after {
            content: "Drag & Drop your media files here" !important;
            position: absolute !important;
            top: 8.5rem !important;
            left: 0 !important; right: 0 !important;
            text-align: center !important;
            color: #fff !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            z-index: 1 !important;
        }
        /* Browse Files fake button overlay */
        .stFileUploader [data-testid="stFileUploaderDropzone"]::after {
            content: "Browse Files" !important;
            position: absolute !important;
            bottom: 2.5rem !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            background: linear-gradient(to right, #0055ff, #00f0ff) !important;
            color: #fff !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
            padding: 0.5rem 1.5rem !important;
            box-shadow: 0 0 15px rgba(0,240,255,0.4) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            pointer-events: none !important;
            z-index: 20 !important;
        }
        /* Subtext */
        .stFileUploader [data-testid="stFileUploaderDropzone"]::before {
            content: "You can select multiple Images and Videos" !important;
            position: absolute !important;
            bottom: 6rem !important;
            left: 0 !important; right: 0 !important;
            text-align: center !important;
            color: #9ca3af !important;
            font-size: 1rem !important;
            pointer-events: none !important;
        }
        /* Hide native button */
        .stFileUploader [data-testid="stFileUploaderDropzone"] button {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

    else:

        st.markdown('<style> .stFileUploader { position: absolute !important; top: -9999px !important; opacity: 0 !important; pointer-events: none !important; } </style>', unsafe_allow_html=True)
        cards_html = ""

        for i, file in enumerate(uploaded_files):
            file_bytes = file.read()
            file.seek(0)
            file_size_mb = len(file_bytes) / (1024 * 1024)
            is_video = file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

            if is_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(file_bytes)
                tfile.close()
                cap = cv2.VideoCapture(tfile.name)
                media_html = '<i class="fa-solid fa-film" style="font-size: 2rem; color: #6b7280;"></i>'
                ret, first_frame = cap.read()

                if ret:
                    _, buffer = cv2.imencode('.jpg', first_frame)
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    media_html = f'<img src="data:image/jpeg;base64,{b64_img}" style="width:100%; height:100%; object-fit:cover;">'
                cap.release()
                os.unlink(tfile.name)

            else:
                mime = "image/png" if file.name.lower().endswith("png") else "image/jpeg"

                try:
                    img = Image.open(io.BytesIO(file_bytes))
                    img.thumbnail((150, 150))
                    buf = io.BytesIO()
                    img.save(buf, format=img.format if img.format else 'JPEG')
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                except Exception:
                    b64 = base64.b64encode(file_bytes).decode('utf-8')
                media_html = f'<img src="data:{mime};base64,{b64}" style="width:100%; height:100%; object-fit:cover;">'
            cards_html += f"""
            <div style="display: flex; flex-direction: row; align-items: center; justify-content: flex-start; background: rgba(17, 25, 40, 0.6); border: 1px solid #374151; border-radius: 0.75rem; padding: 1.25rem 1.5rem; margin-bottom: 1rem; width: 100%; box-sizing: border-box; gap: 1rem;">
                <div style="width: 5rem; height: 5rem; background: #000; border-radius: 0.5rem; overflow: hidden; flex-shrink: 0; border: 1px solid #374151;">
                    {media_html}
                </div>
                <div style="display: flex; flex-direction: column; flex: 1; min-width: 0;">
                    <div style="color: #fff; font-weight: 700; font-size: 1.1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 0.25rem;">{file.name}</div>
                    <div style="color: #9ca3af; font-weight: 400; font-size: 0.9rem;">{file_size_mb:.2f} MB</div>
                </div>
                <div class="custom-delete-btn" data-index="{i}" data-filename="{file.name}" style="color: #ef4444; font-size: 1.25rem; background: #1f2937; width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; opacity: 0.8; cursor: pointer; transition: opacity 0.3s;" title="Remove {file.name}">
                    <i class="fa-solid fa-xmark"></i>
                </div>
            </div>
            """

        st.markdown(cards_html, unsafe_allow_html=True)
        components.html("""
        <script>
            const parentWin = window.parent;

            if (!parentWin.__deepvisionListener) {
                parentWin.document.addEventListener('click', function(e) {
                    // Handle Custom Add More Button

                    if (e.target.closest('.custom-add-more')) {
                        let fileInput = parentWin.document.querySelector('input[type=file]');

                        if (fileInput) fileInput.click();
                    }
                    // Handle Custom Delete Button (Index-based with triple fallback)
                    let delBtn = e.target.closest('.custom-delete-btn');

                    if (delBtn) {
                        let index = parseInt(delBtn.getAttribute('data-index'));
                        let filename = delBtn.getAttribute('data-filename');
                        // Strategy 1: Find by standard Streamlit item container
                        let fileItems = Array.from(parentWin.document.querySelectorAll('[data-testid="stUploadedFile"]'));

                        if (fileItems.length > 0 && fileItems[index]) {
                            let btn = fileItems[index].querySelector('button');

                            if (btn) { btn.click(); return; }
                        }
                        // Strategy 2: Get all buttons in the uploader that lack text (Delete buttons are typically just an SVG)
                        let allBtns = Array.from(parentWin.document.querySelectorAll('.stFileUploader button'));
                        let iconOnlyBtns = allBtns.filter(b => !b.textContent.trim());

                        if (iconOnlyBtns.length > 0 && iconOnlyBtns[index]) {
                            iconOnlyBtns[index].click();
                            return;
                        }
                        // Strategy 3: Find by aria-label matching the filename
                        let ariaBtns = allBtns.filter(b => b.hasAttribute('aria-label') && b.getAttribute('aria-label').includes(filename));

                        if (ariaBtns.length > 0) {
                            ariaBtns[0].click();
                            return;
                        }
                        console.error("DeepVision: Could not locate native delete button for", filename);
                    }
                });
                parentWin.__deepvisionListener = true;
            }
        </script>
        """, height=0)
        col_add, col_analyze = st.columns([1, 2], gap="medium")

        with col_add:

            st.markdown('''
            <button class="custom-add-more">
                <i class="fa-solid fa-plus"></i> Add More
            </button>
            ''', unsafe_allow_html=True)

        with col_analyze:

            st.markdown('<div id="analyze-btn-marker"></div>', unsafe_allow_html=True)
            analyze_clicked = st.button("🔍 Analyze All Media", use_container_width=True)

    if analyze_clicked:

        st.markdown("<hr style='border-color:#1f2937; margin:2rem 0;'>", unsafe_allow_html=True)
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            status_text.markdown(f"**Analyzing:** `{file.name}` ({i+1}/{total_files})")
            file_bytes = file.read()
            file_size_mb = len(file_bytes) / (1024 * 1024)
            is_video = file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

            if not is_video:

                try:
                    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

                except Exception as e:
                    continue
                rgb = np.array(image)
                res = face_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                valid_detections = [d for d in res.detections if d.bounding_box.width >= 50 and d.bounding_box.height >= 50]

                if not valid_detections:

                    st.error(f"❌ No detectable valid face found in {file.name}")
                    continue
                is_fake = False
                max_score = 0.0

                for det in valid_detections:
                    face_crop = get_face_crop(rgb, det, margin=0.20)

                    if face_crop.size == 0: continue
                    pil_normal = Image.fromarray(face_crop)
                    t_normal = test_transforms(pil_normal).unsqueeze(0).to(device)

                    with torch.no_grad():
                        score = torch.softmax(model_image(t_normal), dim=1)[0, 0].item()
                        max_score = max(max_score, score)

                        if score >= 0.50:
                            is_fake = True

                if is_fake:
                    c, icon, conf = "fake", '<i class="fa-solid fa-triangle-exclamation mr-1"></i> FAKE', max_score * 100
                    reason = "Deepfake Artifacts Detected"

                else:
                    c, icon, conf = "real", '<i class="fa-solid fa-check-circle mr-1"></i> REAL', (1 - max_score) * 100
                    reason = "Authentic Media"
                conf = min(99.9, round(conf, 1))
                mime = "image/png" if file.name.lower().endswith("png") else "image/jpeg"

                try:
                    img_thumb = Image.open(io.BytesIO(file_bytes))
                    img_thumb.thumbnail((150, 150))
                    buf_thumb = io.BytesIO()
                    img_thumb.save(buf_thumb, format=img_thumb.format if img_thumb.format else 'JPEG')
                    b64_img = base64.b64encode(buf_thumb.getvalue()).decode('utf-8')

                except Exception:
                    b64_img = base64.b64encode(file_bytes).decode('utf-8')
                media_html = f'<img src="data:{mime};base64,{b64_img}" style="width:100%; height:100%; object-fit:cover;">'

                st.markdown(f'''
                <div class="result-card {c}">
                    <div class="media-thumbnail hidden-mobile">
                        {media_html}
                    </div>
                    <div class="file-info-container">
                        <div class="file-info">{file.name}</div>
                        <div class="file-size">{file_size_mb:.2f} MB</div>
                        <div class="status-container">
                            <div class="result-header">
                                <span class="result-status {c}">{icon}</span>
                                <span class="result-confidence {c}">{conf}%</span>
                            </div>
                            <div class="progress-bg">
                                <div class="progress-fill {c}" style="width:{conf}%;"></div>
                            </div>
                            <p class="result-reason">{reason}</p>
                            <p class="result-stats">Confidence: {conf}%</p>
                        </div>
                    </div>
                </div>''', unsafe_allow_html=True)

            else:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(file_bytes)
                tfile.flush()
                tfile.close()
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                media_html = '<i class="fa-solid fa-film text-4xl text-gray-500"></i>'
                ret, first_frame = cap.read()

                if ret:
                    _, buffer = cv2.imencode('.jpg', first_frame)
                    b64_img = base64.b64encode(buffer).decode('utf-8')
                    media_html = f'<img src="data:image/jpeg;base64,{b64_img}" style="width:100%; height:100%; object-fit:cover;">'
                cap.release()

                if total_frames <= 0:

                    st.error(f"❌ Video contains no readable frames: {file.name}")
                    os.unlink(tfile.name)
                    continue
                cap = cv2.VideoCapture(tfile.name)
                num_frames = 50
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                raw_scores = []

                for f_idx, idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()

                    if not ret: continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    detection_result = face_detector.detect(mp_img)

                    if not detection_result.detections: continue
                    best_det = max(detection_result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)

                    if best_det.bounding_box.width < 50 or best_det.bounding_box.height < 50: continue
                    face_crop = get_face_crop(frame_rgb, best_det, margin=0.20)

                    if face_crop.size == 0: continue
                    pil_img = Image.fromarray(face_crop)
                    input_tensor = test_transforms(pil_img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model_video(input_tensor)
                        prob_fake = torch.softmax(output, dim=1)[0, 0].item()
                        raw_scores.append(prob_fake)
                    base_progress = i / total_files
                    file_progress = ((f_idx + 1) / num_frames) * (1 / total_files)
                    progress_bar.progress(min(base_progress + file_progress, 1.0))
                cap.release()
                os.unlink(tfile.name)

                if len(raw_scores) == 0:

                    st.error(f"❌ No detectable faces found to analyze: {file.name}")
                    continue
                mean_score = float(np.mean(raw_scores))
                fake_ratio = sum(1 for s in raw_scores if s >= 0.50) / len(raw_scores)
                top_20_percent = sorted(raw_scores, reverse=True)[:max(1, len(raw_scores) // 5)]
                top_20_mean = float(np.mean(top_20_percent))
                max_consec = 0
                curr_consec = 0

                for s in raw_scores:

                    if s >= 0.50:
                        curr_consec += 1
                        max_consec = max(max_consec, curr_consec)

                    else:
                        curr_consec = 0
                is_fake = False
                reason = "Authentic Media"

                if mean_score >= 0.40:
                    is_fake = True
                    reason = "High Deepfake Probability (Mean)"

                elif max_consec >= 4:
                    is_fake = True
                    reason = "Consecutive Fake Frames Detected"

                elif mean_score >= 0.33 and top_20_mean >= 0.46:
                    is_fake = True
                    reason = "Suspicious Spikes in Face Analysis"

                elif mean_score < 0.25 and top_20_mean >= 0.46 and fake_ratio >= 0.04:
                    is_fake = True
                    reason = "Clear Spikes + High Fake Ratio"

                elif top_20_mean >= 0.50 and fake_ratio >= 0.10 and mean_score >= 0.30:
                    is_fake = True
                    reason = "High Fake Peaks"

                elif fake_ratio >= 0.15 and max_consec >= 4:
                    is_fake = True
                    reason = "Consecutive Pattern Detected"
                c = "fake" if is_fake else "real"
                icon = '<i class="fa-solid fa-triangle-exclamation mr-1"></i> FAKE' if is_fake else '<i class="fa-solid fa-check-circle mr-1"></i> REAL'
                conf = max(mean_score, top_20_mean) * 100 if is_fake else (1 - mean_score) * 100
                conf = min(99.9, round(conf, 1))
                stats = f"Mean: {mean_score:.4f} | Top20%: {top_20_mean:.4f} | FakeRatio: {fake_ratio:.2%} | MaxConsec: {max_consec}"

                st.markdown(f'''
                <div class="result-card {c}">
                    <div class="media-thumbnail hidden-mobile">
                        {media_html}
                    </div>
                    <div class="file-info-container">
                        <div class="file-info">{file.name}</div>
                        <div class="file-size">{file_size_mb:.2f} MB</div>
                        <div class="status-container">
                            <div class="result-header">
                                <span class="result-status {c}">{icon}</span>
                                <span class="result-confidence {c}">{conf}%</span>
                            </div>
                            <div class="progress-bg">
                                <div class="progress-fill {c}" style="width:{conf}%;"></div>
                            </div>
                            <p class="result-reason">{reason}</p>
                            <p class="result-stats">{stats}</p>
                        </div>
                    </div>
                </div>''', unsafe_allow_html=True)
            progress_bar.progress((i + 1) / total_files)
        status_text.markdown("✅ **Analysis Complete!**")

st.markdown('<div class="custom-footer"><p>DeepVision Graduation Project</p></div>', unsafe_allow_html=True)
