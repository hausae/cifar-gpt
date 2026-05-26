# ============================================================
#  ICT120 Final Project — Step 6: Web Deployment
#  CIFAR-10 Image Classifier — Streamlit + TensorFlow/Keras
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES  = ["airplane","automobile","bird","cat","deer",
                "dog","frog","horse","ship","truck"]
CLASS_EMOJIS = {
    "airplane":"✈️","automobile":"🚗","bird":"🐦","cat":"🐱",
    "deer":"🦌","dog":"🐶","frog":"🐸","horse":"🐴","ship":"🚢","truck":"🚛"
}
IMG_SIZE    = 32
MODEL_PATH  = "cifar10_model.keras"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #F5EDE2;
    --cream:    #FBF6F0;
    --card:     #FFFFFF;
    --dark:     #2C1A0E;
    --dark2:    #3D2314;
    --brown:    #7B4A22;
    --mid:      #9B6138;
    --light:    #D4A882;
    --pale:     #EDD9C0;
    --gold:     #C9933A;
    --gold-lt:  #F0D9A8;
    --text:     #2C1A0E;
    --muted:    #9B7355;
    --border:   #E8D5BE;
    --r:        14px;
    --r-sm:     9px;
    --sh:       0 2px 12px rgba(44,26,14,0.07);
    --sh-lg:    0 6px 28px rgba(44,26,14,0.13);
}
*, *::before, *::after { box-sizing: border-box; }
* { font-family: 'Inter', sans-serif !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }
html, body,
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
}
[data-testid="stHorizontalBlock"] { gap: 14px !important; align-items: stretch !important; }
h1 { font-size:2.5rem !important; font-weight:800 !important; color:var(--dark) !important;
     line-height:1.15 !important; letter-spacing:-0.03em !important; margin:0 !important; }
h2 { font-size:1.5rem !important; font-weight:700 !important; color:var(--dark) !important; letter-spacing:-0.02em !important; }
h3 { font-size:1.05rem !important; font-weight:600 !important; color:var(--dark) !important; }
p, li { color: var(--text) !important; }
hr { border:none !important; border-top:1px solid var(--border) !important; margin:1.5rem 0 !important; }
.sec-lbl { font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
           color:var(--muted); margin-bottom:10px; display:block; }
.chip { display:inline-block; padding:4px 12px; border-radius:20px;
        font-size:11px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.chip-dark  { background:var(--dark);    color:var(--gold-lt); }
.chip-gold  { background:var(--gold-lt); color:var(--brown); border:1px solid var(--gold); }
.chip-pale  { background:var(--pale);    color:var(--brown);  border:1px solid var(--light); }
.dark-card  { background:var(--dark2); border-radius:var(--r); padding:22px 24px;
              box-shadow:var(--sh-lg); height:100%; }
.dark-card * { color:var(--pale) !important; }
.dark-card h3 { color:#fff !important; font-size:14px !important; margin-bottom:12px !important; }
.arch-row { display:flex; align-items:center; gap:10px; padding:7px 0;
            border-bottom:1px solid rgba(212,168,130,0.2); }
.arch-row:last-child { border-bottom:none; }
.arch-dot  { width:7px; height:7px; border-radius:50%; background:var(--gold); flex-shrink:0; }
.arch-name { font-size:11px; font-weight:600; color:#fff !important; width:90px; flex-shrink:0; }
.arch-val  { font-size:11px; color:var(--light) !important; font-family:'JetBrains Mono',monospace !important; }
.pred-card { background:var(--dark2); border-radius:var(--r); padding:22px 26px;
             display:flex; align-items:center; gap:20px; box-shadow:var(--sh-lg); margin-bottom:14px; }
.pred-icon  { font-size:52px; line-height:1; flex-shrink:0; }
.pred-lbl   { font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
              color:var(--light) !important; margin-bottom:4px; }
.pred-name  { font-size:30px; font-weight:800; color:#fff !important;
              text-transform:capitalize; line-height:1; }
.pred-conf  { font-size:14px; font-weight:500; color:var(--gold-lt) !important; margin-top:6px; }
.cbar { margin-bottom:9px; }
.cbar-head  { display:flex; justify-content:space-between; margin-bottom:3px; }
.cbar-cls   { font-size:12px; font-weight:500; color:var(--text) !important; text-transform:capitalize; }
.cbar-cls-b { font-size:12px; font-weight:700; color:var(--dark) !important; text-transform:capitalize; }
.cbar-pct   { font-size:11px; font-family:'JetBrains Mono',monospace !important;
              font-weight:600; color:var(--brown) !important; }
.cbar-track { height:8px; background:var(--pale); border-radius:4px; overflow:hidden; }
.cbar-fill  { height:100%; border-radius:4px; background:linear-gradient(90deg,var(--mid),var(--gold)); }
.cbar-dim   { height:100%; border-radius:4px; background:var(--light); opacity:.4; }
[data-testid="stFileUploader"] {
    background: var(--cream) !important;
    border: 2px dashed var(--light) !important;
    border-radius: var(--r-sm) !important;
    padding: 6px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--mid) !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent !important; border:none !important; }

/* ── Fix doubled "uploadupload" button text ── */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--pale) !important;
    color: transparent !important;          /* hide all text layers first */
    border: 1px solid var(--light) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0 !important;               /* collapse any inherited text */
    box-shadow: none !important;
    position: relative !important;
    min-width: 130px !important;
    padding: 8px 18px !important;
}
/* Re-inject the label cleanly via ::after so it only appears once */
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Browse files";
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--dark) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--light) !important; }
[data-testid="stFileUploaderDropzone"] button:hover::after { color: var(--dark) !important; }

/* ── Dropzone instruction text ── */
[data-testid="stFileUploaderDropzoneInstructions"] { text-align: center !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--muted) !important;
    font-size: 12px !important;
}

/* ── Uploaded file chip ── */
[data-testid="stFileUploaderFile"] {
    background: var(--pale) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stFileUploaderFile"] * { color: var(--dark) !important; }
[data-testid="stFileUploaderDeleteBtn"] button {
    background: transparent !important;
    border: none !important;
    font-size: 0 !important;               /* hide any duplicate × text */
}
[data-testid="stFileUploaderDeleteBtn"] button::after {
    content: "✕";
    font-size: 13px !important;
    color: var(--muted) !important;
}
[data-testid="stMetric"] { background:var(--card) !important; border:1px solid var(--border) !important;
    border-radius:var(--r-sm) !important; padding:12px 16px !important; box-shadow:var(--sh) !important; }
[data-testid="stMetricLabel"] p { color:var(--muted) !important; font-size:10px !important;
    font-weight:700 !important; text-transform:uppercase !important; letter-spacing:.06em !important; }
[data-testid="stMetricValue"] * { color:var(--dark) !important; font-weight:800 !important; }
[data-testid="stProgressBar"] > div { background:var(--pale) !important; border-radius:4px !important; }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--mid),var(--gold)) !important; border-radius:4px !important; }
[data-testid="stAlert"] { border-radius:var(--r-sm) !important; }
.stSuccess { background:#f2ede6 !important; border-left-color:var(--gold) !important; }
.stSuccess * { color:var(--dark) !important; }
.stError    { background:#fdf0ea !important; border-left-color:#b85c38 !important; }
.stInfo     { background:var(--cream) !important; border-left-color:var(--light) !important; }
.stInfo *   { color:var(--muted) !important; }
[data-testid="stExpander"] { background:var(--cream) !important;
    border:1px solid var(--border) !important; border-radius:var(--r-sm) !important; }
table { border-collapse:collapse; width:100%; }
th { background:var(--pale) !important; color:var(--dark) !important; padding:8px 14px;
     font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
     border:1px solid var(--border); }
td { padding:7px 14px; border:1px solid var(--border); font-size:12px; color:var(--text) !important; }
tr:nth-child(even) td { background:var(--cream) !important; }
.stCaption p { color:var(--muted) !important; font-size:11px !important; }
.vok  { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px;
        font-weight:700; background:var(--pale); color:var(--brown); border:1px solid var(--light); }
.verr { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px;
        font-weight:700; background:#fce8e0; color:#7a2e10; border:1px solid #e8b4a0; }
[data-testid="stImage"] img { border-radius:var(--r-sm) !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loader ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file `{MODEL_PATH}` not found. Make sure it is uploaded in the same folder as app.py."
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 32, 32, 3)

def predict(model, pil_img):
    tensor = preprocess(pil_img)
    probs = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs


# ════════════════════════════════════════════════════════════
#  HERO ROW
# ════════════════════════════════════════════════════════════
col_hero, col_arch = st.columns([1.55, 1], gap="large")
with col_hero:
    st.markdown("""
<div style="background:#fff;border:1px solid var(--border);border-radius:var(--r);
            padding:26px 28px;box-shadow:var(--sh);height:100%">
  <span class="sec-lbl">ICT120 Final Project</span>
  <h1>Your CIFAR&#8209;10<br>Image Classifier</h1>
  <p style="font-size:13.5px;color:var(--muted);line-height:1.65;max-width:400px;margin:10px 0 18px">
    Upload any image. The CNN resizes it to 32×32, normalizes pixels
    to [0,1], and predicts one of 10 object classes with a confidence score.
  </p>
  <div style="display:flex;gap:7px;flex-wrap:wrap">
    <span class="chip chip-dark">CNN · TensorFlow Keras</span>
    <span class="chip chip-gold">20 Epochs · Adam</span>
    <span class="chip chip-pale">10 Classes</span>
    <span class="chip chip-pale">Batch 64</span>
  </div>
</div>""", unsafe_allow_html=True)

with col_arch:
    st.markdown("""
<div class="dark-card">
  <h3>Model Architecture</h3>
  <div class="arch-row"><div class="arch-dot"></div><div class="arch-name">Input</div><div class="arch-val">32 × 32 × 3</div></div>
  <div class="arch-row"><div class="arch-dot"></div><div class="arch-name">Conv2D</div><div class="arch-val">32 filters, 3×3, ReLU</div></div>
  <div class="arch-row"><div class="arch-dot"></div><div class="arch-name">MaxPool</div><div class="arch-val">2×2</div></div>
  <div class="arch-row"><div class="arch-dot"></div><div class="arch-name">Conv2D</div><div class="arch-val">64 filters, 3×3, ReLU</div></div>
  <div class="arch-row"><div class="arch-dot"></div><div class="arch-name">MaxPool</div><div class="arch-val">2×2</div></div>
  <div class="arch-row"><div class="arch-dot"></div><div class="arch-name">Dense</div><div class="arch-val">128 units, ReLU</div></div>
  <div class="arch-row" style="border:none"><div class="arch-dot" style="background:#fff"></div><div class="arch-name">Output</div><div class="arch-val">10 units, Softmax</div></div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
with st.spinner("Loading model…"):
    model, model_error = load_model()

if model_error:
    st.error(f"⚠️ **Model not loaded:** {model_error}")
    st.markdown("""
<div style="background:#fff;border:1px solid var(--border);border-radius:var(--r);padding:20px 24px;margin-top:8px">
  <span class="sec-lbl">How to fix</span>
  <p style="font-size:13px;margin-bottom:12px">
    Put <code>cifar10_model.keras</code> in the same folder as <code>app.py</code>, then redeploy.
  </p>
</div>""", unsafe_allow_html=True)
    st.stop()

# Stat strip
for col, val, lbl, dark in zip(
    st.columns(4, gap="small"),
    ["10", "32²", "20", "✓"],
    ["Classes", "Input Size", "Epochs", "Model Ready"],
    [False, False, False, True]
):
    bg = "var(--dark2)" if dark else "#fff"
    vc = "#fff"         if dark else "var(--dark)"
    lc = "var(--light)" if dark else "var(--muted)"
    col.markdown(f"""
<div style="background:{bg};border:1px solid var(--border);border-radius:var(--r-sm);
            padding:16px 10px;text-align:center;box-shadow:var(--sh)">
  <div style="font-size:2rem;font-weight:800;color:{vc};line-height:1">{val}</div>
  <div style="font-size:10px;font-weight:600;color:{lc};text-transform:uppercase;
              letter-spacing:.07em;margin-top:4px">{lbl}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CLASSIFY
# ════════════════════════════════════════════════════════════
st.markdown("## Classify an Image")
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

col_up, col_res = st.columns([1, 1.4], gap="large")

with col_up:
    with st.container(border=True):
        st.markdown('<span class="sec-lbl">Upload Image</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("upload", type=["jpg","jpeg","png","webp"],
                                    key="main_upload", label_visibility="collapsed")
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, use_column_width=True)
            with st.expander("🔬 32×32 preprocessed input"):
                tiny = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                st.image(tiny, width=120)
                arr = np.array(tiny, dtype="float32") / 255.0
                st.caption(f"Shape {arr.shape}  ·  Min {arr.min():.2f}  ·  Max {arr.max():.2f}  ·  Mean {arr.mean():.2f}")

with col_res:
    if not uploaded:
        st.markdown("""
<div style="background:#fff;border:1px solid var(--border);border-radius:var(--r);
            min-height:220px;display:flex;align-items:center;justify-content:center;
            flex-direction:column;gap:10px;box-shadow:var(--sh)">
  <span style="font-size:38px;opacity:.25">🧠</span>
  <p style="font-size:13px;color:var(--muted);text-align:center;max-width:200px">
    Upload an image on the left to see the prediction
  </p>
</div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Running inference…"):
            pred_class, confidence, probs = predict(model, pil_img)
        emoji = CLASS_EMOJIS[pred_class]
        st.markdown(f"""
<div class="pred-card">
  <div class="pred-icon">{emoji}</div>
  <div>
    <div class="pred-lbl">Predicted Class</div>
    <div class="pred-name">{pred_class}</div>
    <div class="pred-conf">Confidence: {confidence:.1f}%</div>
  </div>
</div>""", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<span class="sec-lbl">All Class Probabilities</span>', unsafe_allow_html=True)
            bars = ""
            for i in np.argsort(probs)[::-1]:
                cls    = CLASS_NAMES[i]
                pct    = float(probs[i]) * 100
                is_top = cls == pred_class
                bars += f"""
<div class="cbar">
  <div class="cbar-head">
    <span class="{'cbar-cls-b' if is_top else 'cbar-cls'}">{CLASS_EMOJIS[cls]} {cls}</span>
    <span class="cbar-pct">{pct:.1f}%</span>
  </div>
  <div class="cbar-track">
    <div class="{'cbar-fill' if is_top else 'cbar-dim'}" style="width:{pct}%"></div>
  </div>
</div>"""
            st.markdown(bars, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  VALIDATION RUNS
# ════════════════════════════════════════════════════════════
st.markdown("## Validation Runs")
st.markdown('<p style="font-size:13px;color:var(--muted);margin-top:-4px">Upload one sample image per class. Screenshot each prediction for your report.</p>', unsafe_allow_html=True)
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

val_labels  = ["Airplane ✈️","Automobile 🚗","Bird 🐦","Cat 🐱","Deer 🦌"]
val_classes = ["airplane","automobile","bird","cat","deer"]
val_results = []

for i, (col, label, true_cls) in enumerate(zip(st.columns(5, gap="small"), val_labels, val_classes)):
    with col:
        with st.container(border=True):
            st.markdown(f'<span class="sec-lbl">{label}</span>', unsafe_allow_html=True)
            vfile = st.file_uploader(label, type=["jpg","jpeg","png","webp"],
                                     key=f"val_{i}", label_visibility="collapsed")
            if vfile:
                vpil = Image.open(vfile)
                st.image(vpil, use_column_width=True)
                with st.spinner(""):
                    vcls, vconf, _ = predict(model, vpil)
                correct = vcls == true_cls
                st.markdown(
                    f'<span class="{"vok" if correct else "verr"}">{"✅" if correct else "❌"} {vcls.capitalize()}</span>',
                    unsafe_allow_html=True)
                st.caption(f"{vconf:.1f}%")
                val_results.append({"true":true_cls,"predicted":vcls,"confidence":vconf,"correct":correct})
            else:
                st.markdown('<div style="height:70px;display:flex;align-items:center;justify-content:center;font-size:22px;opacity:.25">📷</div>', unsafe_allow_html=True)

if val_results:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("### Validation Summary")
    n_correct = sum(r["correct"] for r in val_results)
    n_total   = len(val_results)
    mc = st.columns(3)
    mc[0].metric("Tests Run", f"{n_total} / 5")
    mc[1].metric("Correct",   n_correct)
    mc[2].metric("Accuracy",  f"{n_correct/n_total*100:.0f}%")
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    rows = ["| # | True Class | Predicted | Confidence | Result |", "|---|---|---|---|---|"]
    for j, r in enumerate(val_results, 1):
        rows.append(f"| {j} | {CLASS_EMOJIS[r['true']]} {r['true'].capitalize()} "
                    f"| {CLASS_EMOJIS.get(r['predicted'],'')} {r['predicted'].capitalize()} "
                    f"| {r['confidence']:.1f}% | {'✅ Correct' if r['correct'] else '❌ Wrong'} |")
    st.markdown("\n".join(rows))
    st.markdown(
    '<h4 style="color:#2C1A0E;">Confusion Matrix</h4>',
    unsafe_allow_html=True
)
    t_true = [r["true"] for r in val_results]
    t_pred = [r["predicted"] for r in val_results]
    u_rows = sorted(set(t_true),          key=lambda c: CLASS_NAMES.index(c))
    u_cols = sorted(set(t_true + t_pred), key=lambda c: CLASS_NAMES.index(c))
    cm = ["| True \\ Pred | " + " | ".join(c.capitalize() for c in u_cols) + " |",
          "|---| " + " | ".join("---" for _ in u_cols) + " |"]
    for tr in u_rows:
        cells = []
        for pr in u_cols:
            cnt = sum(1 for r in val_results if r["true"]==tr and r["predicted"]==pr)
            cells.append((f"**{cnt}** ✅" if tr==pr else f"**{cnt}** ❌") if cnt else "·")
        cm.append(f"| {CLASS_EMOJIS[tr]} {tr.capitalize()} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(cm))

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="font-size:11px;color:var(--muted);text-align:center">ICT120 Final Project · CIFAR-10 CNN Image Classifier · Step 6 Web Deployment</p>', unsafe_allow_html=True)
