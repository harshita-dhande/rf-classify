"""
dashboard.py
Ultra-Modern Management Dashboard UI for 2.4 GHz RF Spectrum Monitoring:
- Clean, professional typography (Google Font 'Plus Jakarta Sans').
- Zero emojis in icons - uses clean SVG line art and modern Dribbble-inspired UI components.
- Lime Green (#dcf763), Dark Charcoal (#141518), and Soft Cream (#e9edec) color scheme.
- Full Manual Input Support: Text paste (I, Q or complex), File Upload (.mat, .csv, .npy), Signal Examples Library.
- Replaces dropdown menu with tactile Preset Example selectors and manual stream editors.
"""

import os
import re
import io
import time
import importlib
import numpy as np
import scipy.signal as signal
import scipy.io as sio
import streamlit as st
import matplotlib.pyplot as plt

# Import data generator and ONNX session exporter
data_gen = importlib.import_module("01_data_generator")
stft_conv = importlib.import_module("02_stft_converter")
import onnxruntime as ort

CLASSES = data_gen.CLASSES
FS = data_gen.FS
SIGNAL_LENGTH = data_gen.SIGNAL_LENGTH

# Page Configuration
st.set_page_config(
    page_title="2.4 GHz RF Spectrum Monitor & Manual Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Light & Lime Theme matching reference design
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Main Background & Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #e9edec !important;
        color: #141518 !important;
    }
    
    /* Monospace for code & numbers */
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hide Default Streamlit Chrome Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Container Padding */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1560px !important;
    }

    /* Top Title Header Styling */
    .main-header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #141518;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }

    /* Cards Layout Base */
    .custom-card {
        background-color: #ffffff;
        border-radius: 22px;
        padding: 22px;
        border: 1px solid #e1e6e4;
        box-shadow: 0 4px 16px rgba(0,0,0,0.02);
        margin-bottom: 16px;
    }

    .custom-card-lime {
        background-color: #dcf763;
        border: 1px solid #cbe64d;
        color: #141518;
    }

    .custom-card-dark {
        background-color: #141518;
        color: #ffffff;
        border: 1px solid #141518;
    }

    /* Card Metrics Styling */
    .card-label {
        font-size: 0.80rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #687082;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .card-label-dark {
        color: #9da4b4;
    }

    .metric-value-large {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
        margin-bottom: 6px;
    }

    .metric-subtitle {
        font-size: 0.85rem;
        font-weight: 500;
        color: #525866;
    }

    /* Status Pill Badges */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        background-color: rgba(20,21,24,0.08);
        color: #141518;
    }

    .status-pill-lime {
        background-color: #141518;
        color: #dcf763;
    }
    
    .status-pill-dark {
        background-color: #2e3138;
        color: #dcf763;
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #141518 !important;
        border-right: none !important;
        padding: 1.2rem 1rem !important;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #9da4b4 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }

    /* Sidebar Textarea Styling */
    section[data-testid="stSidebar"] textarea {
        background-color: #202227 !important;
        border: 1px solid #2e3138 !important;
        border-radius: 12px !important;
        color: #dcf763 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* Sidebar Radio / Tab pills */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        background-color: #202227;
        padding: 6px;
        border-radius: 16px;
        border: 1px solid #2e3138;
    }

    /* Sidebar Slider */
    section[data-testid="stSidebar"] div[data-testid="stSliderTickBarMinMax"] span,
    section[data-testid="stSidebar"] div[data-testid="stSliderThumbValue"] {
        color: #dcf763 !important;
        font-weight: 700 !important;
    }

    /* Sidebar Action Button */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #dcf763 !important;
        color: #141518 !important;
        border: 1px solid #cbe64d !important;
        border-radius: 30px !important;
        font-weight: 800 !important;
        font-size: 0.90rem !important;
        padding: 10px 16px !important;
        box-shadow: 0 4px 12px rgba(220, 247, 99, 0.25) !important;
        width: 100% !important;
        transition: transform 0.15s ease !important;
    }
    
    section[data-testid="stSidebar"] div.stButton > button * {
        color: #141518 !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover {
        transform: translateY(-2px) !important;
        background-color: #e5f979 !important;
    }

    .sidebar-logo-box {
        background-color: #202227;
        border-radius: 18px;
        padding: 16px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #2e3138;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .sidebar-logo-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #2e3138;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 8px;
    }

    /* Custom Progress Bar Meters */
    .meter-container {
        margin-bottom: 12px;
    }

    .meter-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .meter-track {
        height: 10px;
        background-color: #f0f3f2;
        border-radius: 10px;
        overflow: hidden;
    }

    .meter-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.4s ease;
    }

    /* Example Signal Card */
    .example-sig-card {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 18px;
        border: 1px solid #e1e6e4;
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .example-sig-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.04);
    }

    /* Resource Action Cards */
    .resource-card {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 16px;
        border: 1px solid #e1e6e4;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .resource-icon-circle {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background-color: #f4f6f5;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* Streamlit Tabs Styling */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        margin-bottom: 18px !important;
    }

    div[data-baseweb="tab"] {
        background-color: #ffffff !important;
        border: 1px solid #e1e6e4 !important;
        border-radius: 30px !important;
        padding: 8px 20px !important;
        font-weight: 700 !important;
        font-size: 0.90rem !important;
        color: #525866 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }

    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #141518 !important;
        color: #ffffff !important;
        border-color: #141518 !important;
        box-shadow: 0 4px 12px rgba(20,21,24,0.15) !important;
    }
    
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_onnx_session(onnx_path="exports/stft_radn_model.onnx"):
    """Loads ONNX Runtime session."""
    if not os.path.exists(onnx_path):
        benchmark_mod = importlib.import_module("benchmark_2d")
        benchmark_mod.export_onnx(onnx_out_path=onnx_path)
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return session


def compute_stft_tensor(iq_signal):
    """Computes normalized log-PSD STFT tensor [1, 1, 32, 33] from 1056 complex samples."""
    # Ensure exact length of 1056 samples
    if len(iq_signal) < SIGNAL_LENGTH:
        padded = np.zeros(SIGNAL_LENGTH, dtype=np.complex128)
        padded[:len(iq_signal)] = iq_signal
        iq_signal = padded
    elif len(iq_signal) > SIGNAL_LENGTH:
        iq_signal = iq_signal[:SIGNAL_LENGTH]

    kw = signal.windows.kaiser(64, beta=14.0)
    _, _, Zxx = signal.stft(
        iq_signal,
        fs=FS,
        window=kw,
        nperseg=64,
        noverlap=32,
        boundary=None,
        padded=False,
        return_onesided=False
    )
    psd_mag = np.abs(Zxx[:33, :]).T  # shape: (32, 33)
    log_psd = 10.0 * np.log10(psd_mag**2 + 1e-12)
    
    mean_val = np.mean(log_psd)
    std_val = np.std(log_psd) if np.std(log_psd) > 1e-6 else 1e-6
    norm_psd = (log_psd - mean_val) / std_val
    
    tensor = norm_psd.reshape(1, 1, 32, 33).astype(np.float32)
    return tensor, norm_psd, iq_signal


def format_iq_as_csv_text(iq_array, max_lines=1056):
    """Formats complex I/Q numpy array as readable 'I, Q' CSV text."""
    lines = ["# In-Phase (I), Quadrature (Q) Baseband Samples"]
    sub = iq_array[:max_lines]
    for val in sub:
        lines.append(f"{val.real:+.6f}, {val.imag:+.6f}")
    return "\n".join(lines)


def format_iq_as_complex_text(iq_array, max_lines=1056):
    """Formats complex I/Q numpy array as Python complex notation (e.g. 0.123-0.456j)."""
    sub = iq_array[:max_lines]
    return ", ".join([f"{val.real:+.5f}{val.imag:+.5f}j" for val in sub])


def parse_manual_iq(text):
    """
    Parses arbitrary manual text into a 1D complex numpy array.
    Supports:
    - CSV lines: I, Q
    - Space-separated numbers
    - Python complex numbers (e.g. 0.12-0.34j)
    """
    if not text or not text.strip():
        return None, "Empty input provided"

    # Clean comments starting with # or //
    clean_lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            clean_lines.append(line)
    
    cleaned_text = "\n".join(clean_lines).strip()
    if not cleaned_text:
        return None, "No numeric content found"

    # Strategy 1: Check for python complex numbers ('j')
    if "j" in cleaned_text:
        tokens = re.split(r'[\n,;]+', cleaned_text)
        complex_vals = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                complex_vals.append(complex(token))
            except ValueError:
                continue
        if len(complex_vals) >= 8:
            return np.array(complex_vals, dtype=np.complex128), f"Parsed {len(complex_vals)} complex samples (I+Qj format)"

    # Strategy 2: Check for CSV / space separated I, Q pairs
    vals_i = []
    vals_q = []
    for line in clean_lines:
        tokens = [t for t in re.split(r'[\s,;]+', line) if t.strip()]
        if len(tokens) >= 2:
            try:
                vals_i.append(float(tokens[0]))
                vals_q.append(float(tokens[1]))
            except ValueError:
                continue
        elif len(tokens) == 1:
            try:
                vals_i.append(float(tokens[0]))
                vals_q.append(0.0)
            except ValueError:
                continue

    if len(vals_i) >= 8:
        arr = np.array(vals_i, dtype=np.float64) + 1j * np.array(vals_q, dtype=np.float64)
        return arr, f"Parsed {len(arr)} (I, Q) baseband samples"

    # Strategy 3: Flat series of numbers paired into I and Q
    all_floats = []
    for token in re.split(r'[\s,;]+', cleaned_text):
        if token.strip():
            try:
                all_floats.append(float(token.strip()))
            except ValueError:
                continue
    if len(all_floats) >= 16:
        # Pair up into (I, Q)
        if len(all_floats) % 2 != 0:
            all_floats = all_floats[:-1]
        half = len(all_floats) // 2
        arr_i = np.array(all_floats[0::2])
        arr_q = np.array(all_floats[1::2])
        return arr_i + 1j * arr_q, f"Parsed {len(arr_i)} alternating (I, Q) samples"

    return None, "Unrecognized format. Please use 'I, Q' CSV lines or complex '0.1-0.2j' notation."


@st.cache_data
def get_canonical_examples():
    """Generates and loads deterministic canonical examples of each RF signal."""
    np.random.seed(42)
    examples = {
        "Wi-Fi (802.11 OFDM)": {
            "sig": data_gen.generate_wifi_signal(),
            "desc": "Wideband 64-subcarrier OFDM baseband with 16-sample cyclic prefix (16.6 MHz BW, 20 MSPS).",
            "standard": "IEEE 802.11a/g/n",
            "category": "Single Signal"
        },
        "Bluetooth (BLE GFSK)": {
            "sig": data_gen.generate_bluetooth_signal(),
            "desc": "Narrowband continuous-phase GFSK burst (1 Mbps, BT=0.5) with carrier frequency hopping offset.",
            "standard": "IEEE 802.15.1",
            "category": "Single Signal"
        },
        "Zigbee (802.15.4 O-QPSK)": {
            "sig": data_gen.generate_zigbee_signal(),
            "desc": "Offset QPSK with half-sine pulse shaping and 2.0 Mchips/s DSSS pseudo-random spreading.",
            "standard": "IEEE 802.15.4",
            "category": "Single Signal"
        },
        "SmartBAN (802.15.6)": {
            "sig": data_gen.generate_smartban_signal(),
            "desc": "Medical Body Area Network duty-cycled short pulses (500 ksps) with inter-burst silence intervals.",
            "standard": "IEEE 802.15.6",
            "category": "Single Signal"
        },
        "Unknown / AWGN Noise": {
            "sig": data_gen.generate_noise_signal(),
            "desc": "Complex Additive White Gaussian Noise (AWGN) representing clean thermal noise floor.",
            "standard": "Thermal Noise Floor",
            "category": "Single Signal"
        }
    }

    # Load recorded overlapping examples if available
    mat_wifi_bt = "data/examples/sample_wifi_bt_overlap.mat"
    if os.path.exists(mat_wifi_bt):
        m = sio.loadmat(mat_wifi_bt)
        examples["Wi-Fi + BT Overlap (Scenario S4)"] = {
            "sig": m['iqSignal'].flatten(),
            "desc": "Real multi-signal collision: Bluetooth hops into active Wi-Fi outer subcarriers (Scenario S4).",
            "standard": "Wi-Fi + Bluetooth Coexistence",
            "category": "Overlapping Mixture"
        }

    mat_wifi_zigbee = "data/examples/sample_wifi_zigbee_overlap.mat"
    if os.path.exists(mat_wifi_zigbee):
        m = sio.loadmat(mat_wifi_zigbee)
        examples["Wi-Fi + Zigbee Overlap (Scenario S5)"] = {
            "sig": m['iqSignal'].flatten(),
            "desc": "Real multi-signal collision: Zigbee DSSS transmission colliding inside active Wi-Fi channel (Scenario S5).",
            "standard": "Wi-Fi + Zigbee Coexistence",
            "category": "Overlapping Mixture"
        }

    return examples


def main():
    # Load ONNX Inference Session
    session = load_onnx_session()
    input_name = session.get_inputs()[0].name

    # Load Canonical Signal Examples
    canonical_examples = get_canonical_examples()

    # Initialize Session State
    if "iq_signal" not in st.session_state:
        default_preset = "Wi-Fi (802.11 OFDM)"
        sig_data = canonical_examples[default_preset]["sig"]
        st.session_state["iq_signal"] = sig_data
        st.session_state["signal_source_title"] = default_preset
        st.session_state["manual_iq_text"] = format_iq_as_csv_text(sig_data)
        st.session_state["snr_add_db"] = 0.0

    # -------------------------------------------------------------
    # SIDEBAR: Manual Interaction & Signal Ingestion (NO DROPDOWN)
    # -------------------------------------------------------------
    st.sidebar.markdown("""
    <div class="sidebar-logo-box">
        <div class="sidebar-logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#dcf763" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2v20M2 12h20M4.93 4.93l14.14 14.14M4.93 19.07l14.14-14.14"/>
            </svg>
        </div>
        <div style="font-weight: 800; font-size: 1.05rem; letter-spacing: -0.02em; color: #ffffff;">RF MONITOR</div>
        <div style="font-size: 0.75rem; color: #9da4b4; margin-top: 2px;">STFT-RADN Engine v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. PRESET SIGNAL SELECTORS (REPLACES DROPDOWN MENU)
    st.sidebar.markdown("<div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #9da4b4; margin-bottom: 8px;'>1. LOAD PRESET SIGNAL EXAMPLES</div>", unsafe_allow_html=True)
    st.sidebar.caption("Click to load the exact raw I/Q samples of any protocol:")

    # Quick preset button grid
    preset_keys = list(canonical_examples.keys())
    for p_key in preset_keys:
        is_selected = (st.session_state.get("signal_source_title") == p_key)
        btn_label = f"✓ {p_key}" if is_selected else p_key
        if st.sidebar.button(btn_label, key=f"btn_preset_{p_key}", use_container_width=True):
            loaded_sig = canonical_examples[p_key]["sig"]
            st.session_state["iq_signal"] = loaded_sig
            st.session_state["signal_source_title"] = p_key
            st.session_state["manual_iq_text"] = format_iq_as_csv_text(loaded_sig)
            st.rerun()

    st.sidebar.markdown("<hr style='border-color: #2e3138; margin: 18px 0;'>", unsafe_allow_html=True)

    # 2. MANUAL I/Q STREAM INPUT
    st.sidebar.markdown("<div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #9da4b4; margin-bottom: 8px;'>2. MANUAL I/Q STREAM INPUT</div>", unsafe_allow_html=True)

    input_mode = st.sidebar.radio(
        "Input Mode",
        options=["Direct Text / Paste", "File Upload (.mat, .csv, .npy)", "Tone Synthesizer"],
        index=0,
        label_visibility="collapsed"
    )

    if input_mode == "Direct Text / Paste":
        st.sidebar.caption("Edit or paste I/Q numbers directly (format: `I, Q` or `I+Qj`):")
        manual_text = st.sidebar.text_area(
            "Raw I/Q Stream",
            value=st.session_state.get("manual_iq_text", ""),
            height=200,
            key="textarea_iq"
        )
        
        # Sample counter & format tools
        parsed_sig, parse_msg = parse_manual_iq(manual_text)
        if parsed_sig is not None:
            n_samples = len(parsed_sig)
            badge_color = "#dcf763" if n_samples == SIGNAL_LENGTH else "#ffd166"
            st.sidebar.markdown(f"<div style='font-size: 0.75rem; color: {badge_color}; font-weight: 700;'>• {parse_msg}</div>", unsafe_allow_html=True)
            if n_samples != SIGNAL_LENGTH:
                st.sidebar.caption(f"Note: Will be auto-adjusted to {SIGNAL_LENGTH} samples.")
        else:
            st.sidebar.markdown(f"<div style='font-size: 0.75rem; color: #ff6b6b; font-weight: 700;'>• {parse_msg}</div>", unsafe_allow_html=True)

    elif input_mode == "File Upload (.mat, .csv, .npy)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload I/Q capture file",
            type=["mat", "csv", "npy", "txt"],
            key="iq_file_uploader"
        )
        if uploaded_file is not None:
            fname = uploaded_file.name.lower()
            try:
                if fname.endswith(".mat"):
                    mat_data = sio.loadmat(uploaded_file)
                    # Search for key with complex array
                    candidate_keys = [k for k in mat_data.keys() if not k.startswith("__")]
                    chosen_key = "iqSignal" if "iqSignal" in candidate_keys else candidate_keys[0]
                    parsed_sig = mat_data[chosen_key].flatten()
                    st.session_state["signal_source_title"] = f"Uploaded MAT: {uploaded_file.name}"
                elif fname.endswith(".npy"):
                    parsed_sig = np.load(uploaded_file).flatten()
                    st.session_state["signal_source_title"] = f"Uploaded NPY: {uploaded_file.name}"
                else:  # CSV / TXT
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    parsed_sig, parse_msg = parse_manual_iq(content)
                    st.session_state["signal_source_title"] = f"Uploaded CSV: {uploaded_file.name}"

                if parsed_sig is not None and len(parsed_sig) > 0:
                    st.session_state["iq_signal"] = parsed_sig
                    st.session_state["manual_iq_text"] = format_iq_as_csv_text(parsed_sig)
                    st.sidebar.success(f"Loaded {len(parsed_sig)} samples from {uploaded_file.name}!")
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")

    else:  # Tone Synthesizer
        st.sidebar.caption("Generate a custom test tone / continuous wave (CW):")
        f_offset_mhz = st.sidebar.slider("Tone Frequency (MHz)", min_value=-8.0, max_value=8.0, value=2.0, step=0.5)
        phase_deg = st.sidebar.slider("Initial Phase (°)", min_value=0, max_value=360, value=0, step=15)
        if st.sidebar.button("Generate Custom CW Tone", use_container_width=True):
            t = np.arange(SIGNAL_LENGTH) / FS
            cw_sig = np.exp(1j * (2 * np.pi * f_offset_mhz * 1e6 * t + np.deg2rad(phase_deg)))
            st.session_state["iq_signal"] = cw_sig
            st.session_state["signal_source_title"] = f"Custom CW Tone ({f_offset_mhz:+.1f} MHz)"
            st.session_state["manual_iq_text"] = format_iq_as_csv_text(cw_sig)
            st.rerun()

    st.sidebar.markdown("<hr style='border-color: #2e3138; margin: 18px 0;'>", unsafe_allow_html=True)

    # 3. CHANNEL IMPAIRMENTS
    st.sidebar.markdown("<div style='font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #9da4b4; margin-bottom: 8px;'>3. CHANNEL NOISE INJECTION</div>", unsafe_allow_html=True)
    apply_noise = st.sidebar.checkbox("Add AWGN Channel Noise", value=False)
    if apply_noise:
        snr_val = st.sidebar.slider("Injected SNR (dB)", min_value=-20.0, max_value=10.0, value=-5.0, step=1.0)
    else:
        snr_val = 999.0  # Clean

    # 4. PRIMARY INFERENCE TRIGGER BUTTON
    st.sidebar.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    classify_btn = st.sidebar.button("⚡ Classify Manual Input", use_container_width=True)

    # Update active I/Q signal if classify button is clicked in Text Mode
    if classify_btn and input_mode == "Direct Text / Paste":
        parsed_sig, parse_msg = parse_manual_iq(manual_text)
        if parsed_sig is not None and len(parsed_sig) >= 8:
            st.session_state["iq_signal"] = parsed_sig
            st.session_state["signal_source_title"] = "Custom Manual Text Input"
            st.session_state["manual_iq_text"] = manual_text

    # Prepare active signal
    active_raw_sig = st.session_state.get("iq_signal")
    if active_raw_sig is None or len(active_raw_sig) == 0:
        active_raw_sig = canonical_examples["Wi-Fi (802.11 OFDM)"]["sig"]

    # Optionally inject AWGN
    if apply_noise and snr_val < 100.0:
        working_sig = data_gen.inject_awgn_and_cfo(active_raw_sig, snr_val)
    else:
        working_sig = active_raw_sig

    # Compute Kaiser-windowed STFT tensor [1, 1, 32, 33]
    tensor_input, spectrogram_2d, final_iq_1056 = compute_stft_tensor(working_sig)

    # Execute ONNX Inference
    start_t = time.perf_counter()
    logits = session.run(None, {input_name: tensor_input})[0][0]
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    pred_idx = int(np.argmax(probs))
    pred_class_name = CLASSES[pred_idx]
    confidence_pct = probs[pred_idx] * 100.0

    source_title = st.session_state.get("signal_source_title", "Custom Manual Input")

    # -------------------------------------------------------------
    # MAIN HEADER & TITLE BANNER
    # -------------------------------------------------------------
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
        <div>
            <div class="main-header-title">
                2.4 GHz RF Spectrum Monitor & Manual Classifier
            </div>
            <div style="color: #687082; font-size: 0.95rem; font-weight: 500;">
                Edge-Optimized STFT-RADN Deep Learning Model with Live Manual Sample Ingestion
            </div>
        </div>
        <div style="display: flex; gap: 10px; margin-top: 8px;">
            <div style="background: #ffffff; color: #141518; padding: 8px 16px; border-radius: 30px; font-weight: 700; font-size: 0.82rem; border: 1px solid #e1e6e4; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                ONNX CPU Engine
            </div>
            <div style="background: #ffffff; color: #141518; padding: 8px 16px; border-radius: 30px; font-weight: 700; font-size: 0.82rem; border: 1px solid #e1e6e4;">
                N = 1,056 Baseband Samples
            </div>
            <div style="background: #141518; color: #dcf763; padding: 8px 18px; border-radius: 30px; font-weight: 700; font-size: 0.82rem;">
                Manual Ingestion Active
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TOP 3 HERO CARDS
    # -------------------------------------------------------------
    c1, c2, c3 = st.columns([1.1, 1, 1])

    with c1:
        st.markdown(f"""
        <div class="custom-card custom-card-lime">
            <div class="card-label">
                <span>PREDICTED STANDARD</span>
                <span class="status-pill status-pill-lime">{confidence_pct:.1f}% CONFIDENCE</span>
            </div>
            <div class="metric-value-large">{pred_class_name}</div>
            <div class="metric-subtitle">
                Active Source: <strong>{source_title}</strong>
            </div>
            <div style="margin-top: 14px; height: 8px; background: rgba(20,21,24,0.12); border-radius: 10px;">
                <div style="height: 100%; width: {confidence_pct:.1f}%; background: #141518; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        pwr_val = np.mean(np.abs(final_iq_1056)**2)
        papr_db = 10 * np.log10(np.max(np.abs(final_iq_1056)**2) / (pwr_val + 1e-12))
        noise_str = f"{snr_val:.1f} dB SNR" if apply_noise else "Clean / Unimpaired"
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">
                <span>SIGNAL POWER & PAPR</span>
                <span class="status-pill">{noise_str}</span>
            </div>
            <div class="metric-value-large">{papr_db:.1f} <span style="font-size: 1.1rem; font-weight: 600; color: #687082;">dB PAPR</span></div>
            <div class="metric-subtitle">
                Mean Power: <strong>{pwr_val:.3f}</strong> | Sample Rate: <strong>20 MSPS</strong>
            </div>
            <div style="margin-top: 14px; height: 8px; background: #f0f3f2; border-radius: 10px;">
                <div style="height: 100%; width: {min(100.0, (papr_db / 15.0) * 100):.1f}%; background: #dcf763; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="custom-card custom-card-dark">
            <div class="card-label card-label-dark">
                <span>STFT-RADN LATENCY</span>
                <span style="color: #dcf763; font-weight: 700; font-size: 0.8rem;">~10.6 ms TARGET</span>
            </div>
            <div class="metric-value-large" style="color: #ffffff;">{latency_ms:.2f} <span style="font-size: 1.1rem; font-weight: 600; color: #dcf763;">ms</span></div>
            <div class="metric-subtitle" style="color: #9da4b4;">
                Model: <strong>852,653</strong> Params | Kaiser β=14.0
            </div>
            <div style="margin-top: 14px; height: 8px; background: #2a2c32; border-radius: 10px;">
                <div style="height: 100%; width: {min(100.0, (latency_ms / 10.6) * 100):.1f}%; background: #dcf763; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # MAIN TABS: 1. Inference & Spectrum, 2. Examples Library, 3. Raw Data Inspector
    # -------------------------------------------------------------
    tab_infer, tab_examples, tab_inspector = st.tabs([
        "📊 1. Inference & Spectrum Monitoring",
        "🔬 2. Input Signal Examples & Anatomy",
        "📋 3. Raw Input Data Inspector & Exporter"
    ])

    # =============================================================
    # TAB 1: INFERENCE & SPECTRUM MONITORING
    # =============================================================
    with tab_infer:
        col_left, col_right = st.columns([1.75, 1.0], gap="medium")

        with col_left:
            st.markdown("""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.02em; color: #141518;">
                    Kaiser-Windowed 2D STFT Spectrogram & Waveform
                </div>
                <div style="font-size: 0.80rem; font-weight: 600; color: #687082; background: #ffffff; padding: 4px 12px; border-radius: 20px; border: 1px solid #e1e6e4;">
                    Input Tensor: [1, 1, 32, 33]
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Matplotlib figure with Spectrogram, 1D Waveform, and FFT Spectrum
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 7.8), gridspec_kw={'height_ratios': [2.2, 1.0, 1.0]})
            fig.patch.set_facecolor('#ffffff')

            # 1. 2D STFT Spectrogram
            ax1.set_facecolor('#ffffff')
            im = ax1.imshow(
                spectrogram_2d.T,
                aspect='auto',
                origin='lower',
                cmap='inferno',
                extent=[0, 32, -10, 10]
            )
            ax1.set_title(f"Log-PSD Spectrogram (Kaiser β=14.0) | {source_title}", color='#141518', fontsize=10.5, fontweight='bold', pad=8)
            ax1.set_xlabel("Time Frames (32)", color='#687082', fontsize=8.5, fontweight='medium')
            ax1.set_ylabel("Frequency (MHz)", color='#687082', fontsize=8.5, fontweight='medium')
            ax1.tick_params(colors='#141518', labelsize=8)
            for spine in ax1.spines.values():
                spine.set_color('#e1e6e4')

            cbar = fig.colorbar(im, ax=ax1, orientation='vertical', pad=0.02)
            cbar.ax.tick_params(colors='#687082', labelsize=7.5)
            cbar.set_label('Norm Log-PSD (dB)', color='#687082', fontsize=8)

            # 2. 1D Complex Baseband Waveform (I & Q)
            ax2.set_facecolor('#ffffff')
            t_us = np.arange(len(final_iq_1056)) / (FS / 1e6)
            ax2.plot(t_us, final_iq_1056.real, label="I Channel (Real)", color="#141518", linewidth=1.1, alpha=0.9)
            ax2.plot(t_us, final_iq_1056.imag, label="Q Channel (Imag)", color="#b0d418", linewidth=1.1, alpha=0.85)
            ax2.set_title(f"1D Baseband Waveform (N = {len(final_iq_1056)} complex samples)", color='#141518', fontsize=9.5, fontweight='bold', pad=6)
            ax2.set_xlabel("Time (µs)", color='#687082', fontsize=8.5)
            ax2.set_ylabel("Amplitude", color='#687082', fontsize=8.5)
            ax2.tick_params(colors='#141518', labelsize=8)
            ax2.legend(loc="upper right", facecolor="#ffffff", edgecolor="#e1e6e4", labelcolor="#141518", fontsize=7.5)
            for spine in ax2.spines.values():
                spine.set_color('#e1e6e4')

            # 3. FFT Power Spectral Density
            ax3.set_facecolor('#ffffff')
            fft_mag = np.abs(np.fft.fftshift(np.fft.fft(final_iq_1056)))**2
            fft_db = 10 * np.log10(fft_mag + 1e-12)
            fft_db -= np.max(fft_db)  # normalize to 0 dB peak
            freq_mhz = np.fft.fftshift(np.fft.fftfreq(len(final_iq_1056), d=1/FS)) / 1e6
            ax3.plot(freq_mhz, fft_db, color="#141518", linewidth=1.0)
            ax3.fill_between(freq_mhz, -60, fft_db, color="#dcf763", alpha=0.35)
            ax3.set_ylim([-55, 5])
            ax3.set_title("Baseband Power Spectrum (FFT)", color='#141518', fontsize=9.5, fontweight='bold', pad=6)
            ax3.set_xlabel("Baseband Frequency (MHz)", color='#687082', fontsize=8.5)
            ax3.set_ylabel("Relative Power (dB)", color='#687082', fontsize=8.5)
            ax3.tick_params(colors='#141518', labelsize=8)
            for spine in ax3.spines.values():
                spine.set_color('#e1e6e4')

            plt.tight_layout()
            st.pyplot(fig)

        with col_right:
            st.markdown("""
            <div style="font-size: 1.15rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 12px; color: #141518;">
                Classification Probabilities
            </div>
            """, unsafe_allow_html=True)

            for idx, c_name in enumerate(CLASSES):
                prob = float(probs[idx])
                pct = prob * 100.0
                is_top = (idx == pred_idx)
                
                bg_color = "#dcf763" if is_top else "#141518"
                text_style = "font-weight: 800; color: #141518;" if is_top else "font-weight: 600; color: #525866;"
                badge = f"<span style='background: #dcf763; color: #141518; font-size: 0.72rem; padding: 2px 8px; border-radius: 12px; font-weight: 800; margin-left: 6px;'>PREDICTED</span>" if is_top else ""
                
                st.markdown(f"""
                <div class="meter-container" style="background: #ffffff; padding: 12px 16px; border-radius: 16px; border: 1px solid #e1e6e4;">
                    <div class="meter-header" style="{text_style}">
                        <span>{c_name} {badge}</span>
                        <span>{pct:.1f}%</span>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill" style="width: {pct:.1f}%; background-color: {bg_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Manual Quick Ingestion Assistant Card
            st.markdown(f"""
            <div class="custom-card" style="margin-top: 14px; padding: 18px;">
                <div style="font-weight: 700; font-size: 0.90rem; color: #141518; margin-bottom: 6px;">
                    Active Ingestion Pipeline
                </div>
                <div style="font-size: 0.80rem; color: #525866; line-height: 1.5;">
                    Current input was ingested via <strong>{source_title}</strong>. 
                    The STFT-RADN model received <strong>{len(final_iq_1056)}</strong> complex time-domain samples, 
                    computed a <strong>32×33</strong> Kaiser STFT log-PSD matrix, and inferred the PHY class in <strong>{latency_ms:.2f} ms</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # =============================================================
    # TAB 2: INPUT SIGNAL EXAMPLES & ANATOMY
    # =============================================================
    with tab_examples:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #141518; letter-spacing: -0.02em;">
                Exact Input Signals Anatomy & Reference Library
            </div>
            <div style="color: #687082; font-size: 0.90rem;">
                Here are the concrete physical inputs fed into the model for each wireless standard. 
                Each signal consists of <strong>1,056 complex baseband I/Q pairs</strong> at 20 MSPS (duration: 52.8 µs). 
                Inspect the actual numerical numbers and click <strong>Load into Manual Editor</strong> to classify or modify them.
            </div>
        </div>
        """, unsafe_allow_html=True)

        for ex_name, ex_data in canonical_examples.items():
            sig = ex_data["sig"]
            std = ex_data["standard"]
            desc = ex_data["desc"]
            cat = ex_data["category"]
            
            with st.container():
                col_info, col_table, col_btn = st.columns([1.5, 2.0, 0.8], gap="medium")
                
                with col_info:
                    cat_color = "#141518" if cat == "Single Signal" else "#c2410c"
                    st.markdown(f"""
                    <div style="background: #ffffff; padding: 18px; border-radius: 18px; border: 1px solid #e1e6e4; height: 100%;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 800; font-size: 1.05rem; color: #141518;">{ex_name}</span>
                            <span style="background: rgba(20,21,24,0.06); color: {cat_color}; font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 12px;">{std}</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #525866; line-height: 1.45; margin-bottom: 10px;">
                            {desc}
                        </div>
                        <div style="font-size: 0.78rem; color: #687082;">
                            • Length: <strong>{len(sig)} samples</strong><br>
                            • Peak Power: <strong>{np.max(np.abs(sig)**2):.2f}</strong><br>
                            • RMS Voltage: <strong>{np.sqrt(np.mean(np.abs(sig)**2)):.2f} V</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_table:
                    # Preview of first 6 raw samples in a clean table
                    preview_data = []
                    for s_i in range(min(6, len(sig))):
                        val = sig[s_i]
                        preview_data.append({
                            "Sample #": f"[{s_i}]",
                            "In-Phase (I)": f"{val.real:+.4f}",
                            "Quadrature (Q)": f"{val.imag:+.4f}",
                            "Magnitude |z|": f"{np.abs(val):.4f}",
                            "Phase (°)": f"{np.rad2deg(np.angle(val)):+.1f}°"
                        })
                    st.caption("First 6 Complex Baseband Samples Preview:")
                    st.dataframe(preview_data, use_container_width=True, hide_index=True)

                with col_btn:
                    st.markdown("<div style='margin-top: 26px;'></div>", unsafe_allow_html=True)
                    if st.button(f"📥 Load {ex_name.split()[0]}", key=f"tab2_load_{ex_name}", use_container_width=True):
                        st.session_state["iq_signal"] = sig
                        st.session_state["signal_source_title"] = ex_name
                        st.session_state["manual_iq_text"] = format_iq_as_csv_text(sig)
                        st.rerun()

                st.markdown("<hr style='border-color: #e1e6e4; margin: 14px 0;'>", unsafe_allow_html=True)

    # =============================================================
    # TAB 3: RAW INPUT DATA INSPECTOR & EXPORTER
    # =============================================================
    with tab_inspector:
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #141518;">
                Full 1,056 Baseband Samples Inspector
            </div>
            <div style="color: #687082; font-size: 0.88rem;">
                Examine, download, or copy the exact numerical payload currently residing in memory and passed into the neural network.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_full_table, col_constellation = st.columns([1.5, 1.0], gap="medium")

        with col_full_table:
            # Full table of samples
            full_table = []
            for idx_s, val in enumerate(final_iq_1056):
                full_table.append({
                    "Index": idx_s,
                    "In-Phase (I)": float(np.round(val.real, 6)),
                    "Quadrature (Q)": float(np.round(val.imag, 6)),
                    "Magnitude": float(np.round(np.abs(val), 6)),
                    "Phase (rad)": float(np.round(np.angle(val), 4))
                })
            st.dataframe(full_table, height=380, use_container_width=True)

            # Export Buttons
            csv_buf = io.StringIO()
            csv_buf.write("index,i,q,magnitude,phase_rad\n")
            for r in full_table:
                csv_buf.write(f"{r['Index']},{r['In-Phase (I)']},{r['Quadrature (Q)']},{r['Magnitude']},{r['Phase (rad)']}\n")
            
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "⬇️ Download Raw I/Q as CSV",
                    data=csv_buf.getvalue(),
                    file_name="rf_baseband_samples.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c_dl2:
                npy_buf = io.BytesIO()
                np.save(npy_buf, final_iq_1056)
                st.download_button(
                    "⬇️ Download as NumPy (.npy)",
                    data=npy_buf.getvalue(),
                    file_name="rf_baseband_samples.npy",
                    mime="application/octet-stream",
                    use_container_width=True
                )

        with col_constellation:
            st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #141518; margin-bottom: 6px;'>I/Q Constellation / Polar Phase Trajectory</div>", unsafe_allow_html=True)
            fig_iq, ax_iq = plt.subplots(figsize=(4.5, 4.2))
            fig_iq.patch.set_facecolor('#ffffff')
            ax_iq.set_facecolor('#ffffff')
            ax_iq.scatter(final_iq_1056.real, final_iq_1056.imag, c='#141518', s=8, alpha=0.6, edgecolors='none')
            ax_iq.axhline(0, color='#e1e6e4', linestyle='--', linewidth=0.8)
            ax_iq.axvline(0, color='#e1e6e4', linestyle='--', linewidth=0.8)
            ax_iq.set_xlabel("In-Phase (I)", color='#687082', fontsize=8)
            ax_iq.set_ylabel("Quadrature (Q)", color='#687082', fontsize=8)
            ax_iq.tick_params(colors='#141518', labelsize=7.5)
            for spine in ax_iq.spines.values():
                spine.set_color('#e1e6e4')
            plt.tight_layout()
            st.pyplot(fig_iq)


if __name__ == "__main__":
    main()
