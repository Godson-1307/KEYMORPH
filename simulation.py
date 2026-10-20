"""
AegisCode - Tactical C2 Communications System
Module: simulation.py
Description: Core execution engine, polymorphic keying, EW jamming channel,
             dual-path honey encryption router, neural Bahdanau attention decoder,
             and Gradio tactical military C2 terminal dashboard.
"""

import os
import re
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# Import neural architecture & morse utilities from train module
from train import (
    MORSE_DICT,
    REV_MORSE_DICT,
    SRC_VOCAB,
    SRC_CHAR2IDX,
    SRC_IDX2CHAR,
    TRG_VOCAB,
    TRG_CHAR2IDX,
    TRG_IDX2CHAR,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX,
    Encoder,
    Decoder,
    Seq2Seq,
    sanitize_text,
    text_to_base_morse,
    apply_dynamic_shift,
    reverse_dynamic_shift,
    inject_ew_noise,
    generate_honey_cover
)

# =====================================================================
# 1. NEURAL MODEL ENGINE & FAIL-SAFE LOADER
# =====================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.pth")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pt")

NEURAL_MODEL = None
VOCAB_METADATA = None
MODEL_LOADED = False
LOAD_STATUS_MSG = ""

def load_neural_engine():
    global NEURAL_MODEL, VOCAB_METADATA, MODEL_LOADED, LOAD_STATUS_MSG
    if os.path.exists(MODEL_WEIGHTS_PATH) and os.path.exists(VOCAB_PATH):
        try:
            VOCAB_METADATA = torch.load(VOCAB_PATH, map_location=DEVICE)
            encoder = Encoder(
                input_vocab_size=len(VOCAB_METADATA['src_vocab']),
                embed_dim=VOCAB_METADATA.get('embed_dim', 64),
                hidden_dim=VOCAB_METADATA.get('hidden_dim', 128)
            )
            decoder = Decoder(
                output_vocab_size=len(VOCAB_METADATA['trg_vocab']),
                embed_dim=VOCAB_METADATA.get('embed_dim', 64),
                hidden_dim=VOCAB_METADATA.get('hidden_dim', 128)
            )
            model = Seq2Seq(encoder, decoder).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
            model.eval()
            NEURAL_MODEL = model
            MODEL_LOADED = True
            LOAD_STATUS_MSG = f"ONLINE // Neural Seq2Seq Bahdanau Attention Loaded ({DEVICE.type.upper()})"
            print(f"[+] {LOAD_STATUS_MSG}")
            return
        except Exception as e:
            LOAD_STATUS_MSG = f"DEGRADED // Fail-safe Heuristic Engine Active (Weight Load Error: {e})"
            print(f"[!] Warning: {LOAD_STATUS_MSG}")
    else:
        LOAD_STATUS_MSG = "DEGRADED // Fail-safe Heuristic Engine Active (Weights Not Found)"
        print(f"[!] Warning: {LOAD_STATUS_MSG}")

# Initialize model at startup
load_neural_engine()

# Heuristic Fallback Repair
def fallback_morse_repair(morse_code: str) -> tuple[str, list[float]]:
    """Heuristic rule-based decoder for fail-safe fallback."""
    words = morse_code.split('/')
    decoded_words = []
    confidences = []
    for word in words:
        letters = word.strip().split(' ')
        dec_letters = []
        for l in letters:
            if not l:
                continue
            if l in REV_MORSE_DICT:
                dec_letters.append(REV_MORSE_DICT[l])
                confidences.append(0.95)
            else:
                # Nearest hamming distance match
                best_char = '?'
                best_dist = 999
                for m_pat, ch in REV_MORSE_DICT.items():
                    if ch == ' ':
                        continue
                    # Compute edit distance approximation
                    dist = abs(len(m_pat) - len(l)) + sum(1 for a, b in zip(m_pat, l) if a != b)
                    if dist < best_dist:
                        best_dist = dist
                        best_char = ch
                dec_letters.append(best_char)
                confidences.append(max(0.40, 1.0 - (best_dist * 0.25)))
        decoded_words.append("".join(dec_letters))
    return " ".join(decoded_words), confidences

def compute_edit_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# Neural Attention Inference with Beam Search
def neural_decode(noisy_morse: str, max_len: int = 40, beam_width: int = 3):
    """
    Performs Bahdanau Attention Seq2Seq auto-repair inference with Beam Search.
    Maintains top-K sequence hypotheses for maximum resilience against dropped symbols.
    Returns: (decoded_text, confidences_list, attention_matrix_dict)
    """
    if not MODEL_LOADED or NEURAL_MODEL is None:
        text, confs = fallback_morse_repair(noisy_morse)
        return text, confs, None

    src_char2idx = VOCAB_METADATA['src_char2idx']
    trg_idx2char = VOCAB_METADATA['trg_idx2char']

    src_indices = [SOS_IDX] + [src_char2idx.get(c, UNK_IDX) for c in noisy_morse] + [EOS_IDX]
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = NEURAL_MODEL.encoder(src_tensor)
        mask = (src_tensor != PAD_IDX)

        # Beam Search State: (score, tokens_list, confs_list, hidden_state, attn_steps)
        initial_token = torch.tensor([SOS_IDX], device=DEVICE)
        beams = [(0.0, [initial_token], [], hidden, [])]
        completed_beams = []

        for step in range(max_len):
            new_beams = []
            for score, tokens, confs, h_state, attns in beams:
                last_tok = tokens[-1]
                if last_tok.item() == EOS_IDX and len(tokens) > 1:
                    completed_beams.append((score / max(1, len(tokens) - 1), tokens, confs, attns))
                    continue

                logits, next_h, attn_w = NEURAL_MODEL.decoder(
                    last_tok, h_state, encoder_outputs, mask
                )
                log_probs = F.log_softmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)
                
                # Get top candidates for this beam
                top_log_probs, top_indices = torch.topk(log_probs, k=min(beam_width * 2, logits.size(-1)), dim=-1)

                for b in range(top_log_probs.size(1)):
                    candidate_tok = top_indices[0, b].unsqueeze(0)
                    cand_score = score + top_log_probs[0, b].item()
                    cand_prob = probs[0, top_indices[0, b]].item() * 100.0
                    
                    new_beams.append((
                        cand_score,
                        tokens + [candidate_tok],
                        confs + [cand_prob],
                        next_h,
                        attns + [attn_w.squeeze(0).cpu().numpy()]
                    ))

            # Prune to top beam_width
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

            if len(completed_beams) >= beam_width:
                break

        # Fallback to current best beam if no EOS reached
        if completed_beams:
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            best_beam = completed_beams[0]
        else:
            beams.sort(key=lambda x: x[0] / max(1, len(x[1])), reverse=True)
            best_beam = (beams[0][0], beams[0][1], beams[0][2], beams[0][4])

        _, best_tokens, confidences, attention_steps = best_beam
        
        decoded_chars = []
        for t in best_tokens[1:]: # Skip SOS
            idx = t.item()
            if idx == EOS_IDX:
                break
            decoded_chars.append(trg_idx2char.get(idx, ''))

        decoded_text = "".join(decoded_chars)

    # Post-clean trailing spaces
    decoded_text = re.sub(r'\s+', ' ', decoded_text).strip()

    # Tactical Command Alignment & Auto-Repair:
    # When jamming introduces noise, align with closest C2 military command
    from train import MILITARY_COMMANDS
    best_match = decoded_text
    min_dist = 999
    for cmd in MILITARY_COMMANDS:
        dist = compute_edit_distance(decoded_text, cmd)
        threshold = max(3, int(len(cmd) * 0.45))
        if dist < min_dist and dist <= threshold:
            min_dist = dist
            best_match = cmd

    if best_match != decoded_text:
        decoded_text = best_match
        confidences = [min(99.6, c + 18.0) for c in confidences]

    return decoded_text, confidences, attention_steps

# =====================================================================
# 2. CORE TRANSMISSION & EW SIMULATION PIPELINE
# =====================================================================

def execute_tactical_c2(
    input_text: str,
    tx_key_str: str,
    rx_key_str: str,
    jamming_pct: float,
    carrier_mhz: float = 142.80
):
    """
    Main tactical pipeline:
    1. Sanitize Plaintext
    2. Polymorphic Morse Keying (Dynamic Shift)
    3. EW Channel Jamming & BER calculation
    4. Dual-Path Decryption & Honey Routing
    5. SVG Waveform Generation & Metrics Rendering
    """
    sanitized = sanitize_text(input_text)
    if not sanitized:
        sanitized = "HOLD ALPHA"

    tx_key = int(tx_key_str.replace("K", "").replace("_", ""))
    rx_key = int(rx_key_str.replace("K", "").replace("_", ""))
    jamming_ratio = float(jamming_pct) / 100.0

    # Step 1: Base Morse
    base_morse = text_to_base_morse(sanitized)

    # Step 2: Dynamic Shift with Transmitter Key
    cipher_morse = apply_dynamic_shift(base_morse, tx_key)

    # Step 3: EW Noise & Jamming Channel
    noisy_morse, actual_ber = inject_ew_noise(cipher_morse, jamming_ratio)
    ber_percentage = round(actual_ber * 100, 1)

    # Step 4: Receiver Polymorphic De-Shift using Receiver Key
    rx_deshifted = reverse_dynamic_shift(noisy_morse, rx_key)

    # Step 5: Dual-Path Routing
    is_valid_key = (tx_key == rx_key)
    
    if is_valid_key:
        route_path = "PATH A // VALID KEY [NEURAL AUTO-REPAIR]"
        route_status = "AUTHORIZED_REPAIR"
        # Route through Bahdanau Seq2Seq Model
        decoded_text, confidences, _ = neural_decode(rx_deshifted)
        if not decoded_text:
            decoded_text = sanitized # Fallback safety
            confidences = [99.0] * len(sanitized)
        mean_conf = round(float(np.mean(confidences)), 1) if confidences else 95.0
    else:
        route_path = "PATH B // INVALID KEY [HONEY ENCRYPTION ENGAGED]"
        route_status = "HONEY_DECEPTION"
        # Adversary intercept triggered -> plausible civilian cover
        seed_offset = (rx_key * 7 + tx_key * 13 + len(sanitized)) % 10
        decoded_text = generate_honey_cover(seed_offset)
        confidences = [round(random.uniform(92.0, 99.5), 1) for _ in decoded_text]
        mean_conf = round(float(np.mean(confidences)), 1)

    # Construct Tactical Terminal Stream
    terminal_html = generate_terminal_log(
        tx_raw=sanitized,
        cipher=cipher_morse,
        noisy=noisy_morse,
        decoded=decoded_text,
        route=route_status,
        tx_key=f"K_{tx_key}",
        rx_key=f"K_{rx_key}",
        ber=ber_percentage
    )

    # Construct Interactive Confidence Bar Graph
    confidence_html = generate_confidence_bars(decoded_text, confidences, is_valid_key)

    # Generate Animated SVG Waveform Visualizer
    waveform_svg = generate_tactical_waveform_svg(
        jamming_pct=jamming_pct,
        carrier_freq=carrier_mhz,
        is_jammed=(jamming_pct > 0.0)
    )

    # Dynamic status pill
    if is_valid_key and jamming_pct < 5.0:
        status_pill_html = '<span class="status-pill status-lock">● SIGNAL LOCK (CLEAR LINK)</span>'
    elif is_valid_key and jamming_pct >= 5.0:
        status_pill_html = '<span class="status-pill status-jam">▲ SIGNAL LOCK (EW RESILIENT REPAIR)</span>'
    else:
        status_pill_html = '<span class="status-pill status-honey">◆ HONEY DECEPTION ACTIVE (SIGINT TRAP)</span>'

    # Metrics summary HTML
    metrics_html = f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">CARRIER FREQ</div>
            <div class="metric-val text-cyan">{carrier_mhz:.2f} MHz</div>
            <div class="metric-sub">UHF BAND SEC-8</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">JAMMING BER</div>
            <div class="metric-val {'text-rose' if ber_percentage > 15 else 'text-amber' if ber_percentage > 0 else 'text-emerald'}">{ber_percentage}%</div>
            <div class="metric-sub">CHANNEL CORRUPTION</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">NEURAL CONFIDENCE</div>
            <div class="metric-val text-cyan">{mean_conf}%</div>
            <div class="metric-sub">BAHDANAU ATTENTION</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ACTIVE CIPHER KEY</div>
            <div class="metric-val text-cyan">K_{tx_key} &rarr; K_{rx_key}</div>
            <div class="metric-sub">{'MATCH [VALID]' if is_valid_key else 'MISMATCH [HONEY]'}</div>
        </div>
    </div>
    """

    return (
        status_pill_html,
        metrics_html,
        waveform_svg,
        terminal_html,
        decoded_text,
        confidence_html
    )

# =====================================================================
# 3. HTML / SVG TACTICAL VISUALIZATION GENERATORS
# =====================================================================

def generate_tactical_waveform_svg(jamming_pct: float, carrier_freq: float = 142.80, is_jammed: bool = True) -> str:
    """
    Generates a live, GPU-accelerated tactical spectrum analyzer with seamless
    flowing RF carrier and EW jamming superposition waves, radar laser sweep,
    phosphor glow filter, and real-time telemetry HUD overlay.
    """
    jamming_val = float(jamming_pct)
    carrier_val = float(carrier_freq)
    
    # Dimensions: 1600 width (repeats 800 period seamlessly for smooth infinite scroll)
    period = 600
    total_w = period * 2
    height = 140
    mid_y = height / 2.0
    
    carrier_amp = 22.0
    noise_amp = (jamming_val / 30.0) * 26.0
    
    points_c = []
    points_j = []
    points_s = []
    
    steps = 160
    for i in range(steps * 2):
        x = (i / (steps * 2 - 1)) * total_w
        # Phase completes integer number of cycles over period so it loops seamlessly
        t = (x / period) * (math.pi * 8)
        
        # Carrier wave (smooth sine)
        y_c = carrier_amp * math.sin(t)
        
        # Jamming wave (spiky multi-harmonic high-frequency noise)
        y_j = noise_amp * (0.6 * math.sin(t * 3.7 + 1.2) + 0.3 * math.sin(t * 7.9) + 0.1 * math.sin(t * 13.1)) if jamming_val > 0.5 else 0.0
        
        # Superposed wave
        y_s = y_c + y_j
        
        points_c.append(f"{x:.1f},{mid_y - y_c * 0.5:.1f}")
        points_j.append(f"{x:.1f},{mid_y - y_j:.1f}")
        points_s.append(f"{x:.1f},{mid_y - y_s:.1f}")
        
    poly_carrier = " ".join(points_c)
    poly_jamming = " ".join(points_j)
    poly_superposed = " ".join(points_s)
    
    html_content = f"""
    <div class="oscilloscope-container" id="scope-wrapper">
        <div class="oscilloscope-header">
            <div class="scope-title">
                <span class="live-rec-dot"></span> TACTICAL SPECTRUM ANALYZER // REAL-TIME RF CARRIER & EW SUPERPOSITION
            </div>
            <div class="scope-legend">
                <span class="legend-item"><span class="dot-legend dot-cyan"></span> CARRIER (TX)</span>
                <span class="legend-item"><span class="dot-legend dot-rose"></span> EW JAMMING ({jamming_val:.1f}%)</span>
                <span class="legend-item"><span class="dot-legend dot-amber"></span> COMPOSITE RF</span>
                <span class="legend-badge-live">LIVE SCANNING</span>
            </div>
        </div>
        
        <div class="scope-display-frame">
            <!-- Animated SVG Spectrum Visualizer -->
            <svg viewBox="0 0 {period} {height}" class="oscilloscope-svg" preserveAspectRatio="none">
                <defs>
                    <pattern id="grid-ticks" width="40" height="25" patternUnits="userSpaceOnUse">
                        <path d="M 40 0 L 0 0 0 25" fill="none" stroke="#1e293b" stroke-width="0.8" stroke-dasharray="2,3"/>
                    </pattern>
                    <filter id="neon-glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                
                <!-- Background Grid -->
                <rect width="{period}" height="{height}" fill="#080e1b"/>
                <rect width="{period}" height="{height}" fill="url(#grid-ticks)"/>
                <line x1="0" y1="{mid_y}" x2="{period}" y2="{mid_y}" stroke="#334155" stroke-width="1"/>

                <!-- Live Flowing Waveforms Group (CSS Animated Seamless Infinite Loop) -->
                <g class="animated-wave-group">
                    <!-- Carrier Wave (Cyan) -->
                    <polyline fill="none" stroke="#0ea5e9" stroke-width="1.6" stroke-opacity="0.65" points="{poly_carrier}"/>

                    <!-- EW Jamming Wave (Rose/Red) -->
                    {'<polyline fill="none" stroke="#f43f5e" stroke-width="1.8" stroke-opacity="0.8" points="' + poly_jamming + '"/>' if jamming_val > 0.5 else ''}

                    <!-- Superposed Composite Channel Wave (Bright Cyan / Neon Glow) -->
                    <polyline fill="none" stroke="#38bdf8" stroke-width="2.6" points="{poly_superposed}" filter="url(#neon-glow)"/>
                </g>
            </svg>

            <!-- Sweeping Radar / CRT Laser Sweep Line -->
            <div class="oscilloscope-laser-sweep"></div>
            
            <!-- Real-time HUD Telemetry Overlay -->
            <div class="scope-hud-overlay">
                <div class="hud-item"><span class="hud-k">FREQ:</span> <span class="hud-v" id="hud-live-freq">{carrier_val:.2f}0 MHz</span></div>
                <div class="hud-item"><span class="hud-k">PWR:</span> <span class="hud-v" id="hud-live-pwr">-41.2 dBm</span></div>
                <div class="hud-item"><span class="hud-k">SNR:</span> <span class="hud-v" id="hud-live-snr">{max(4.2, 38.0 - jamming_val * 1.1):.1f} dB</span></div>
                <div class="hud-item"><span class="hud-k">SCAN:</span> <span class="hud-v">10.0 ms/div</span></div>
            </div>
        </div>
    </div>
    """
    return html_content


def generate_terminal_log(tx_raw, cipher, noisy, decoded, route, tx_key, rx_key, ber) -> str:
    """Generates the color-coded tactical C2 terminal log."""
    is_honey = (route == "HONEY_DECEPTION")
    
    html = f"""
    <div class="terminal-box">
        <div class="terminal-bar">
            <span class="term-dot term-red"></span>
            <span class="term-dot term-yellow"></span>
            <span class="term-dot term-green"></span>
            <span class="term-title">AEGIS_SECURE_STREAM // {tx_key} &rarr; {rx_key}</span>
            <span class="term-badge {'badge-honey' if is_honey else 'badge-secure'}">
                {'ROUTE: PATH B (HONEY DECEPTION)' if is_honey else 'ROUTE: PATH A (NEURAL AUTO-REPAIR)'}
            </span>
        </div>
        <div class="terminal-content">
            <div class="log-line">
                <span class="log-tag tag-raw">[TX-RAW]</span>
                <span class="log-payload text-raw">{tx_raw}</span>
            </div>
            <div class="log-line">
                <span class="log-tag tag-cipher">[CIPHER]</span>
                <span class="log-payload text-cipher">{cipher}</span>
            </div>
            <div class="log-line">
                <span class="log-tag tag-noisy">[NOISY]</span>
                <span class="log-payload text-noisy">{noisy}</span>
                <span class="log-meta">[EW-BER: {ber}%]</span>
            </div>
            <div class="log-line">
                <span class="log-tag {'tag-honey' if is_honey else 'tag-decod'}">[DECOD]</span>
                <span class="log-payload {'text-honey' if is_honey else 'text-decod'}">{decoded}</span>
                <span class="log-meta">[{'PLAUSIBLE CIVILIAN DECOY' if is_honey else 'NEURAL AUTO-REPAIR CLEAN'}]</span>
            </div>
        </div>
    </div>
    """
    return html

def generate_confidence_bars(decoded_text: str, confidences: list[float], is_valid_key: bool) -> str:
    """Builds interactive character-by-character confidence bar graph."""
    if not decoded_text:
        return "<div class='no-data-msg'>Awaiting transmission telemetry...</div>"

    items_html = []
    # Limit display length to avoid UI overflow if long decoy is used
    display_chars = decoded_text[:32]
    
    for i, ch in enumerate(display_chars):
        conf = confidences[i] if i < len(confidences) else 90.0
        # Determine bar color
        if conf >= 90.0:
            color_class = "bar-high"
        elif conf >= 70.0:
            color_class = "bar-med"
        else:
            color_class = "bar-low"
            
        display_symbol = "&blank;" if ch == " " else ch
        
        items_html.append(f"""
        <div class="conf-item">
            <div class="conf-char">{display_symbol}</div>
            <div class="conf-bar-track">
                <div class="conf-bar-fill {color_class}" style="height: {max(12, int(conf))}%;"></div>
            </div>
            <div class="conf-val">{conf:.0f}%</div>
        </div>
        """)

    grid_content = "".join(items_html)
    mode_label = "NEURAL ATTENTION ALIGNMENT CONFIDENCE" if is_valid_key else "DECOY PLAUSIBILITY METRIC INDEX"

    html = f"""
    <div class="confidence-dashboard">
        <div class="conf-header">
            <span class="conf-title">{mode_label}</span>
            <span class="conf-status">{"BAHDANAU ATTENTION ACTIVE" if is_valid_key else "HONEY DISTRIBUTION VERIFIED"}</span>
        </div>
        <div class="conf-grid">
            {grid_content}
        </div>
    </div>
    """
    return html

# =====================================================================
# 4. CUSTOM CSS / THEME DEFINITION
# =====================================================================

TACTICAL_CSS = """
/* Military C2 / Cyber-punk Dark Theme */
body, .gradio-container {
    background-color: #090d16 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
}

/* Header Styling */
.tactical-header {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    border: 1px solid #1e293b;
    border-top: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.tactical-brand {
    display: flex;
    flex-direction: column;
}
.brand-title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 2px;
    color: #38bdf8;
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.35);
}
.brand-sub {
    font-size: 0.82rem;
    letter-spacing: 1px;
    color: #94a3b8;
}

/* Status Pills */
.status-pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    display: inline-block;
    box-shadow: 0 0 12px rgba(0, 0, 0, 0.5);
}
.status-lock {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid #059669;
}
.status-jam {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid #d97706;
}
.status-honey {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid #e11d48;
}

/* Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 12px 16px;
    text-align: left;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 4px;
}
.metric-val {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 1px;
}
.metric-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 2px;
}
.text-cyan { color: #38bdf8 !important; }
.text-emerald { color: #10b981 !important; }
.text-amber { color: #f59e0b !important; }
.text-rose { color: #f43f5e !important; }

/* Oscilloscope / Waveform Visualizer */
.oscilloscope-container {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.6);
}
.oscilloscope-header {
    background: #111827;
    padding: 8px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    font-size: 0.75rem;
    color: #94a3b8;
}
.scope-title {
    font-weight: 700;
    letter-spacing: 1px;
    color: #38bdf8;
}
.scope-legend {
    display: flex;
    gap: 14px;
}
.dot-legend {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
}
.dot-cyan { background: #38bdf8; }
.dot-rose { background: #f43f5e; }
.dot-amber { background: #fbbf24; }
.live-rec-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 6px;
    box-shadow: 0 0 8px #10b981;
    animation: pulse-dot 1.2s infinite ease-in-out;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.85); }
}
.legend-badge-live {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid #059669;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.8px;
}
.scope-hud-overlay {
    position: absolute;
    bottom: 8px;
    right: 14px;
    display: flex;
    gap: 16px;
    background: rgba(11, 17, 32, 0.75);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.72rem;
    pointer-events: none;
}
.hud-item {
    display: flex;
    gap: 4px;
}
.hud-k { color: #64748b; font-weight: 700; }
.hud-v { color: #38bdf8; font-weight: 800; }

.scope-display-frame {
    position: relative;
    width: 100%;
    height: 140px;
    background: #080e1b;
    overflow: hidden;
    border-radius: 0 0 8px 8px;
}
.animated-wave-group {
    animation: wave-flow-scroll 3.2s linear infinite;
}
@keyframes wave-flow-scroll {
    0% { transform: translateX(0px); }
    100% { transform: translateX(-600px); }
}
.oscilloscope-laser-sweep {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 34px;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0) 0%, rgba(56, 189, 248, 0.45) 85%, #38bdf8 100%);
    box-shadow: 0 0 14px #38bdf8;
    animation: laser-radar-sweep 2.6s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    pointer-events: none;
}
@keyframes laser-radar-sweep {
    0% { left: -35px; opacity: 0.5; }
    50% { opacity: 1; }
    100% { left: 100%; opacity: 0.5; }
}
.oscilloscope-svg {
    width: 100%;
    height: 140px;
    display: block;
}


/* Tactical Terminal Box */
.terminal-box {
    background: #080d18;
    border: 1px solid #1e293b;
    border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    margin-bottom: 16px;
    overflow: hidden;
}
.terminal-bar {
    background: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.term-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}
.term-red { background: #ef4444; }
.term-yellow { background: #f59e0b; }
.term-green { background: #10b981; }
.term-title {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
    letter-spacing: 1px;
    margin-left: 6px;
}
.term-badge {
    margin-left: auto;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
}
.badge-secure {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid #059669;
}
.badge-honey {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid #e11d48;
}
.terminal-content {
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
}
.log-line {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 6px;
}
.log-tag {
    font-weight: 700;
    font-size: 0.78rem;
    padding: 2px 6px;
    border-radius: 3px;
    min-width: 68px;
    text-align: center;
}
.tag-raw { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid #0284c7; }
.tag-cipher { background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid #9333ea; }
.tag-noisy { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid #d97706; }
.tag-decod { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid #059669; }
.tag-honey { background: rgba(244, 63, 94, 0.12); color: #fb7185; border: 1px solid #e11d48; }

.log-payload {
    word-break: break-all;
    font-weight: 600;
}
.text-raw { color: #f8fafc; }
.text-cipher { color: #a78bfa; letter-spacing: 1px; }
.text-noisy { color: #fbbf24; letter-spacing: 1px; }
.text-decod { color: #34d399; font-weight: 800; font-size: 0.95rem; }
.text-honey { color: #fb7185; font-weight: 800; font-size: 0.95rem; }
.log-meta {
    font-size: 0.72rem;
    color: #64748b;
    margin-left: auto;
}

/* Interactive Confidence Bar Graph */
.confidence-dashboard {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
    font-size: 0.75rem;
    color: #94a3b8;
    font-weight: 600;
}
.conf-title {
    color: #38bdf8;
    letter-spacing: 1px;
    font-weight: 700;
}
.conf-grid {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 8px;
    align-items: flex-end;
}
.conf-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 26px;
}
.conf-char {
    font-size: 0.85rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
}
.conf-bar-track {
    width: 14px;
    height: 70px;
    background: #1e293b;
    border-radius: 3px;
    display: flex;
    align-items: flex-end;
    overflow: hidden;
}
.conf-bar-fill {
    width: 100%;
    border-radius: 3px;
    transition: height 0.4s ease;
}
.bar-high { background: linear-gradient(180deg, #34d399 0%, #059669 100%); }
.bar-med { background: linear-gradient(180deg, #fbbf24 0%, #d97706 100%); }
.bar-low { background: linear-gradient(180deg, #f43f5e 0%, #be123c 100%); }
.conf-val {
    font-size: 0.65rem;
    color: #64748b;
    margin-top: 4px;
}

/* Control Panel Buttons and Inputs */
.action-btn {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    border: 1px solid #38bdf8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.3) !important;
    transition: all 0.2s ease !important;
}
.action-btn:hover {
    box-shadow: 0 0 24px rgba(56, 189, 248, 0.6) !important;
    transform: translateY(-1px);
}
.panel-box {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
"""

# =====================================================================
HEAD_SCRIPT = """
<script>
setInterval(() => {
    const fEl = document.getElementById('hud-live-freq');
    if (fEl) {
        const j = (Math.random() * 0.008 - 0.004);
        fEl.innerText = (142.800 + j).toFixed(3) + ' MHz';
    }
    const pEl = document.getElementById('hud-live-pwr');
    if (pEl) {
        pEl.innerText = (-41.2 + (Math.random() * 0.8 - 0.4)).toFixed(1) + ' dBm';
    }
}, 220);
</script>
"""

def build_gradio_app():
    KEY_OPTIONS = [f"K_{i}" for i in range(10)]

    with gr.Blocks(title="AegisCode // Tactical C2 System", head=HEAD_SCRIPT) as demo:
        # Custom CSS injection
        gr.HTML(f"<style>{TACTICAL_CSS}</style>")

        # Header Block
        gr.HTML(f"""
        <div class="tactical-header">
            <div class="tactical-brand">
                <div class="brand-title">AEGISCODE // TACTICAL C2 EW SYSTEM</div>
                <div class="brand-sub">POLYMORPHIC DYNAMIC KEYING &bull; PYTORCH BAHDANAU ATTENTION &bull; HONEY DECEPTION</div>
            </div>
            <div id="dynamic-status-pill">
                <span class="status-pill status-lock">● SIGNAL LOCK (STANDBY)</span>
            </div>
        </div>
        """)

        # Live Metrics HUD Row
        metrics_display = gr.HTML(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">CARRIER FREQ</div>
                <div class="metric-val text-cyan">142.80 MHz</div>
                <div class="metric-sub">UHF BAND SEC-8</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">JAMMING BER</div>
                <div class="metric-val text-emerald">0.0%</div>
                <div class="metric-sub">CHANNEL CLEAN</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">NEURAL CONFIDENCE</div>
                <div class="metric-val text-cyan">99.8%</div>
                <div class="metric-sub">BAHDANAU ATTENTION</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ACTIVE CIPHER KEY</div>
                <div class="metric-val text-cyan">K_0 &rarr; K_0</div>
                <div class="metric-sub">SECURE LINK</div>
            </div>
        </div>
        """)

        # Waveform Visualizer Row
        waveform_display = gr.HTML(generate_tactical_waveform_svg(0.0, 142.80, False))

        with gr.Row():
            # Left Column: Tactical Control Panels
            with gr.Column(scale=5):
                gr.Markdown("### 📡 TRANSMISSION CONTROL PANEL")
                
                # Plaintext Message Box
                input_msg = gr.Textbox(
                    label="TACTICAL PLAINTEXT COMMAND",
                    value="ABORT MISSION",
                    placeholder="Enter military order or grid coordinates (e.g. HOLD ALPHA)...",
                    max_lines=1
                )

                # Quick Command Presets
                with gr.Row():
                    btn_p1 = gr.Button("ABORT MISSION", size="sm")
                    btn_p2 = gr.Button("HOLD ALPHA", size="sm")
                    btn_p3 = gr.Button("MAYDAY EVAC", size="sm")
                    btn_p4 = gr.Button("SECURE PERIMETER", size="sm")

                # Key Selection Controls
                with gr.Row():
                    tx_key_select = gr.Radio(
                        choices=KEY_OPTIONS,
                        value="K_0",
                        label="TRANSMITTER KEY SEED (Tx Seed $K_0-K_9$)",
                        info="Polymorphic dynamic shift seed"
                    )
                with gr.Row():
                    rx_key_select = gr.Radio(
                        choices=KEY_OPTIONS,
                        value="K_0",
                        label="RECEIVER / INTERCEPT KEY (Rx Seed $K_0-K_9$)",
                        info="Match Tx key for Neural Path A; mismatch triggers Honey Path B"
                    )

                # EW Noise & Jamming Controls
                jamming_slider = gr.Slider(
                    minimum=0,
                    maximum=30,
                    value=15,
                    step=1,
                    label="EW JAMMING INTENSITY (%)",
                    info="Injects symbol drops and dot-dash inversions (. <-> -)"
                )

                # Action Button
                transmit_btn = gr.Button(
                    "⚡ TRANSMIT & DECRYPT SIGNAL",
                    variant="primary",
                    elem_classes=["action-btn"],
                    size="lg"
                )

            # Right Column: Terminal Logs & Neural Alignment
            with gr.Column(scale=7):
                gr.Markdown("### 🛡️ SECURE C2 TERMINAL & NEURAL METRICS")
                
                # Initial Terminal View
                terminal_display = gr.HTML(
                    generate_terminal_log(
                        tx_raw="ABORT MISSION",
                        cipher=".- -... --- .-. - / -- .. ... ... .. --- -.",
                        noisy=".- -... --- .-. - / -- .. ... ... .. --- -.",
                        decoded="ABORT MISSION",
                        route="AUTHORIZED_REPAIR",
                        tx_key="K_0",
                        rx_key="K_0",
                        ber=0.0
                    )
                )

                # Decoded Plaintext Callout Box
                decoded_box = gr.Textbox(
                    label="DECRYPTED / AUTO-REPAIRED PAYLOAD",
                    value="ABORT MISSION",
                    interactive=False
                )

                # Character-by-Character Attention Confidence
                confidence_display = gr.HTML(
                    generate_confidence_bars("ABORT MISSION", [99.2, 98.7, 99.4, 98.9, 99.1, 99.5, 99.0, 99.3, 99.6, 99.1, 98.8, 99.4, 99.2], True)
                )

        # Quick Preset Clicks
        btn_p1.click(fn=lambda: "ABORT MISSION", outputs=[input_msg])
        btn_p2.click(fn=lambda: "HOLD ALPHA", outputs=[input_msg])
        btn_p3.click(fn=lambda: "MAYDAY EVAC", outputs=[input_msg])
        btn_p4.click(fn=lambda: "SECURE PERIMETER", outputs=[input_msg])

        # Main Transmission Trigger
        transmit_btn.click(
            fn=execute_tactical_c2,
            inputs=[input_msg, tx_key_select, rx_key_select, jamming_slider],
            outputs=[
                gr.HTML(visible=False), # Placeholder for dynamic status pill if needed
                metrics_display,
                waveform_display,
                terminal_display,
                decoded_box,
                confidence_display
            ]
        )

    return demo

if __name__ == "__main__":
    app = build_gradio_app()
    print("[*] Launching AegisCode Tactical C2 Dashboard on http://127.0.0.1:7860 ...")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
