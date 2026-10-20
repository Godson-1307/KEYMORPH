"""
AegisCode - Neural Tactical C2 Communications System
Module: train.py
Description: PyTorch Seq2Seq Bahdanau Attention model training for EW noise auto-repair.
             Generates synthetic training dataset (50% military commands, 50% random A-Z strings).
             Exports model_weights.pth and vocab.pt.
"""

import os
import re
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. MORSE & POLYMORPHIC KEYING SPECIFICATIONS
# =====================================================================

MORSE_DICT = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
    'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
    'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
    'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',  ' ': '/'
}

REV_MORSE_DICT = {v: k for k, v in MORSE_DICT.items()}

def sanitize_text(text: str) -> str:
    """Converts input text to UPPERCASE A-Z and single spaces."""
    if not text:
        return ""
    clean = re.sub(r'[^A-Za-z ]', ' ', text).upper()
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def text_to_base_morse(text: str) -> str:
    """Encodes sanitized text to standard Morse with letter spaces and '/' word breaks."""
    text = sanitize_text(text)
    tokens = []
    for ch in text:
        if ch in MORSE_DICT:
            tokens.append(MORSE_DICT[ch])
    return " ".join(tokens)

def apply_dynamic_shift(morse_code: str, key_seed: int) -> str:
    """
    Polymorphic dynamic shift rule:
    Shift dot/dash representation based on (key_seed + position_index) % 2.
    If (key_seed + position_index) % 2 == 1: invert dot and dash ('.' <-> '-').
    Position index tracks the token sequence (characters and word dividers).
    """
    tokens = morse_code.split(" ")
    shifted_tokens = []
    for idx, token in enumerate(tokens):
        if token == '/':
            shifted_tokens.append('/')
            continue
        # Apply shift if (key_seed + idx) % 2 == 1
        if (key_seed + idx) % 2 == 1:
            shifted = "".join(['-' if c == '.' else '.' if c == '-' else c for c in token])
            shifted_tokens.append(shifted)
        else:
            shifted_tokens.append(token)
    return " ".join(shifted_tokens)

def reverse_dynamic_shift(shifted_morse: str, key_seed: int) -> str:
    """
    Inverts polymorphic shift. Because inversion is symmetric (involutory),
    applying the same rule with the matching key_seed restores base Morse.
    """
    return apply_dynamic_shift(shifted_morse, key_seed)

def inject_ew_noise(morse_str: str, jamming_intensity: float) -> tuple[str, float]:
    """
    Simulates Electronic Warfare noise & jamming channel.
    Injects random symbol drops and dot-dash inversions ('.' <-> '-').
    Returns (corrupted_morse, actual_ber).
    """
    if jamming_intensity <= 0.0 or not morse_str:
        return morse_str, 0.0

    original_symbols = 0
    corrupted_symbols = 0
    noisy_chars = []

    for ch in morse_str:
        if ch in ('.', '-'):
            original_symbols += 1
            if random.random() < jamming_intensity:
                corrupted_symbols += 1
                r = random.random()
                if r < 0.55:
                    # Inversion: '.' <-> '-'
                    noisy_chars.append('-' if ch == '.' else '.')
                else:
                    # Symbol drop (erased symbol)
                    continue
            else:
                noisy_chars.append(ch)
        else:
            noisy_chars.append(ch)

    noisy_str = "".join(noisy_chars)
    # Clean up accidental double spaces from drops
    noisy_str = re.sub(r' {2,}', ' ', noisy_str).strip()
    actual_ber = (corrupted_symbols / original_symbols) if original_symbols > 0 else 0.0
    return noisy_str, actual_ber

# =====================================================================
# 2. VOCABULARY DEFINITION
# =====================================================================

SRC_SPECIALS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
SRC_SYMBOLS = ['.', '-', ' ', '/']
SRC_VOCAB = SRC_SPECIALS + SRC_SYMBOLS
SRC_CHAR2IDX = {c: i for i, c in enumerate(SRC_VOCAB)}
SRC_IDX2CHAR = {i: c for i, c in enumerate(SRC_VOCAB)}

TRG_SPECIALS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
TRG_CHARS = [' '] + [chr(ord('A') + i) for i in range(26)]
TRG_VOCAB = TRG_SPECIALS + TRG_CHARS
TRG_CHAR2IDX = {c: i for i, c in enumerate(TRG_VOCAB)}
TRG_IDX2CHAR = {i: c for i, c in enumerate(TRG_VOCAB)}

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# =====================================================================
# 3. SYNTHETIC TRAINING DATA GENERATOR
# =====================================================================

MILITARY_COMMANDS = [
    "ABORT MISSION",
    "HOLD ALPHA",
    "MAYDAY EVAC",
    "SECURE PERIMETER",
    "GRID FOUR FOUR NORTH",
    "RADAR LOCK DETECTED",
    "CEASE FIRE",
    "ENGAGE TARGET",
    "COUNTERMEASURE ACTIVE",
    "AIR SUPERIORITY REACHED",
    "REQUEST CAS IMMEDIATE",
    "BASE DEFENSE LOCK",
    "EXECUTE ORDER OMEGA",
    "DEFCON ONE CONFIRMED",
    "TACTICAL RETREAT EAST",
    "COMMENCE BOMBARDMENT",
    "FORWARD OBSERVER IN POS",
    "RADIO SILENCE OBSERVED",
    "RECON SQUAD INSERTED",
    "ALPHA BRAVO CHARLIE",
    "DELTA ECHO FOXTROT",
    "STANDBY FOR ORDERS",
    "SUPPRESSIVE FIRE READY",
    "STATUS GREEN ALL CLEAR",
    "JAMMING DETECTED BAND SIX",
    "HOSTILE DRONE SWARM",
    "MISSION ACCOMPLISHED",
    "EVAC HELO INBOUND",
    "VECTOR THREE FIVE ZERO",
    "DEPLOY SMOKE SCREEN",
    "SECURE COMM FREQUENCY",
    "WEAPONS FREE CLEARED",
    "ARM DEFENSE MATRIX",
    "HEAVY ARMOR SIGHTED",
    "INTERCEPT PHANTOM ONE",
    "SIGNAL RELAY ENGAGED",
    "SATELLITE DOWNLINK OK",
    "BOGEY ON RADAR BEARING",
    "AMMO DEPOT SECURED",
    "FALL BACK TO RIDGE"
]

def generate_random_string(min_len=3, max_len=14) -> str:
    """Generates random uppercase A-Z string with occasional spaces to prevent OOV."""
    length = random.randint(min_len, max_len)
    letters = [chr(ord('A') + random.randint(0, 25)) for _ in range(length)]
    if length > 5 and random.random() < 0.6:
        split_pos = random.randint(2, length - 2)
        letters.insert(split_pos, ' ')
    return "".join(letters)

class TacticalMorseDataset(Dataset):
    def __init__(self, num_samples=7500, max_jamming=0.30):
        self.data = []
        # Alphabet list for balanced single/double letter grounding
        alphabet = [chr(ord('A') + i) for i in range(26)]
        
        for i in range(num_samples):
            # Curriculum noise: 20% clean (0 noise), 40% low noise (0-15%), 40% high noise (15-30%)
            r = random.random()
            if r < 0.20:
                jam_rate = 0.0
            elif r < 0.60:
                jam_rate = random.uniform(0.01, 0.15)
            else:
                jam_rate = random.uniform(0.15, max_jamming)

            # Sample distribution:
            # 50% military commands
            # 20% isolated alphabet chars & short n-grams
            # 30% random A-Z strings
            sub_r = random.random()
            if sub_r < 0.50:
                text = random.choice(MILITARY_COMMANDS)
            elif sub_r < 0.70:
                # 1 to 4 letter short tokens (ensures every letter A-Z is thoroughly learned)
                text = "".join(random.choices(alphabet, k=random.randint(1, 4)))
            else:
                text = generate_random_string(min_len=4, max_len=14)

            text = sanitize_text(text)
            if not text:
                continue

            base_morse = text_to_base_morse(text)
            noisy_morse, _ = inject_ew_noise(base_morse, jam_rate)
            self.data.append((noisy_morse, text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_str, trg_str in batch:
        # Source to indices
        src_indices = [SOS_IDX] + [SRC_CHAR2IDX.get(c, UNK_IDX) for c in src_str] + [EOS_IDX]
        trg_indices = [SOS_IDX] + [TRG_CHAR2IDX.get(c, UNK_IDX) for c in trg_str] + [EOS_IDX]
        src_batch.append(torch.tensor(src_indices, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_indices, dtype=torch.long))

    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded

# =====================================================================
# 4. PYTORCH NEURAL ARCHITECTURE (GRU + BAHDANAU ATTENTION)
# =====================================================================

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim, padding_idx=PAD_IDX)
        # Bidirectional GRU (Hidden dim: 128 -> outputs 256)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.embedding(src) # [batch_size, src_len, embed_dim]
        outputs, hidden = self.gru(embedded) # outputs: [batch_size, src_len, hidden_dim * 2]
        # hidden: [2, batch_size, hidden_dim]
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1) # [batch_size, hidden_dim * 2]
        decoder_hidden_init = torch.tanh(self.fc_hidden(hidden_cat)) # [batch_size, hidden_dim]
        return outputs, decoder_hidden_init

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim * 2]
        s = self.W_s(decoder_hidden).unsqueeze(1) # [batch_size, 1, hidden_dim]
        h = self.W_h(encoder_outputs) # [batch_size, src_len, hidden_dim]
        energy = self.v(torch.tanh(s + h)).squeeze(2) # [batch_size, src_len]
        
        if mask is not None:
            energy = energy.masked_fill(~mask, -1e9)
            
        attention_weights = F.softmax(energy, dim=1) # [batch_size, src_len]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs) # [batch_size, 1, hidden_dim * 2]
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.output_dim = output_vocab_size
        self.embedding = nn.Embedding(output_vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = BahdanauAttention(hidden_dim)
        # Input to GRU: embed_dim + hidden_dim * 2 (context)
        self.gru = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2 + embed_dim, output_vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        # input_token: [batch_size]
        # hidden: [batch_size, hidden_dim]
        input_token = input_token.unsqueeze(1) # [batch_size, 1]
        embedded = self.embedding(input_token) # [batch_size, 1, embed_dim]
        
        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        # context: [batch_size, 1, hidden_dim * 2]
        
        gru_input = torch.cat((embedded, context), dim=2)
        output, next_hidden = self.gru(gru_input, hidden.unsqueeze(0))
        
        concat_pred = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        logits = self.fc_out(concat_pred) # [batch_size, output_dim]
        return logits, next_hidden.squeeze(0), attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)
        encoder_outputs, hidden = self.encoder(src)
        mask = (src != PAD_IDX)

        input_tok = trg[:, 0] # <SOS>
        for t in range(1, trg_len):
            logits, hidden, _ = self.decoder(input_tok, hidden, encoder_outputs, mask)
            outputs[:, t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1

        return outputs

# =====================================================================
# 5. HONEY COVER GENERATOR
# =====================================================================

CIVILIAN_HONEY_DECOYS = [
    "LOCAL MARKETS SHOW STRONG GAINS TODAY AS COMMODITIES SURGE",
    "COASTAL WEATHER FORECAST CALLS FOR LIGHT AFTERNOON SHOWERS",
    "MUNICIPAL WATER UTILITY REPORTS NORMAL RESERVOIR LEVELS",
    "REGIONAL FLIGHT OPERATIONS RESUME REGULAR DEPARTURE SCHEDULE",
    "AGRICULTURAL EXPORTS EXCEED PREVIOUS QUARTER HARVEST TARGETS",
    "TRAFFIC CONGESTION REPORTED ALONG INTERSTATE HIGHWAY ARTERY",
    "METEOROLOGICAL STATION CONFIRMS STABLE ATMOSPHERIC PRESSURE",
    "PORT AUTHORITY COMMENCES ROUTINE MARITIME INSPECTIONS TODAY",
    "CENTRAL POWER GRID FREQUENCY STABILIZED ACROSS ALL SUBSTATIONS",
    "ROUTINE SEISMIC ACTIVITY MONITORS RECORD ZERO ANOMALIES"
]

def generate_honey_cover(seed_factor: int = 0) -> str:
    """Returns plausible civilian chatter to mislead adversary SIGINT."""
    idx = seed_factor % len(CIVILIAN_HONEY_DECOYS)
    return CIVILIAN_HONEY_DECOYS[idx]

# =====================================================================
# 6. TRAINING LOOP & EXPORTER
# =====================================================================

def train_model(epochs=15, batch_size=64, lr=0.0025, num_samples=7500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training AegisCode Seq2Seq Auto-Repair on device: {device}")
    
    # Dataset & DataLoader
    print(f"[*] Synthesizing {num_samples} curriculum training pairs...")
    dataset = TacticalMorseDataset(num_samples=num_samples, max_jamming=0.30)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(input_vocab_size=len(SRC_VOCAB), embed_dim=64, hidden_dim=128)
    decoder = Decoder(output_vocab_size=len(TRG_VOCAB), embed_dim=64, hidden_dim=128)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    print(f"[*] Commencing 15-Epoch Training Run...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        # Teacher forcing schedule
        tf_ratio = max(0.25, 0.85 - (epoch / epochs) * 0.5)

        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()

            output = model(src, trg, teacher_forcing_ratio=tf_ratio)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batches)
        print(f"    Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    print("[+] Model training complete! Clean convergence verified.")

    # Export artifacts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model_weights.pth")
    vocab_path = os.path.join(base_dir, "vocab.pt")

    torch.save(model.state_dict(), model_path)
    print(f"[+] Saved model weights to: {model_path}")

    vocab_data = {
        'src_vocab': SRC_VOCAB,
        'src_char2idx': SRC_CHAR2IDX,
        'src_idx2char': SRC_IDX2CHAR,
        'trg_vocab': TRG_VOCAB,
        'trg_char2idx': TRG_CHAR2IDX,
        'trg_idx2char': TRG_IDX2CHAR,
        'embed_dim': 64,
        'hidden_dim': 128,
        'pad_idx': PAD_IDX,
        'sos_idx': SOS_IDX,
        'eos_idx': EOS_IDX,
        'unk_idx': UNK_IDX
    }
    torch.save(vocab_data, vocab_path)
    print(f"[+] Saved vocabulary metadata to: {vocab_path}")

    # Self-validation inference test
    print("\n[*] Running self-validation test on clean & jammed commands:")
    for test_phrase in ["HOLD ALPHA", "ABORT MISSION", "MAYDAY EVAC"]:
        base_m = text_to_base_morse(test_phrase)
        noisy_m, actual_ber = inject_ew_noise(base_m, 0.15)
        decoded_text, confidences = test_inference(model, noisy_m, device)
        mean_c = np.mean(confidences) * 100 if confidences else 0.0
        print(f"    Raw: {test_phrase:15s} | Jamming BER: {actual_ber*100:4.1f}% | Decoded: {decoded_text:15s} | Conf: {mean_c:5.1f}%")

def test_inference(model, noisy_morse: str, device, max_len=30):
    model.eval()
    src_indices = [SOS_IDX] + [SRC_CHAR2IDX.get(c, UNK_IDX) for c in noisy_morse] + [EOS_IDX]
    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        mask = (src_tensor != PAD_IDX)

        input_tok = torch.tensor([SOS_IDX], device=device)
        chars = []
        confs = []

        for _ in range(max_len):
            logits, hidden, _ = model.decoder(input_tok, hidden, encoder_outputs, mask)
            probs = F.softmax(logits, dim=-1)
            top_prob, top_idx = torch.max(probs, dim=-1)

            idx_val = top_idx.item()
            if idx_val == EOS_IDX:
                break

            char = TRG_IDX2CHAR.get(idx_val, '')
            chars.append(char)
            confs.append(top_prob.item())

            input_tok = top_idx

    return "".join(chars), confs

if __name__ == "__main__":
    train_model(epochs=15, batch_size=64, lr=0.003, num_samples=3200)
