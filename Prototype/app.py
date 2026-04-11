import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import os
import math
import warnings
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import ASTModel
from scipy.signal import butter, sosfilt
import subprocess

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE     = 16000
DURATION        = 6
N_MELS          = 128
CHECKPOINT_PATH = "best_model.pth"
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTS  = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}

# Permanent WAV conversion settings
SAVE_CONVERTED_WAV = True
CONVERTED_WAV_DIR = "converted_wavs"

# Global bias
LOGIT_BIAS        = -0.10
LOGIT_TEMPERATURE = 1.0

LOSSY_EXTS = {'.mp3', '.aac', '.m4a', '.ogg', '.flac'}

# No format penalties
FORMAT_LOGIT_BIAS = {
    '.mp3': 0.0,
    '.aac': 0.0,
    '.m4a': 0.0,
    '.ogg': 0.0,
    '.flac':-0.20,
    '.wav': -0.12,
}

CLIP_THRESHOLD  = 0.95
CLIP_RATIO_SOFT = 0.01
CLIP_RATIO_HARD = 0.10
CLIP_MIN_WEIGHT = 0.40

PROB_FLOOR   = 0.01
PROB_CEILING = 0.99

# Weights
NEURAL_WEIGHT_LOSSLESS   = 0.75
ACOUSTIC_WEIGHT_LOSSLESS = 0.25
NEURAL_WEIGHT_LOSSY      = 0.92
ACOUSTIC_WEIGHT_LOSSY    = 0.08

# AI thresholds
AI_THRESHOLD_LOSSLESS = 70
AI_THRESHOLD_LOSSY    = 68

# Acoustic features
SECTION_CONSISTENCY_SCALE = 1.2
DYNAMIC_RANGE_SCALE       = 0.15
TONAL_VARIATION_SCALE     = 0.80
HARMONIC_MOVEMENT_SCALE   = 2.0

# Spectral flatness
SF_CENTER     = 0.18
SF_SIGMA_LOW  = 0.25
SF_SIGMA_HIGH = 0.30

# Zero crossing rate
ZCR_CENTER = 0.20
ZCR_SIGMA  = 0.40

USE_WAV_RECONSTRUCTION = False

# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder='.')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
class HybridASTDetector(nn.Module):
    def __init__(self, ast_backbone):
        super().__init__()
        self.ast = ast_backbone
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1,  kernel_size=3, padding=1), nn.BatchNorm2d(1),  nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.residual_downsample = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.MaxPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        residual = x.unsqueeze(1)
        cnn_out  = self.cnn_block1(residual)
        cnn_out  = self.cnn_block2(cnn_out)
        res      = self.residual_downsample(residual)
        if res.shape != cnn_out.shape:
            res = torch.nn.functional.interpolate(
                res, size=cnn_out.shape[2:], mode='bilinear', align_corners=False)
        cnn_out = (cnn_out + res).squeeze(1)
        if cnn_out.shape[-1] != 1024 or cnn_out.shape[-2] != 128:
            cnn_out = torch.nn.functional.interpolate(
                cnn_out.unsqueeze(1), size=(128, 1024),
                mode='bilinear', align_corners=False).squeeze(1)
        ast_out   = self.ast(cnn_out).last_hidden_state
        cls_token = ast_out[:, 0, :]
        mean_pool = ast_out[:, 1:, :].mean(dim=1)
        return self.classifier(torch.cat([cls_token, mean_pool], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY WEIGHTING (Must be defined before use)
# ══════════════════════════════════════════════════════════════════════════════
def clipping_weight(chunk_tensor):
    arr        = chunk_tensor.squeeze().numpy()
    clip_ratio = float(np.mean(np.abs(arr) >= CLIP_THRESHOLD))
    if clip_ratio <= CLIP_RATIO_SOFT:
        return 1.0, clip_ratio
    if clip_ratio >= CLIP_RATIO_HARD:
        return CLIP_MIN_WEIGHT, clip_ratio
    t      = (clip_ratio - CLIP_RATIO_SOFT) / (CLIP_RATIO_HARD - CLIP_RATIO_SOFT)
    weight = 1.0 - t * (1.0 - CLIP_MIN_WEIGHT)
    return weight, clip_ratio


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=160, n_mels=N_MELS
)


def convert_to_16k_wav_ffmpeg(input_path, original_filename=None):
    """Use ffmpeg to convert to 16kHz WAV - more reliable than torchaudio"""
    base_name = os.path.splitext(os.path.basename(original_filename or input_path))[0]
    base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    os.makedirs(CONVERTED_WAV_DIR, exist_ok=True)
    output_path = os.path.join(CONVERTED_WAV_DIR, f"{base_name}_16khz.wav")
    
    # Use ffmpeg for reliable conversion
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ar', '16000',           # sample rate
        '-ac', '1',               # mono
        '-c:a', 'pcm_s16le',      # 16-bit PCM
        '-y',                     # overwrite output
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  [ffmpeg] Converted to: {output_path}")
        
        # Get duration
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                       '-of', 'default=noprint_wrappers=1:nokey=1', output_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration_sec = float(duration_result.stdout.strip()) if duration_result.stdout else 0
        
        return output_path, duration_sec
    except subprocess.CalledProcessError as e:
        print(f"  [ffmpeg] Error: {e.stderr}")
        # Fall back to torchaudio
        return convert_to_16k_wav_torchaudio(input_path, original_filename)


def convert_to_16k_wav_torchaudio(input_path, original_filename=None):
    """Fallback: Convert using torchaudio"""
    try:
        wav, sr = torchaudio.load(input_path)
    except Exception:
        data, sr = sf.read(input_path, always_2d=True)
        wav = torch.from_numpy(data.T).float()

    original_sr = sr
    original_channels = wav.shape[0]

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)
        print(f"  [torchaudio] Resampled from {original_sr}Hz to {SAMPLE_RATE}Hz")

    # Normalize peak
    peak = wav.abs().max().item()
    if peak > 1e-6:
        wav = wav / peak

    base_name = os.path.splitext(os.path.basename(original_filename or input_path))[0]
    base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    
    os.makedirs(CONVERTED_WAV_DIR, exist_ok=True)
    output_path = os.path.join(CONVERTED_WAV_DIR, f"{base_name}_16khz.wav")
    torchaudio.save(output_path, wav, SAMPLE_RATE, encoding='PCM_S', bits_per_sample=16)
    
    duration_sec = wav.shape[1] / SAMPLE_RATE
    print(f"  [torchaudio] Saved to: {output_path}")
    
    return output_path, duration_sec


def convert_to_16k_wav(input_path, ext, save_permanent=True, original_filename=None):
    """Main conversion function - tries ffmpeg first, then torchaudio"""
    return convert_to_16k_wav_ffmpeg(input_path, original_filename)


def mel_from_chunk(chunk, is_lossy=False, ext='.wav'):
    if is_lossy and USE_WAV_RECONSTRUCTION:
        chunk = apply_wav_domain_reconstruction(chunk, ext)

    mel    = mel_transform(chunk)
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, multiplier=10.0, amin=1e-10,
        db_multiplier=0.0, top_db=80.0
    )
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    mel_db = mel_db.squeeze(0).transpose(0, 1)
    T = mel_db.shape[0]
    if T < 1024:
        mel_db = torch.nn.functional.pad(mel_db, (0, 0, 0, 1024 - T))
    elif T > 1024:
        mel_db = mel_db[:1024, :]
    return mel_db.unsqueeze(0)


def apply_wav_domain_reconstruction(chunk_tensor, ext):
    if not USE_WAV_RECONSTRUCTION:
        return chunk_tensor

    arr = chunk_tensor.squeeze().numpy().astype(np.float64)
    sr  = SAMPLE_RATE

    if ext in ('.mp3', '.aac', '.m4a'):
        block = 576
        fade  = 32
        for b in range(block, len(arr) - block, block):
            if b + fade >= len(arr):
                break
            fade_out = np.linspace(1.0, 0.0, fade)
            fade_in  = np.linspace(0.0, 1.0, fade)
            arr[b:b+fade] = arr[b:b+fade] * 0.7 + (
                arr[b-fade:b] * fade_out + arr[b:b+fade] * fade_in) * 0.3

    nyq    = sr / 2.0
    hp_wn  = np.clip(40.0 / nyq, 1e-6, 0.9999)
    sos_hp = butter(4, hp_wn, btype='high', output='sos')
    arr    = sosfilt(sos_hp, arr)

    if ext in ('.mp3', '.aac', '.m4a'):
        hs_wn   = np.clip(6000.0 / nyq, 1e-6, 0.9999)
        sos_hs  = butter(2, hs_wn, btype='high', output='sos')
        hf_part = sosfilt(sos_hs, arr)
        arr     = arr + 0.05 * hf_part

    frame_len = int(sr * 0.020)
    n_frames  = len(arr) // frame_len
    rms_frames = np.array([
        np.sqrt(np.mean(arr[i*frame_len:(i+1)*frame_len]**2) + 1e-12)
        for i in range(n_frames)
    ])
    for i in range(1, n_frames - 1):
        if rms_frames[i+1] > rms_frames[i] * 5.0 and rms_frames[i] < rms_frames[i-1] * 0.5:
            start = i * frame_len
            end   = min((i+1) * frame_len, len(arr))
            arr[start:end] *= 0.6

    rms_target = 10 ** (-16.0 / 20.0)
    rms_actual = np.sqrt(np.mean(arr**2) + 1e-12)
    if rms_actual > 1e-6:
        gain = rms_target / rms_actual
        gain = np.clip(gain, 0.1, 10.0)
        arr  = arr * gain

    arr = np.clip(arr, -1.0, 1.0)
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def prepare_full_audio(waveform):
    target_len = SAMPLE_RATE * DURATION
    total      = waveform.shape[1]
    total_secs = total / SAMPLE_RATE

    if total < target_len:
        chunk = torch.nn.functional.pad(waveform, (0, target_len - total))
    else:
        chunk = waveform[:, :target_len]

    return chunk, total_secs


# ══════════════════════════════════════════════════════════════════════════════
#  ACOUSTIC FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def compute_spectral_flatness(chunk_np):
    spec  = np.abs(np.fft.rfft(chunk_np, n=1024)) ** 2 + 1e-10
    geo   = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    return float(geo / arith)

def compute_dynamic_range(chunk_np, frame_len=1600):
    frames = [chunk_np[i:i+frame_len] for i in range(0, len(chunk_np)-frame_len, frame_len)]
    if not frames:
        return 0.0
    rms_vals = [np.sqrt(np.mean(f**2) + 1e-9) for f in frames]
    return float(np.std(rms_vals))

def compute_zero_crossing_rate(chunk_np):
    crossings = np.sum(np.abs(np.diff(np.sign(chunk_np)))) / 2
    return float(crossings / len(chunk_np))

def compute_spectral_centroid(chunk_np):
    mag   = np.abs(np.fft.rfft(chunk_np, n=1024))
    freqs = np.fft.rfftfreq(1024, d=1/SAMPLE_RATE)
    denom = np.sum(mag) + 1e-10
    return float(np.sum(freqs * mag) / denom / (SAMPLE_RATE / 2))

def compute_spectral_rolloff(chunk_np, roll_percent=0.85):
    mag    = np.abs(np.fft.rfft(chunk_np, n=1024)) ** 2
    freqs  = np.fft.rfftfreq(1024, d=1/SAMPLE_RATE)
    cumsum = np.cumsum(mag)
    idx    = min(np.searchsorted(cumsum, roll_percent * cumsum[-1]), len(freqs)-1)
    return float(freqs[idx] / (SAMPLE_RATE / 2))

def compute_temporal_flux(chunk_np, frame_len=1600):
    frames = [chunk_np[i:i+frame_len] for i in range(0, len(chunk_np)-frame_len, frame_len)]
    if len(frames) < 2:
        return 0.0
    specs = [np.abs(np.fft.rfft(f, n=min(1024, frame_len))) for f in frames]
    return float(np.mean([np.sum((specs[i+1]-specs[i])**2) for i in range(len(specs)-1)]))

def compute_chunk_features(chunk_tensor):
    arr = chunk_tensor.squeeze().numpy()
    rms   = float(np.sqrt(np.mean(arr ** 2)) + 1e-9)
    n_rms = np.clip(arr / rms * 0.1, -1.0, 1.0)
    n_amp = np.clip(arr, -1.0, 1.0)
    return {
        'spectral_flatness':  compute_spectral_flatness(n_rms),
        'dynamic_range':      compute_dynamic_range(n_amp),
        'zero_crossing_rate': compute_zero_crossing_rate(n_rms),
        'spectral_centroid':  compute_spectral_centroid(n_rms),
        'spectral_rolloff':   compute_spectral_rolloff(n_rms),
        'temporal_flux':      compute_temporal_flux(n_amp),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FIXED SCORING HELPERS - More realistic scoring
# ══════════════════════════════════════════════════════════════════════════════

def exp_decay_score(val, scale, min_val=0, max_val=100):
    """Exponential decay score - lower values = higher AI score"""
    capped_val = min(val, 0.5)
    raw_score = 100 * math.exp(-capped_val / scale)
    return max(min_val, min(max_val, round(raw_score)))

def asymmetric_gaussian_score(val, center, sigma_low, sigma_high):
    """Gaussian score - values far from center = higher AI score"""
    sigma = sigma_low if val < center else sigma_high
    z = (val - center) / sigma
    z = max(-3, min(3, z))
    raw_score = 100 * (1 - math.exp(-0.5 * z * z))
    return max(5, min(90, round(raw_score)))

def compute_spectral_flatness_score(sf_mean):
    """Special handling for spectral flatness - very flat or very tonal shouldn't auto-mean AI"""
    if 0.1 <= sf_mean <= 0.4:
        return max(5, min(40, round(100 * abs(sf_mean - 0.25) / 0.3)))
    elif sf_mean < 0.05:
        return 60
    elif sf_mean > 0.6:
        return 55
    else:
        return asymmetric_gaussian_score(sf_mean, SF_CENTER, SF_SIGMA_LOW, SF_SIGMA_HIGH)

def compute_dynamic_range_score(dr_mean):
    """Dynamic range scoring - very low dynamic range is common in mastered music"""
    if dr_mean < 0.02:
        return 40
    elif dr_mean < 0.05:
        return 35
    elif dr_mean < 0.10:
        return 30
    elif dr_mean < 0.20:
        return 20
    else:
        return 15

def compute_harmonic_movement_score(sr_std):
    """Harmonic movement - static highs are common in mastered tracks"""
    if sr_std < 0.01:
        return 45
    elif sr_std < 0.03:
        return 40
    elif sr_std < 0.08:
        return 30
    else:
        return 20

def compute_tonal_variation_score(sc_std):
    """Tonal variation - static brightness is common in electronic music"""
    if sc_std < 0.05:
        return 45
    elif sc_std < 0.12:
        return 35
    elif sc_std < 0.25:
        return 25
    else:
        return 15

def acoustic_composite_score(feat):
    """Fixed acoustic composite score - more realistic weighting"""
    
    sf_score = compute_spectral_flatness_score(feat['spectral_flatness'])
    dr_score = compute_dynamic_range_score(feat['dynamic_range'])
    sc_score = compute_tonal_variation_score(feat['spectral_centroid'])
    sr_score = compute_harmonic_movement_score(feat['spectral_rolloff'])
    
    tf_val = feat['temporal_flux']
    if tf_val < 0.5:
        tf_score = 50
    elif tf_val < 1.0:
        tf_score = 35
    elif tf_val < 2.0:
        tf_score = 25
    else:
        tf_score = 15
    
    zcr_val = feat['zero_crossing_rate']
    if zcr_val < 0.05 or zcr_val > 0.45:
        zcr_score = 50
    elif zcr_val < 0.10 or zcr_val > 0.35:
        zcr_score = 40
    else:
        zcr_score = 25
    
    composite = (
        0.20 * sf_score +
        0.20 * dr_score +
        0.20 * sc_score +
        0.20 * sr_score +
        0.10 * tf_score +
        0.10 * zcr_score
    )
    
    return composite


def compression_penalty(feat, is_lossy):
    if not is_lossy:
        return 1.0
    sf_penalty = max(0.0, min(0.03, (feat['spectral_flatness'] - 0.25) * 0.5))
    dr_penalty = max(0.0, min(0.03, 0.12 - feat['dynamic_range'] * 5))
    return 1.0 - (sf_penalty + dr_penalty) * 0.3


# ══════════════════════════════════════════════════════════════════════════════
#  XAI SCORING WITH FIXED EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════

def score_features_for_xai(feat, neural_prob, eff_neural_w, eff_acoustic_w, final_ai_probability=None, is_lossy=False):
    def clip(v, lo=0, hi=100): return max(lo, min(hi, round(v)))

    feats = []
    neural_score = clip(neural_prob * 100)

    is_final_human = final_ai_probability is not None and final_ai_probability < (68 if is_lossy else 60)
    is_mastered = feat['dynamic_range'] < 0.08
    static_harmonics = feat['spectral_rolloff'] < 0.02

    show_override = is_final_human and neural_score < 45 and (static_harmonics or is_mastered)

    if neural_score >= 70:
        neural_why = f'The neural model detected synthesis artifacts or processing patterns consistent with AI generation ({neural_score}%).'
    elif neural_score >= 45:
        neural_why = f'The neural model shows mixed signals ({neural_score}%). Some characteristics lean AI while others appear natural.'
    else:
        if show_override:
            neural_why = f'The neural model detected human-like spectrogram patterns ({neural_score}%). Despite static acoustic features (common in professionally mastered tracks), the neural model correctly identifies natural vocal/instrumental characteristics.'
        else:
            neural_why = f'The neural model detected human-like spectrogram patterns ({neural_score}%). Raw characteristics are consistent with natural audio.'

    feats.append({
        'id': 'neural_score', 'label': 'Neural Spectrogram Score',
        'score': neural_score, 'value': f"{neural_prob:.3f}",
        'weight': eff_neural_w,
        'weight_label': f'{round(eff_neural_w * 100)}% of final score',
        'is_primary': True,
        'what': 'AI probability from the deep neural model on full audio.',
        'why': neural_why,
        'direction': 'higher = more AI-like patterns in the raw spectrogram',
        'overridden_by_neural': False,
    })

    acoustic_feature_weights = {
        'spectral_flatness':   0.20,
        'dynamic_range':       0.20,
        'tonal_variation':     0.20,
        'harmonic_movement':   0.20,
        'section_consistency': 0.10,
        'transient_character': 0.10,
    }

    # Spectral Flatness
    sf_mean = feat['spectral_flatness']
    sf_score = compute_spectral_flatness_score(sf_mean)
    
    if sf_mean < 0.05:
        sf_why = f'Very tonal spectrum detected (flatness={sf_mean:.3f}). This can indicate synthesized sound, but is also common in electronic music and simple instrumentation.'
    elif sf_mean > 0.6:
        sf_why = f'Noise-like spectrum detected (flatness={sf_mean:.3f}). This is common in heavily compressed/mastered tracks — not necessarily AI.'
    elif 0.1 <= sf_mean <= 0.4:
        sf_why = f'Spectral flatness is in the natural human range ({sf_mean:.3f}) — consistent with real instruments and vocals.'
    else:
        sf_why = f'Spectral flatness ({sf_mean:.3f}) is somewhat outside typical range, but not strongly indicative of AI generation.'

    feats.append({
        'id': 'spectral_flatness', 'label': 'Spectral Flatness',
        'score': sf_score, 'value': f"{sf_mean:.3f}",
        'weight': eff_acoustic_w * acoustic_feature_weights['spectral_flatness'],
        'weight_label': f'{round(eff_acoustic_w * acoustic_feature_weights["spectral_flatness"] * 100)}% of final score',
        'is_primary': False,
        'what': 'How "noisy" vs "tonal" the frequency content is.',
        'why': sf_why,
        'direction': 'extreme values (too flat OR too tonal) may indicate synthesis',
        'overridden_by_neural': is_final_human and sf_score > 50 and neural_score < 45,
    })

    # Dynamic Range
    dr_mean = feat['dynamic_range']
    dr_score = compute_dynamic_range_score(dr_mean)
    
    if dr_mean < 0.03:
        dr_why = f'Very steady loudness detected ({dr_mean:.4f}). This is extremely common in modern mastered music — NOT a reliable AI indicator on its own.'
    elif dr_mean < 0.08:
        dr_why = f'Limited dynamic range ({dr_mean:.4f}). Typical of produced and mastered music — not strongly indicative of AI.'
    else:
        dr_why = f'Good loudness variation ({dr_mean:.4f}) — consistent with natural performances and dynamic recordings.'
    
    feats.append({
        'id': 'dynamic_range', 'label': 'Dynamic Range',
        'score': dr_score, 'value': f"{dr_mean:.4f}",
        'weight': eff_acoustic_w * acoustic_feature_weights['dynamic_range'],
        'weight_label': f'{round(eff_acoustic_w * acoustic_feature_weights["dynamic_range"] * 100)}% of final score',
        'is_primary': False,
        'what': 'How much the loudness level varies within the track.',
        'why': dr_why,
        'direction': 'lower variation = potentially more processed (not necessarily AI)',
        'overridden_by_neural': False,
    })

    # Tonal Variation
    sc_std = feat['spectral_centroid']
    sc_score = compute_tonal_variation_score(sc_std)
    
    if sc_std < 0.05:
        sc_why = f'The tonal balance is very static ({sc_std:.4f}). Common in electronic music and loop-based production — not a strong AI indicator.'
    elif sc_std < 0.12:
        sc_why = f'The tonal balance is somewhat static ({sc_std:.4f}). Typical of produced music with consistent instrumentation.'
    else:
        sc_why = f'Natural tonal variation detected ({sc_std:.4f}) — the sound brightens and darkens naturally.'
    
    feats.append({
        'id': 'tonal_variation', 'label': 'Tonal Variation',
        'score': sc_score, 'value': f"{sc_std:.4f}",
        'weight': eff_acoustic_w * acoustic_feature_weights['tonal_variation'],
        'weight_label': f'{round(eff_acoustic_w * acoustic_feature_weights["tonal_variation"] * 100)}% of final score',
        'is_primary': False,
        'what': 'How much the brightness of the sound changes across the track.',
        'why': sc_why,
        'direction': 'less brightness change = more static production',
        'overridden_by_neural': False,
    })

    # Harmonic Movement
    sr_std = feat['spectral_rolloff']
    sr_score = compute_harmonic_movement_score(sr_std)

    if sr_std < 0.01:
        sr_why = f'The high-frequency content is very static ({sr_std:.4f}). This is extremely common in professionally mastered recordings and electronic music — NOT a reliable AI indicator.'
    elif sr_std < 0.03:
        sr_why = f'The high-frequency content is somewhat static ({sr_std:.4f}). Typical of modern production with heavy limiting and compression.'
    else:
        sr_why = f'Natural harmonic movement detected ({sr_std:.4f}) — consistent with live performance and acoustic instruments.'

    feats.append({
        'id': 'harmonic_movement', 'label': 'Harmonic Movement',
        'score': sr_score, 'value': f"{sr_std:.4f}",
        'weight': eff_acoustic_w * acoustic_feature_weights['harmonic_movement'],
        'weight_label': f'{round(eff_acoustic_w * acoustic_feature_weights["harmonic_movement"] * 100)}% of final score',
        'is_primary': False,
        'what': 'How much the high-frequency energy shifts across the track.',
        'why': sr_why,
        'direction': 'less high-frequency movement = more heavily mastered',
        'overridden_by_neural': is_final_human and sr_score > 50 and neural_score < 45,
    })

    # Section Consistency
    tf_cv = feat['temporal_flux']
    if tf_cv < 0.5:
        tf_score = 50
        tf_why = 'The audio is very uniform section-to-section. AI-generated music can have similar-sounding blocks, but so can highly produced electronic music.'
    elif tf_cv < 1.0:
        tf_score = 35
        tf_why = 'Moderate energy variation throughout the track. This is typical of produced music.'
    else:
        tf_score = 20
        tf_why = 'Natural energy variation throughout the track — consistent with human performances.'
    
    feats.append({
        'id': 'section_consistency', 'label': 'Section Consistency',
        'score': tf_score, 'value': f"{tf_cv:.3f}",
        'weight': eff_acoustic_w * acoustic_feature_weights['section_consistency'],
        'weight_label': f'{round(eff_acoustic_w * acoustic_feature_weights["section_consistency"] * 100)}% of final score',
        'is_primary': False,
        'what': 'How consistent the energy is throughout the track.',
        'why': tf_why,
        'direction': 'less relative change = more uniform production',
        'overridden_by_neural': False,
    })

    # Transient Character
    zcr_mean = feat['zero_crossing_rate']
    if zcr_mean < 0.05 or zcr_mean > 0.45:
        zcr_score = 50
        zcr_why = f'Unusual transient pattern detected ({zcr_mean:.3f}). This can occur with heavy processing, synthesis, or extreme compression.'
    elif zcr_mean < 0.10 or zcr_mean > 0.35:
        zcr_score = 40
        zcr_why = f'Somewhat unusual transient character ({zcr_mean:.3f}). May indicate processing or synthesis.'
    else:
        zcr_score = 25
        zcr_why = f'Natural transient character ({zcr_mean:.3f}) — typical of real instruments and voice recordings.'
    
    feats.append({
        'id': 'transient_character', 'label': 'Transient Character',
        'score': zcr_score, 'value': f"{zcr_mean:.3f}",
        'weight': eff_acoustic_w * acoustic_feature_weights['transient_character'],
        'weight_label': f'{round(eff_acoustic_w * acoustic_feature_weights["transient_character"] * 100)}% of final score',
        'is_primary': False,
        'what': 'How sharp or smooth the audio signal transitions are.',
        'why': zcr_why,
        'direction': 'extreme values may indicate synthesis or heavy processing',
        'overridden_by_neural': False,
    })

    primary = [f for f in feats if f['is_primary']]
    secondary = sorted([f for f in feats if not f['is_primary']], key=lambda x: x['score'], reverse=True)
    return primary + secondary


# ══════════════════════════════════════════════════════════════════════════════
#  LABEL + CONFIDENCE
# ══════════════════════════════════════════════════════════════════════════════
def derive_verdict(ai_probability, is_lossy=False):
    threshold = AI_THRESHOLD_LOSSY if is_lossy else AI_THRESHOLD_LOSSLESS

    if ai_probability >= threshold:
        confidence = 'High confidence' if ai_probability >= 85 else 'Moderate confidence'
        explanation = f'Strong AI-like patterns detected ({ai_probability}%).' if ai_probability >= 85 else f'The neural model and acoustic features lean toward AI ({ai_probability}%), but some natural characteristics also present.'
        return ('Likely AI-generated', confidence, explanation)
    else:
        confidence = 'High confidence' if ai_probability <= 30 else 'Moderate confidence'
        explanation = f'Strong human-like patterns detected ({100 - ai_probability}%).' if ai_probability <= 30 else f'The neural model and acoustic features lean toward human ({100 - ai_probability}%), but some AI-like characteristics also present.'
        return ('Likely Human', confidence, explanation)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"[*] Loading model on {device}...")
ast_backbone = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model        = HybridASTDetector(ast_backbone).to(device)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

val_acc = ckpt.get('val_acc', 0)
epoch   = ckpt.get('epoch', 0) + 1
print(f"[✓] Loaded  epoch={epoch}  val_acc={val_acc:.2%}  device={device}")
print(f"[*] Neural weights: lossless={NEURAL_WEIGHT_LOSSLESS:.0%}  lossy={NEURAL_WEIGHT_LOSSY:.0%}")
print(f"[*] Acoustic weights: lossless={ACOUSTIC_WEIGHT_LOSSLESS:.0%}  lossy={ACOUSTIC_WEIGHT_LOSSY:.0%}")
print(f"[*] AI thresholds: lossless={AI_THRESHOLD_LOSSLESS}%  lossy={AI_THRESHOLD_LOSSY}%")
print(f"[*] PERMANENT WAV CONVERSION: {'ENABLED' if SAVE_CONVERTED_WAV else 'DISABLED'}")
print(f"[*] Converted WAV directory: {CONVERTED_WAV_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/converted_wavs/<path:filename>')
def serve_converted_wav(filename):
    """Serve converted WAV files for playback/download"""
    return send_from_directory(CONVERTED_WAV_DIR, filename)


@app.route('/debug/last_analysis', methods=['GET'])
def debug_last_analysis():
    """Debug endpoint to check what file was last analyzed"""
    import glob
    wav_files = glob.glob(os.path.join(CONVERTED_WAV_DIR, "*.wav"))
    return jsonify({
        'converted_wav_dir': CONVERTED_WAV_DIR,
        'wav_files_found': len(wav_files),
        'files': [os.path.basename(f) for f in wav_files],
        'message': 'These are the actual WAV files being analyzed'
    })


@app.route('/classify', methods=['POST'])
def classify():
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files uploaded.'}), 400

    results = []

    for f in files:
        filename = f.filename
        original_ext = os.path.splitext(filename)[1].lower()

        if original_ext not in SUPPORTED_EXTS:
            results.append({'filename': filename, 'error': f'Unsupported format: {original_ext}'})
            continue

        is_lossy_original = original_ext in LOSSY_EXTS

        with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        wav_path = None

        try:
            print(f"\n{'='*60}")
            print(f"[→] INPUT: {filename} (original format: {original_ext})")
            print(f"{'='*60}")

            # STEP 1: Convert original to 16kHz WAV using ffmpeg
            wav_path, total_secs = convert_to_16k_wav(
                tmp_path, original_ext, 
                save_permanent=True, 
                original_filename=filename
            )
            
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"WAV file was not created: {wav_path}")
            
            print(f"\n  ✅ [VERIFY] WAV file exists at: {wav_path}")
            print(f"  ✅ [VERIFY] File size: {os.path.getsize(wav_path)} bytes")

            # STEP 2: RELOAD using soundfile
            print(f"\n  [RELOAD] Loading converted WAV from disk using soundfile...")
            data, reloaded_sr = sf.read(wav_path, always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            
            if reloaded_sr != SAMPLE_RATE:
                print(f"  ⚠️ WARNING: Sample rate is {reloaded_sr}Hz, expected {SAMPLE_RATE}Hz")
                resampler = torchaudio.transforms.Resample(reloaded_sr, SAMPLE_RATE)
                waveform = resampler(waveform)
                reloaded_sr = SAMPLE_RATE
            
            print(f"  ✅ [RELOADED] Successfully loaded: {os.path.basename(wav_path)}")
            print(f"  ✅ [VERIFIED] Sample rate: {reloaded_sr}Hz, Channels: {waveform.shape[0]}, Samples: {waveform.shape[1]}")
            print(f"  ✅ [CONFIRMED] Analyzing clean PCM WAV from disk, NOT original {original_ext.upper()}")
            
            # STEP 3: Analyze the reloaded WAV file
            analysis_ext = '.wav'
            is_lossy_analysis = False
            format_bias = FORMAT_LOGIT_BIAS.get('.wav', 0.0)
            total_bias = LOGIT_BIAS + format_bias
            eff_neural_w = NEURAL_WEIGHT_LOSSLESS
            eff_acoustic_w = ACOUSTIC_WEIGHT_LOSSLESS
            
            print(f"\n  [analysis] Analyzing WAV file (lossless mode)")
            print(f"  [analysis] Bias: {total_bias:+.2f}, Neural weight: {eff_neural_w:.0%}, Acoustic weight: {eff_acoustic_w:.0%}")

            chunk, _ = prepare_full_audio(waveform)

            if chunk.shape[1] < 16000:
                raise ValueError("Audio too short (< 1 second).")

            with torch.no_grad():
                mel = mel_from_chunk(chunk, is_lossy=is_lossy_analysis, ext=analysis_ext).to(device)
                raw_logit = model(mel).squeeze().item()
                clip_weight, clip_ratio = clipping_weight(chunk)
                chunk_feat = compute_chunk_features(chunk)

            cal_logit = (raw_logit + total_bias) / LOGIT_TEMPERATURE
            cal_prob = torch.sigmoid(torch.tensor(cal_logit)).item()
            neural_prob = cal_prob

            acoustic_score = acoustic_composite_score(chunk_feat)
            acoustic_prob = acoustic_score / 100.0

            if clip_ratio > 0.01:
                clip_damping = 1.0 - (min(clip_ratio, 0.10) * 0.5)
                neural_prob = neural_prob * clip_damping
                print(f"  clip_damping={clip_damping:.3f} (clip_ratio={clip_ratio:.4f})")

            base_prob = eff_neural_w * neural_prob + eff_acoustic_w * acoustic_prob
            comp_pen = compression_penalty(chunk_feat, is_lossy_analysis)
            base_prob = base_prob * comp_pen
            print(f"  compression_penalty={comp_pen:.3f}")

            ensemble_prob = max(PROB_FLOOR, min(PROB_CEILING, base_prob))
            ai_probability = round(ensemble_prob * 100, 1)

            verdict, confidence, verdict_explanation = derive_verdict(ai_probability, is_lossy_analysis)

            xai_feats = score_features_for_xai(
                chunk_feat, neural_prob,
                eff_neural_w, eff_acoustic_w,
                final_ai_probability=ai_probability,
                is_lossy=is_lossy_analysis
            )

            display_filename = os.path.basename(wav_path)
            
            print(f"\n{'='*60}")
            print(f"✓ ANALYZED: {display_filename} (converted from {original_ext})")
            print(f"✓ AI Probability: {ai_probability}%")
            print(f"✓ Verdict: {verdict}")
            print(f"{'='*60}\n")

            results.append({
                'original_filename': filename,
                'display_filename': display_filename,
                'analyzed_file': display_filename,
                'original_format': original_ext.upper().replace('.', ''),
                'analyzed_format': 'WAV (16kHz)',
                'conversion_note': f'Converted from {original_ext.upper()} to 16kHz WAV · analyzed as clean PCM',
                'converted_wav_path': f'/converted_wavs/{display_filename}' if SAVE_CONVERTED_WAV else None,
                'ai_probability': ai_probability,
                'verdict': verdict,
                'confidence': confidence,
                'verdict_explanation': verdict_explanation,
                'duration_sec': round(total_secs, 2),
                'xai': xai_feats,
                'score_breakdown': {
                    'neural_contribution': round(eff_neural_w * neural_prob * 100, 1),
                    'acoustic_contribution': round(eff_acoustic_w * acoustic_prob * 100, 1),
                    'neural_weight_pct': round(eff_neural_w * 100),
                    'acoustic_weight_pct': round(eff_acoustic_w * 100),
                },
                'diagnostics': {
                    'analysis_source': wav_path,
                    'converted_from': original_ext,
                    'sample_rate': reloaded_sr,
                    'neural_prob': round(neural_prob, 4),
                    'acoustic_prob': round(acoustic_prob, 4),
                    'base_prob': round(base_prob, 4),
                    'final_prob': ensemble_prob,
                    'compression_penalty': round(comp_pen, 4),
                    'clip_ratio': round(clip_ratio, 4),
                }
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'filename': filename, 'error': str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return jsonify({'results': results})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("[*] Starting SoundScan at http://localhost:5000")
    print("[*] ✅ VERSION: FIXED XAI SCORING + CLIPPING WEIGHT FIX")
    print(f"[*] Converted WAV files will be saved to: {CONVERTED_WAV_DIR}/")
    print("[*] All audio is converted to 16kHz PCM WAV using ffmpeg")
    print("[*] ✅ Acoustic features now have realistic scoring (15-60% range)")
    print("[*] ✅ Static features correctly attributed to mastering, not AI")
    print("[*] 🔍 Debug endpoint: http://localhost:5000/debug/last_analysis")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)