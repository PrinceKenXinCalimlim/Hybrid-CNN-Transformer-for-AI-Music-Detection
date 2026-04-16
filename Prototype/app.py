import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import os
import math
import warnings
import tempfile
import numpy as np
import json
import uuid
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import ASTModel, pipeline
import subprocess

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000
HOP_LENGTH      = 160
FRAMES_NEEDED   = 1024
SAMPLES_NEEDED  = FRAMES_NEEDED * HOP_LENGTH
DURATION_SEC    = SAMPLES_NEEDED / SAMPLE_RATE

CHECKPOINT_PATH = "best_model.pth"
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTS  = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
SAVE_CONVERTED_WAV = True
CONVERTED_WAV_DIR  = "converted_wavs"
REPORTS_FILE       = "misclassification_reports.jsonl"

LOGIT_TEMPERATURE = 1.0
CLIP_THRESHOLD    = 0.95
CLIP_RATIO_SOFT   = 0.01
CLIP_RATIO_HARD   = 0.10
CLIP_MIN_WEIGHT   = 0.40
PROB_FLOOR        = 3.0
PROB_CEILING      = 97.0

# CRITICAL FIX: Restored neural network weight to be dominant
NEURAL_WEIGHT = 0.85      # Neural network is more reliable
ACOUSTIC_WEIGHT = 0.15    # Acoustic features are supportive only

AI_THRESHOLD = 60
HUMAN_THRESHOLD = 50

# XAI deep config
XAI_IG_STEPS       = 50
XAI_LIME_SEGS_T    = 8
XAI_LIME_SEGS_F    = 4
XAI_LIME_SAMPLES   = 64
XAI_LIME_ALPHA     = 1e-3

# Codec compression penalty config (reduced impact)
CODEC_HF_RATIO_THRESHOLD    = 0.02
CODEC_NEURAL_PENALTY        = 0.05
CODEC_SCORE_DISCOUNT        = 0.95

GENRE_MODEL_PRIMARY   = "dima806/music_genres_classification"
GENRE_MODEL_SECONDARY = "mtg-upf/discogs-maest-10s-pw-129e"
GENRE_SAMPLE_RATE     = 16000
GENRE_WIN_PRIMARY     = 10
GENRE_WIN_SECONDARY   = 10
GENRE_N_WINDOWS       = 3
GENRE_TOP_SUBGENRES   = 3

print(f"[*] Configuration:")
print(f"    Sample rate: {SAMPLE_RATE} Hz")
print(f"    Hop length: {HOP_LENGTH} samples")
print(f"    Frames needed: {FRAMES_NEEDED}")
print(f"    Samples needed: {SAMPLES_NEEDED}")
print(f"    Duration: {DURATION_SEC} seconds")
print(f"[*] Weights: Neural={NEURAL_WEIGHT} Acoustic={ACOUSTIC_WEIGHT}")
print(f"[*] Thresholds: AI>{AI_THRESHOLD}% Human<{HUMAN_THRESHOLD}%")
print(f"[*] Prob range: {PROB_FLOOR}%-{PROB_CEILING}%, Adjustment cap: ±8%")

GTZAN_PARENT = {
    "blues":"Blues","classical":"Classical","country":"Country","disco":"Electronic",
    "hiphop":"Hip-Hop","jazz":"Jazz","metal":"Metal","pop":"Pop","reggae":"Reggae","rock":"Rock",
}
DISCOGS_PARENT_ALIAS = {
    "Funk / Soul":"R&B","Hip Hop":"Hip-Hop","Latin":"World","Stage & Screen":"Soundtrack",
    "Non-Music":None,"Children's":None,"Brass & Military":None,
}

def parse_discogs_label(label):
    parent, sub = (label.split("---",1) if "---" in label else (label, None))
    parent = parent.strip(); sub = sub.strip() if sub else None
    alias  = DISCOGS_PARENT_ALIAS.get(parent)
    if alias is None and parent in DISCOGS_PARENT_ALIAS: return None, None
    return (alias or parent), sub

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# ── Model ─────────────────────────────────────────────────────────────────────
class HybridASTDetector(nn.Module):
    def __init__(self, ast_backbone):
        super().__init__()
        self.ast = ast_backbone
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),
            nn.Conv2d(16,1,3,padding=1),nn.BatchNorm2d(1),nn.ReLU(),nn.MaxPool2d(2))
        self.residual_downsample = nn.Sequential(nn.Conv2d(1,1,1),nn.MaxPool2d(4))
        self.classifier = nn.Sequential(
            nn.Linear(768*2,512),nn.ReLU(),nn.Dropout(0.4),
            nn.Linear(512,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))

    def forward(self, x):
        r = x.unsqueeze(1)
        c = self.cnn_block2(self.cnn_block1(r))
        res = self.residual_downsample(r)
        if res.shape != c.shape:
            res = torch.nn.functional.interpolate(res, size=c.shape[2:], mode="bilinear", align_corners=False)
        c = (c + res).squeeze(1)
        if c.shape[-1] != 1024 or c.shape[-2] != 128:
            c = torch.nn.functional.interpolate(c.unsqueeze(1),(128,1024),mode="bilinear",align_corners=False).squeeze(1)
        h = self.ast(c).last_hidden_state
        return self.classifier(torch.cat([h[:,0,:], h[:,1:,:].mean(1)], dim=1))

# ══════════════════════════════════════════════════════════════════════════════
# ── PRINT CLEAR VERDICT FUNCTION ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def print_clear_verdict(verdict, ai_probability, segment_results, weighted_neural, weighted_acoustic, total_adjustment):
    """Print a highly readable verdict summary."""
    print("\n" + "=" * 70)
    print("🎵 AUDIO ANALYSIS RESULT")
    print("=" * 70)
    
    # Clear verdict with emoji
    if "Human" in verdict:
        verdict_emoji = "👤 HUMAN"
        verdict_color = "\033[92m"  # Green
    elif "AI" in verdict:
        verdict_emoji = "🤖 AI GENERATED"
        verdict_color = "\033[91m"  # Red
    else:
        verdict_emoji = "❓ NOT SURE"
        verdict_color = "\033[93m"  # Yellow
    
    print(f"\n{verdict_color}╔════════════════════════════════════════════════════════════════╗\033[0m")
    print(f"{verdict_color}║  FINAL VERDICT: {verdict_emoji:<54} ║\033[0m")
    
    # Show probability bar
    bar_length = 40
    filled = int(ai_probability / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"{verdict_color}║  AI PROBABILITY: {ai_probability:>5.1f}%  {bar}  ║\033[0m")
    
    human_pct = 100 - ai_probability
    if ai_probability >= 60:
        confidence_text = f"⚠️ This is LIKELY AI-GENERATED ({ai_probability:.1f}% confident)"
    elif human_pct >= 60:
        confidence_text = f"✓ This is LIKELY HUMAN PERFORMANCE ({human_pct:.1f}% confident)"
    else:
        confidence_text = f"❓ UNCLEAR - Could be either AI or human"
    
    print(f"{verdict_color}║  {confidence_text:<62} ║\033[0m")
    print(f"{verdict_color}╚════════════════════════════════════════════════════════════════╝\033[0m")
    
    # Segment summary
    print("\n📊 SEGMENT BREAKDOWN:")
    print("-" * 60)
    print(f"  {'':2s} {'SECTION':10s} {'AI %':>8s} {'HUMAN %':>8s} {'VERDICT':12s} {'AGREEMENT':12s}")
    print(f"  {'─' * 58}")
    
    for seg in segment_results:
        ai_pct = seg["neural_prob"] * 100
        human_pct_seg = 100 - ai_pct
        
        if ai_pct < 40:
            verdict_seg = "👤 HUMAN"
        elif ai_pct > 60:
            verdict_seg = "🤖 AI"
        else:
            verdict_seg = "❓ MIXED"
        
        corr_text = "✓ AGREES" if seg["corroborated"] else "✗ CONFLICT"
        print(f"  {corr_text[0]}  {seg['name'].upper():10s} {ai_pct:7.1f}% {human_pct_seg:7.1f}% {verdict_seg:12s} {corr_text:12s}")
    
    # Trust indicator
    print("\n🔍 TRUST INDICATORS:")
    print("-" * 60)
    
    corroborated_count = sum(1 for s in segment_results if s["corroborated"])
    conflicting_count = len(segment_results) - corroborated_count
    
    if corroborated_count >= conflicting_count:
        print(f"  ✅ {corroborated_count} segment(s) where AI and acoustic analysis AGREE")
        print(f"  ⚠️ {conflicting_count} segment(s) with CONFLICTING signals")
        if "Human" in verdict:
            print("  → System TRUSTS the HUMAN-sounding corroborated segments")
            print("  → Conflicting AI-sounding segments are IGNORED (no acoustic support)")
        elif "AI" in verdict:
            print("  → System TRUSTS the AI-sounding corroborated segments")
            print("  → Conflicting human-sounding segments are IGNORED")
    else:
        print(f"  ⚠️ {conflicting_count} segment(s) with CONFLICTING signals (majority)")
        print(f"  ✅ {corroborated_count} segment(s) where signals AGREE (minority)")
        print("  → Using weighted confidence scoring to resolve conflicts")
    
    # Scoring breakdown
    print("\n📈 SCORING BREAKDOWN:")
    print("-" * 60)
    print(f"  Neural Network (85% weight):  {weighted_neural*100:5.1f}% AI")
    print(f"  Acoustic Features (15% weight): {weighted_acoustic*100:5.1f}% AI")
    print(f"  Adjustment:                     {total_adjustment:+.1f}%")
    print(f"  {'─' * 40}")
    print(f"  FINAL AI SCORE:                {ai_probability:5.1f}%")
    
    # Simple explanation
    print("\n💡 SIMPLE EXPLANATION:")
    print("-" * 60)
    if "Human" in verdict:
        if ai_probability <= 30:
            print("  ✓ This sounds like a REAL HUMAN recording.")
            print("  ✓ Strong natural characteristics (timing, dynamics, pitch variation)")
            print("  ✓ Any AI-like sections are not acoustically supported")
        else:
            print("  ✓ This likely contains HUMAN PERFORMANCE elements.")
            print("  ✓ The overall evidence points to human-origin audio")
            print("  ⚠️ Some sections show AI-like patterns but lack acoustic confirmation")
    elif "AI" in verdict:
        print("  🤖 This sounds like AI-GENERATED audio.")
        print("  🤖 Patterns match synthetic/computer-generated content")
        if human_pct > 20:
            print("  ⚠️ Some human-like qualities detected but AI patterns dominate")
    else:
        print("  ❓ This is UNCLEAR - could be either AI or human.")
        print("  ❓ The system needs more confident signals to decide")
        print("  ❓ Consider uploading a longer or clearer audio sample")
    
    print("\n" + "=" * 70 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# ── Human Performance Detection (REDUCED IMPACT) ──────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def detect_human_performance(feat_dict):
    """
    Detects characteristics that strongly indicate human performance.
    Returns additive adjustment (reduced impact so it doesn't override neural net).
    """
    adjustment = 0.0
    reasons = []
    
    # Reduced adjustments - these were over-correcting before
    if feat_dict["beat_regularity"] < 0.65:
        adjustment -= 3.0  # Was -8.0
        reasons.append("natural timing variations")
    elif feat_dict["beat_regularity"] < 0.75:
        adjustment -= 1.5  # Was -4.0
        reasons.append("slight timing variations")
    
    if feat_dict["pitch_stability"] < 0.65:
        adjustment -= 2.5  # Was -7.0
        reasons.append("natural pitch variation")
    elif feat_dict["pitch_stability"] < 0.75:
        adjustment -= 1.0  # Was -3.0
        reasons.append("healthy pitch movement")
    
    if feat_dict["dynamic_range"] > 0.05:
        adjustment -= 2.0  # Was -6.0
        reasons.append("good dynamic range")
    elif feat_dict["dynamic_range"] > 0.035:
        adjustment -= 1.0  # Was -3.0
        reasons.append("natural dynamics")
    
    if feat_dict["noise_floor"] > 0.001:
        adjustment -= 2.0  # Was -5.0
        reasons.append("natural background presence")
    elif feat_dict["noise_floor"] > 0.0005:
        adjustment -= 1.0  # Was -2.0
    
    if feat_dict["harmonic_ratio"] < 0.60:
        adjustment -= 2.0  # Was -6.0
        reasons.append("organic harmonic content")
    elif feat_dict["harmonic_ratio"] < 0.70:
        adjustment -= 1.0  # Was -3.0
    
    # Cap the maximum reduction (was -20, now -10)
    adjustment = max(adjustment, -10.0)
    
    return adjustment, reasons

def get_genre_adjustment(genre_name):
    """Return additive adjustment for different genres (reduced impact)."""
    genre_adjustments = {
        "Electronic": -3.0,   # Was -5.0
        "Hip-Hop": -2.0,      # Was -4.0
        "Pop": -2.0,          # Was -3.0
        "Metal": 1.0,         # Was 2.0
        "Classical": 1.0,     # Was 3.0
        "Jazz": 1.0,          # Was 2.0
        "Rock": 0.0,
        "Blues": 0.5,
        "Country": 0.0,
        "Reggae": -1.0,
        "R&B": -1.0,
        "World": 0.5,
    }
    return genre_adjustments.get(genre_name, 0.0)

def compute_human_likelihood_score(feat_dict):
    """
    Compute a human-likelihood score (0-100) based on acoustic features.
    Used for display only, not for probability adjustment.
    """
    beat_reg = feat_dict["beat_regularity"]
    timing_score = 100 if beat_reg < 0.60 else (80 if beat_reg < 0.70 else (60 if beat_reg < 0.80 else (40 if beat_reg < 0.90 else 20)))
    
    pitch_stab = feat_dict["pitch_stability"]
    pitch_score = 100 if pitch_stab < 0.60 else (80 if pitch_stab < 0.70 else (60 if pitch_stab < 0.80 else (40 if pitch_stab < 0.90 else 20)))
    
    dyn_range = feat_dict["dynamic_range"]
    dynamic_score = 100 if dyn_range > 0.06 else (80 if dyn_range > 0.04 else (60 if dyn_range > 0.03 else (40 if dyn_range > 0.02 else 20)))
    
    noise = feat_dict["noise_floor"]
    noise_score = 100 if noise > 0.002 else (80 if noise > 0.001 else (60 if noise > 0.0005 else (40 if noise > 0.0002 else 20)))
    
    harmonic = feat_dict["harmonic_ratio"]
    harmonic_score = 100 if harmonic < 0.55 else (80 if harmonic < 0.65 else (60 if harmonic < 0.75 else (40 if harmonic < 0.85 else 20)))
    
    human_score = (timing_score * 0.25 + pitch_score * 0.25 + 
                   dynamic_score * 0.20 + noise_score * 0.15 + harmonic_score * 0.15)
    
    return human_score

# ══════════════════════════════════════════════════════════════════════════════
# ── Weighted Segment Analysis ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _find_chorus_by_energy(waveform, total_duration_sec, clip_sec):
    """Find the highest-energy region as a proxy for the chorus."""
    dur = total_duration_sec
    if dur <= clip_sec * 1.5:
        # Track is too short to meaningfully find a chorus — use middle
        mid = max(0.0, dur / 2.0 - clip_sec / 2.0)
        return mid
    
    arr = waveform.squeeze().numpy()
    hop_samples = int(SAMPLE_RATE * 0.5)  # 0.5s hop for energy envelope
    frame_len = int(SAMPLE_RATE * 1.0)     # 1s frames
    
    energies = []
    for start in range(0, len(arr) - frame_len, hop_samples):
        frame = arr[start:start + frame_len]
        energies.append(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
    
    if len(energies) < 3:
        return max(0.0, dur / 2.0 - clip_sec / 2.0)
    
    energies = np.array(energies)
    
    # Smooth with a moving average to find sustained loud regions (not transient peaks)
    kernel_size = min(7, len(energies))
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energies, kernel, mode='same')
    else:
        smoothed = energies
    
    # Exclude first and last 10% of the track to avoid intro/outro
    margin_frames = max(1, int(len(smoothed) * 0.10))
    search_region = smoothed[margin_frames:-margin_frames] if margin_frames < len(smoothed) // 2 else smoothed
    best_idx = int(np.argmax(search_region)) + margin_frames
    
    # Convert back to seconds
    chorus_start_sec = best_idx * 0.5
    # Center the clip on the peak
    chorus_start_sec = max(0.0, chorus_start_sec - clip_sec / 2.0)
    # Ensure we don't go past the end
    if chorus_start_sec + clip_sec > dur:
        chorus_start_sec = max(0.0, dur - clip_sec)
    
    return chorus_start_sec


def prepare_weighted_audio_analysis(waveform, total_duration_sec):
    clip_sec = DURATION_SEC
    dur = total_duration_sec
    
    segments = []
    
    # ── Intro ──
    intro_dur = min(clip_sec, dur)
    if intro_dur > 0:
        intro = waveform[:, :int(intro_dur * SAMPLE_RATE)]
        segments.append(("intro", intro, 1.0, 0.0, intro_dur))
    
    # ── Chorus (energy-based detection) ──
    if dur > clip_sec:
        chorus_start = _find_chorus_by_energy(waveform, dur, clip_sec)
        chorus_start_samples = int(chorus_start * SAMPLE_RATE)
        chorus_dur = min(clip_sec, dur - chorus_start)
        chorus_end_samples = chorus_start_samples + int(chorus_dur * SAMPLE_RATE)
        chorus = waveform[:, chorus_start_samples:chorus_end_samples]
        segments.append(("chorus", chorus, 1.5, chorus_start, chorus_start + chorus_dur))
    
    # ── Verse (second sample from ~30% of the track, avoids intro/chorus overlap) ──
    if dur > clip_sec * 3:
        verse_target = dur * 0.30
        # Make sure it doesn't overlap with intro or chorus
        if len(segments) >= 2:
            chorus_start_sec = segments[1][3]
            chorus_end_sec = segments[1][4]
            # If verse would overlap chorus, shift to 70% instead
            if abs(verse_target - chorus_start_sec) < clip_sec:
                verse_target = dur * 0.70
        verse_start = max(clip_sec + 1, verse_target)
        verse_start = min(verse_start, dur - clip_sec)
        verse_start_samples = int(verse_start * SAMPLE_RATE)
        verse_end_samples = verse_start_samples + int(clip_sec * SAMPLE_RATE)
        verse = waveform[:, verse_start_samples:verse_end_samples]
        segments.append(("verse", verse, 1.2, verse_start, verse_start + clip_sec))
    
    # ── Ending ──
    if dur > clip_sec:
        ending_start = max(0.0, dur - clip_sec)
        ending_start_samples = int(ending_start * SAMPLE_RATE)
        ending = waveform[:, ending_start_samples:]
        actual_dur = ending.shape[1] / SAMPLE_RATE
        segments.append(("ending", ending, 0.8, ending_start, ending_start + actual_dur))
    
    if len(segments) == 0:
        full = waveform[:, :min(waveform.shape[1], SAMPLES_NEEDED)]
        segments.append(("full", full, 1.0, 0.0, min(dur, DURATION_SEC)))
    
    return segments

def analyze_weighted_segments(waveform, total_duration_sec, model, mel_transform, device):
    segments = prepare_weighted_audio_analysis(waveform, total_duration_sec)
    
    segment_results = []
    
    for seg_name, seg_waveform, weight, start_sec, end_sec in segments:
        if seg_waveform.shape[1] < SAMPLES_NEEDED:
            seg_waveform = torch.nn.functional.pad(seg_waveform, (0, SAMPLES_NEEDED - seg_waveform.shape[1]))
        elif seg_waveform.shape[1] > SAMPLES_NEEDED:
            seg_waveform = seg_waveform[:, :SAMPLES_NEEDED]
        
        mel_input = mel_from_chunk(seg_waveform, mel_transform)
        
        with torch.no_grad():
            raw_logit = float(model(mel_input.to(device)).squeeze().item())
            neural_prob = torch.sigmoid(torch.tensor(raw_logit)).item()
        
        chunk_feat = compute_chunk_features(seg_waveform)
        acoustic_prob = acoustic_composite_score(chunk_feat) / 100.0
        human_score = compute_human_likelihood_score(chunk_feat)
        
        # Confidence: higher when the model is more decisive (further from 0.5)
        confidence = 0.5 + min(0.5, abs(neural_prob - 0.5))
        
        # Corroboration: does the human acoustic analysis agree with neural direction?
        # human_score > 60 means acoustic features look human
        # neural_prob < 0.5 means neural says human
        # When both agree, that's corroborated evidence
        neural_says_human = neural_prob < 0.5
        acoustic_says_human = human_score > 60
        corroborated = neural_says_human == acoustic_says_human
        
        # Evidence quality: neural confidence × corroboration bonus
        evidence_quality = confidence * (1.15 if corroborated else 0.85)
        
        segment_results.append({
            "name": seg_name,
            "weight": weight,
            "neural_prob": neural_prob,
            "acoustic_prob": acoustic_prob,
            "human_score": human_score,
            "confidence": confidence,
            "evidence_quality": evidence_quality,
            "corroborated": corroborated,
            "raw_logit": raw_logit,
            "features": chunk_feat,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2)
        })
        
        corr_str = "✓ corroborated" if corroborated else "✗ conflicting"
        print(f"  Segment [{seg_name}] {start_sec:.1f}-{end_sec:.1f}s: neural={neural_prob:.1%}, acoustic={acoustic_prob:.1%}, human={human_score:.0f}, conf={confidence:.2f}, {corr_str}")
    
    # ── Aggregation Strategy ──
    # Weight by: segment importance weight × neural confidence × evidence quality
    # This naturally favors the chorus (weight=1.5) and segments where
    # neural and acoustic signals agree
    neural_probs = [s["neural_prob"] for s in segment_results]
    n_segments = len(segment_results)
    
    for seg in segment_results:
        seg["effective_weight"] = seg["weight"] * seg["evidence_quality"]
    
    total_weight = sum(s["effective_weight"] for s in segment_results)
    
    if total_weight > 0:
        weighted_neural = sum(s["effective_weight"] * s["neural_prob"] for s in segment_results) / total_weight
        weighted_acoustic = sum(s["effective_weight"] * s["acoustic_prob"] for s in segment_results) / total_weight
    else:
        weighted_neural = np.mean(neural_probs)
        weighted_acoustic = np.mean([s["acoustic_prob"] for s in segment_results])
    
    # ── High disagreement detection ──
    neural_std = np.std(neural_probs) if len(neural_probs) > 1 else 0.0
    agreement_score = float(np.clip(1.0 - neural_std * 3, 0.0, 1.0))
    
    # When segments strongly disagree (std > 0.20), check if any segment
    # has BOTH high confidence AND acoustic corroboration — that's a stronger 
    # signal than a segment that only has confidence
    if neural_std > 0.20 and n_segments >= 3:
        corroborated_segs = [s for s in segment_results if s["corroborated"]]
        uncorroborated_segs = [s for s in segment_results if not s["corroborated"]]
        
        if corroborated_segs and uncorroborated_segs:
            # Segments with neural-acoustic agreement are more trustworthy
            # Boost their contribution
            corr_neural = np.mean([s["neural_prob"] for s in corroborated_segs])
            blend_factor = min(0.35, (neural_std - 0.20) * 1.5)
            weighted_neural = weighted_neural * (1 - blend_factor) + corr_neural * blend_factor
            print(f"  High disagreement (std={neural_std:.3f}): boosting {len(corroborated_segs)} corroborated segment(s) by {blend_factor:.1%}")
        else:
            # All segments are either corroborated or uncorroborated — 
            # lean toward the most confident one
            best_conf_seg = max(segment_results, key=lambda s: s["confidence"])
            blend_factor = min(0.25, (neural_std - 0.20) * 1.0)
            weighted_neural = weighted_neural * (1 - blend_factor) + best_conf_seg["neural_prob"] * blend_factor
            print(f"  High disagreement (std={neural_std:.3f}): leaning {blend_factor:.1%} toward [{best_conf_seg['name']}] (most confident)")
    
    # ── Human-score safety check ──
    # If the MOST human-sounding segment (high human_score + low neural) exists 
    # and is being outvoted, apply a conservatism pull toward 50%
    # This addresses the scenario where heavily-produced intros/outros 
    # overpower a clearly human vocal section
    if n_segments >= 3 and neural_std > 0.25:
        min_neural_seg = min(segment_results, key=lambda s: s["neural_prob"])
        if min_neural_seg["human_score"] > 70 and min_neural_seg["neural_prob"] < 0.25:
            # Strong human signal in at least one segment — don't be overconfident about AI
            if weighted_neural > 0.65:
                pull = min(0.12, (weighted_neural - 0.65) * 0.5)
                weighted_neural -= pull
                print(f"  Conservatism pull: segment [{min_neural_seg['name']}] has strong human signal (neural={min_neural_seg['neural_prob']:.1%}, human_score={min_neural_seg['human_score']:.0f}), pulling weighted_neural down by {pull:.3f}")
    
    best_segment = max(segment_results, key=lambda s: s["evidence_quality"] * s["weight"])
    
    return weighted_neural, weighted_acoustic, agreement_score, segment_results, best_segment

# ══════════════════════════════════════════════════════════════════════════════
# ── Calibrated Probability (NEURAL DOMINANT) ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibrated_probability(neural_prob, acoustic_prob, feat_dict, genre_result=None, agreement_score=1.0):
    """
    Neural network is the primary signal. Acoustic adjustments are small and
    agreement-scaled — they cannot flip a confident neural prediction.
    
    Key fixes:
    - No double-counting of features (human_performance detection removed, 
      individual feature adjustments handle this)
    - Adjustments are scaled by (1 - agreement_score) so high-agreement segments
      get minimal correction
    - Total adjustment is hard-capped relative to neural confidence
    """
    # Neural network is the primary signal (80% weight)
    base_prob = (neural_prob * NEURAL_WEIGHT + acoustic_prob * ACOUSTIC_WEIGHT) * 100
    
    adjustment = 0.0
    human_reasons = []
    
    # ── Individual feature adjustments (small, no double-counting) ──
    
    # Beat regularity
    beat_reg = feat_dict["beat_regularity"]
    if beat_reg < 0.60:
        adj = -(0.60 - beat_reg) * 8
        adjustment += adj
        human_reasons.append("natural timing variations")
    elif beat_reg > 0.93:
        adjustment += (beat_reg - 0.93) * 12
    
    # Pitch stability
    pitch_stab = feat_dict["pitch_stability"]
    if pitch_stab < 0.60:
        adj = -(0.60 - pitch_stab) * 6
        adjustment += adj
        human_reasons.append("natural pitch variation")
    elif pitch_stab > 0.93:
        adjustment += (pitch_stab - 0.93) * 10
    
    # Dynamic range
    dyn_range = feat_dict["dynamic_range"]
    if dyn_range > 0.06:
        adjustment -= 1.5
        human_reasons.append("good dynamic range")
    elif dyn_range < 0.02:
        adjustment += 1.5
    
    # Noise floor
    noise = feat_dict["noise_floor"]
    if noise > 0.002:
        adjustment -= min(2.0, noise * 500)
        human_reasons.append("natural background presence")
    elif noise < 0.0001:
        adjustment += 1.5
    
    # Harmonic ratio
    harmonic = feat_dict["harmonic_ratio"]
    if harmonic > 0.90:
        adjustment += (harmonic - 0.90) * 15
    elif harmonic < 0.50:
        adjustment -= (0.50 - harmonic) * 8
        human_reasons.append("organic harmonic content")
    
    # ── Genre adjustment (small) ──
    if genre_result and genre_result.get("top"):
        genre_name = genre_result["top"]["label"]
        genre_adj = get_genre_adjustment(genre_name)
        adjustment += genre_adj
        if abs(genre_adj) > 0.5:
            print(f"  Genre adjustment: {genre_name} = {genre_adj:+.1f}%")
    
    # ── Scale adjustments by disagreement ──
    # When segments agree strongly, trust the neural score more (less adjustment)
    # When segments disagree, allow more adjustment
    disagreement_factor = max(0.3, 1.0 - agreement_score)
    adjustment *= disagreement_factor
    
    # ── Hard cap: adjustments can't flip a confident prediction ──
    # If neural is >65% or <35%, cap adjustment to prevent flipping past 50%
    neural_pct = neural_prob * 100
    if neural_pct > 65:
        # Don't let adjustment push below 50
        adjustment = max(adjustment, -(neural_pct - 52))
    elif neural_pct < 35:
        # Don't let adjustment push above 50
        adjustment = min(adjustment, (48 - neural_pct))
    
    # Global cap
    adjustment = max(-8.0, min(8.0, adjustment))
    
    calibrated = base_prob + adjustment
    calibrated = max(PROB_FLOOR, min(PROB_CEILING, calibrated))
    
    return calibrated, adjustment, human_reasons

# ── XAI deep (unchanged) ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _xd_to_numpy(t):
    return t.detach().cpu().float().numpy()

def _xd_norm01(a):
    lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

def _xd_frame_to_sec(idx):
    return round(idx * HOP_LENGTH / SAMPLE_RATE, 3)

def _xd_bin_to_hz(idx, n_mels=128):
    mel_min = 2595 * math.log10(1 + 0 / 700)
    mel_max = 2595 * math.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel_v   = mel_min + (mel_max - mel_min) * idx / (n_mels - 1)
    return round(700 * (10 ** (mel_v / 2595) - 1), 1)

def _xd_topk(arr, k=5):
    k = min(k, len(arr))
    return list(np.argsort(arr)[::-1][:k])

def _xd_saliency(mel_input, model):
    model.eval()
    x = mel_input.clone().detach().requires_grad_(True)
    model(x).squeeze().backward()
    grad = _xd_to_numpy(x.grad.abs()).squeeze(0)
    fi   = _xd_norm01(grad.mean(axis=1))
    bi   = _xd_norm01(grad.mean(axis=0))
    return {
        "heatmap":    _xd_norm01(grad).tolist(),
        "top_frames": [{"frame_idx": i, "importance": round(float(fi[i]), 4),
                        "time_sec": _xd_frame_to_sec(i)} for i in _xd_topk(fi)],
        "top_bins":   [{"bin_idx": i,   "importance": round(float(bi[i]), 4),
                        "freq_hz": _xd_bin_to_hz(i)}   for i in _xd_topk(bi)],
    }

def _xd_integrated_gradients(mel_input, model, n_steps=XAI_IG_STEPS):
    model.eval()
    x    = mel_input
    base = torch.zeros_like(x)
    grads = []
    for step in range(n_steps + 1):
        interp = (base + (step / n_steps) * (x - base)).requires_grad_(True)
        model(interp).squeeze().backward()
        grads.append(_xd_to_numpy(interp.grad))
    avg_grads = np.mean(np.stack(grads, axis=0), axis=0)
    attrs     = _xd_to_numpy(x - base) * avg_grads
    attrs_abs = np.abs(attrs).squeeze(0)
    fi  = _xd_norm01(attrs_abs.mean(axis=1))
    bi  = _xd_norm01(attrs_abs.mean(axis=0))
    return {
        "heatmap":    _xd_norm01(attrs_abs).tolist(),
        "top_frames": [{"frame_idx": i, "importance": round(float(fi[i]), 4),
                        "time_sec": _xd_frame_to_sec(i)} for i in _xd_topk(fi)],
        "top_bins":   [{"bin_idx": i,   "importance": round(float(bi[i]), 4),
                        "freq_hz": _xd_bin_to_hz(i)}   for i in _xd_topk(bi)],
        "n_steps": n_steps,
    }

def _xd_attention_rollout(mel_input, model):
    model.eval()
    with torch.no_grad():
        ast_out = model.ast(mel_input, output_attentions=True)
    all_attns = getattr(ast_out, "attentions", None)
    if not all_attns:
        return {"heatmap": [], "rollout": [], "top_tokens": [], "n_heads": 0,
                "note": "Attention weights unavailable for this model version."}
    mats    = [_xd_to_numpy(a.squeeze(0)) for a in all_attns]
    n_heads = mats[0].shape[0]
    seq_len = mats[0].shape[-1]
    rolled = np.eye(seq_len)
    for m in mats:
        hm   = m.mean(axis=0)
        aug  = 0.5 * hm + 0.5 * np.eye(seq_len)
        aug /= aug.sum(axis=-1, keepdims=True)
        rolled = rolled @ aug
    cls_row  = _xd_norm01(rolled[0, 1:])
    top_toks = [{"token_idx": i, "importance": round(float(cls_row[i]), 4)}
                for i in _xd_topk(cls_row)]
    n_patches = len(cls_row)
    patch_t   = int(math.sqrt(n_patches * 1024 / 128))
    patch_f   = n_patches // max(patch_t, 1)
    if patch_t > 0 and patch_f > 0 and patch_t * patch_f <= n_patches:
        grid = cls_row[: patch_t * patch_f].reshape(patch_t, patch_f)
        tsr  = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
        up   = F.interpolate(tsr, size=(1024, 128), mode="bilinear", align_corners=False)
        hmap = _xd_norm01(_xd_to_numpy(up.squeeze())).tolist()
    else:
        row  = np.interp(np.linspace(0, len(cls_row) - 1, 128),
                         np.arange(len(cls_row)), cls_row)
        hmap = _xd_norm01(np.tile(row, (1024, 1))).tolist()
    return {"heatmap": hmap, "rollout": _xd_norm01(rolled).tolist(),
            "top_tokens": top_toks, "n_heads": n_heads}

def _xd_lime(mel_input, model,
             n_segs_t=XAI_LIME_SEGS_T, n_segs_f=XAI_LIME_SEGS_F,
             n_samples=XAI_LIME_SAMPLES, alpha=XAI_LIME_ALPHA):
    model.eval()
    x  = mel_input
    T, Fm = x.shape[1], x.shape[2]
    t_edges = np.linspace(0, T,  n_segs_t + 1, dtype=int)
    f_edges = np.linspace(0, Fm, n_segs_f + 1, dtype=int)
    n_segs  = n_segs_t * n_segs_f
    fill    = float(x.mean().item())
    rng    = np.random.default_rng(42)
    masks  = rng.integers(0, 2, size=(n_samples, n_segs), dtype=np.uint8)
    logits = np.zeros(n_samples, dtype=np.float32)
    with torch.no_grad():
        base_logit = float(model(x).squeeze().item())
        for i, mask in enumerate(masks):
            p = x.clone(); seg = 0
            for ti in range(n_segs_t):
                for fi in range(n_segs_f):
                    if mask[seg] == 0:
                        p[0, t_edges[ti]:t_edges[ti+1], f_edges[fi]:f_edges[fi+1]] = fill
                    seg += 1
            logits[i] = float(model(p).squeeze().item())
    X   = masks.astype(np.float32)
    y   = logits
    XtX = X.T @ X + alpha * np.eye(n_segs, dtype=np.float32)
    try:    coeffs = np.linalg.solve(XtX, X.T @ y)
    except: coeffs = np.zeros(n_segs, dtype=np.float32)
    y_hat  = X @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    r2     = float(np.clip(1 - ss_res / ss_tot, 0.0, 1.0))
    segments = []
    with torch.no_grad():
        for seg in range(n_segs):
            ti, fi = divmod(seg, n_segs_f)
            t0, t1 = int(t_edges[ti]), int(t_edges[ti+1])
            f0, f1 = int(f_edges[fi]), int(f_edges[fi+1])
            masked = x.clone()
            masked[0, t0:t1, f0:f1] = fill
            delta = round(float(base_logit - model(masked).squeeze().item()), 4)
            segments.append({
                "segment_id":         seg,
                "importance":         round(float(coeffs[seg]), 4),
                "time_start_sec":     _xd_frame_to_sec(t0),
                "time_end_sec":       _xd_frame_to_sec(t1),
                "freq_start_hz":      _xd_bin_to_hz(f0),
                "freq_end_hz":        _xd_bin_to_hz(f1),
                "masked_score_delta": delta,
            })
    segments.sort(key=lambda s: abs(s["importance"]), reverse=True)
    return {"segments": segments, "n_segments": n_segs,
            "n_samples": n_samples, "r2_score": round(r2, 4)}

def _xd_summary(sal, ig, attn, lime):
    peak_sal_sec = sal["top_frames"][0]["time_sec"] if sal.get("top_frames") else None
    peak_attn_sec = None
    if attn.get("heatmap") and len(attn["heatmap"]) > 0:
        hmap = np.array(attn["heatmap"])
        peak_attn_sec = _xd_frame_to_sec(int(np.argmax(hmap.mean(axis=1))))
    most_sus = None
    pos_segs = [s for s in lime.get("segments", []) if s["importance"] > 0]
    if pos_segs:
        b = max(pos_segs, key=lambda s: s["importance"])
        most_sus = {k: b[k] for k in
                    ("time_start_sec","time_end_sec","freq_start_hz","freq_end_hz","importance")}
    agreement = 0.5
    if sal.get("top_frames") and ig.get("top_frames"):
        diff = abs(sal["top_frames"][0]["time_sec"] - ig["top_frames"][0]["time_sec"])
        agreement = round(float(np.clip(1.0 - diff * 2 / _xd_frame_to_sec(1024), 0.0, 1.0)), 3)
    return {
        "most_suspicious_region":  most_sus,
        "peak_attention_time_sec": peak_attn_sec,
        "peak_saliency_time_sec":  peak_sal_sec,
        "method_agreement":        agreement,
    }

def compute_xai_deep(mel_input, model, device):
    model.eval()
    x = mel_input.to(device)
    with torch.no_grad():
        raw_logit = float(model(x).squeeze().item())
    out = {}
    try:    out["saliency"]             = _xd_saliency(x, model)
    except Exception as e: out["saliency"] = {"error": str(e)}
    try:    out["integrated_gradients"] = _xd_integrated_gradients(x, model)
    except Exception as e: out["integrated_gradients"] = {"error": str(e)}
    try:    out["attention"]            = _xd_attention_rollout(x, model)
    except Exception as e: out["attention"] = {"error": str(e)}
    try:    out["lime"]                 = _xd_lime(x, model)
    except Exception as e: out["lime"] = {"error": str(e)}
    out["summary"] = _xd_summary(
        out.get("saliency", {}), out.get("integrated_gradients", {}),
        out.get("attention", {}), out.get("lime", {}),
    )
    return raw_logit, out

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_professional_production(feat):
    conds = [feat["dynamic_range"]<0.04, feat["spectral_flatness"]>0.8, feat["temporal_flux"]<0.8]
    reasons = [r for r,c in zip(["professional compression","full frequency spectrum","consistent energy"],conds) if c]
    adjustment = -len(reasons) * 1.0 if len(reasons) >= 2 else 0.0
    return adjustment, reasons

def clipping_weight(chunk_tensor):
    cr = float(np.mean(np.abs(chunk_tensor.squeeze().numpy()) >= CLIP_THRESHOLD))
    if cr <= CLIP_RATIO_SOFT: return 1.0, cr
    if cr >= CLIP_RATIO_HARD: return CLIP_MIN_WEIGHT, cr
    t = (cr-CLIP_RATIO_SOFT)/(CLIP_RATIO_HARD-CLIP_RATIO_SOFT)
    return 1.0 - t*(1.0-CLIP_MIN_WEIGHT), cr

def detect_codec_compression(waveform_tensor, sr=SAMPLE_RATE):
    arr = waveform_tensor.squeeze().numpy().astype(np.float32)
    n   = len(arr)
    if n < sr // 2:
        return False, 1.0, 0.0, {"reason": "too short to evaluate"}
    fft_out  = np.fft.rfft(arr, n=min(n, 65536))
    freqs    = np.fft.rfftfreq(min(n, 65536), d=1.0 / sr)
    mag_sq   = np.abs(fft_out) ** 2
    total_energy = np.sum(mag_sq) + 1e-12
    hf_energy    = np.sum(mag_sq[freqs > 7000])
    hf_ratio     = float(hf_energy / total_energy)
    band_lo = mag_sq[(freqs >= 6000) & (freqs < 7500)]
    band_hi = mag_sq[(freqs >= 7500) & (freqs <= 8000)]
    shelf_ratio = (np.mean(band_hi) / (np.mean(band_lo) + 1e-12)) if band_lo.size and band_hi.size else 1.0
    region = mag_sq[(freqs >= 4000) & (freqs <= 8000)] + 1e-10
    region_flatness = float(np.exp(np.mean(np.log(region))) / np.mean(region))
    evidence = 0.0
    if hf_ratio < CODEC_HF_RATIO_THRESHOLD:       evidence += 0.5
    if shelf_ratio < 0.15:                         evidence += 0.25
    if region_flatness > 0.60:                     evidence += 0.15
    if hf_ratio < 0.005:                           evidence += 0.10
    is_compressed = evidence >= 0.5
    confidence    = float(np.clip(evidence, 0.0, 1.0))
    details = {
        "hf_ratio":        round(hf_ratio, 5),
        "shelf_ratio":     round(float(shelf_ratio), 4),
        "region_flatness": round(region_flatness, 4),
        "evidence_score":  round(evidence, 3),
    }
    return is_compressed, hf_ratio, confidence, details

def apply_codec_penalty(ai_probability, is_compressed, codec_confidence):
    if not is_compressed:
        return ai_probability, {}
    penalty = min(3.0, codec_confidence * 5)  # Reduced penalty
    adjusted = max(PROB_FLOOR, ai_probability - penalty)
    penalty_info = {
        "applied": True,
        "confidence": round(codec_confidence, 3),
        "penalty_amount": round(penalty, 1),
        "original_score": ai_probability,
        "adjusted_score": adjusted,
        "reason": "Lossy codec compression detected.",
    }
    return adjusted, penalty_info

# ── Genre ─────────────────────────────────────────────────────────────────────
def _sample_windows(total, win, n):
    margin = max(0, int(total*0.05)); usable = total - 2*margin - win
    if usable <= 0: return [max(0, total//2 - win//2)]
    step = usable / max(n-1, 1)
    return [min(margin+round(i*step), total-win) for i in range(n)]

def _clip_window(wav, start, win):
    clip = wav[:, start:min(wav.shape[1], start+win)].squeeze(0).numpy().astype(np.float32)
    return np.pad(clip, (0, max(0, win-clip.shape[0])))

def _run_model(pipeline_fn, waveform, win_secs, label_fn):
    if pipeline_fn is None: return []
    win = GENRE_SAMPLE_RATE * win_secs; results = []
    for start in _sample_windows(waveform.shape[1], win, GENRE_N_WINDOWS):
        try:
            raw = pipeline_fn({"raw":_clip_window(waveform,start,win),"sampling_rate":GENRE_SAMPLE_RATE},
                              top_k=50 if win_secs==GENRE_WIN_SECONDARY else None)
            results.append([e for e in (label_fn(i) for i in raw) if e])
        except Exception as e:
            print(f"  [genre] window {start}: {e}")
    return results

def _primary_label(item):
    lbl = item["label"].lower().replace(" ","").replace("-","")
    parent = GTZAN_PARENT.get(item["label"].lower(), GTZAN_PARENT.get(lbl, item["label"].title()))
    return {"label":parent,"score":float(item["score"]),"parent":parent,"sub":None}

def _secondary_label(item):
    parent, sub = parse_discogs_label(item["label"])
    if parent is None: return None
    return {"label":(f"{parent} — {sub}" if sub else parent),"score":float(item["score"]),"parent":parent,"sub":sub}

def _aggregate(windows):
    if not windows: return {"parent_scores":{},"subgenre_scores":{},"raw":[]}
    n = len(windows); pa = {}; sa = {}
    for w in windows:
        total = sum(i["score"] for i in w) or 1.0
        for i in w:
            s=i["score"]/total; p=i["parent"]; k=(p,i["sub"])
            pa[p]=pa.get(p,0)+s; sa[k]=sa.get(k,0)+s
    pt=sum(pa.values()) or 1.0; st=sum(sa.values()) or 1.0
    raw=sorted([{"label":(f"{p} — {s}" if s else p),"score":round(v/st*100,1),"parent":p,"sub":s}
                for (p,s),v in sa.items()],key=lambda x:x["score"],reverse=True)
    return {"parent_scores":{k:round(v/n/pt*100,1) for k,v in pa.items()},
            "subgenre_scores":sa,"raw":raw}

def _ensemble(pri, sec):
    pp=pri["parent_scores"]; sp=sec["parent_scores"]; sg=sec["subgenre_scores"]
    comb={p:(0.4*pp.get(p,0)+0.6*sp.get(p,0)) if pp.get(p) and sp.get(p)
            else 0.7*pp.get(p,0) or 0.7*sp.get(p,0) for p in set(pp)|set(sp)}
    pt=sum(comb.values()) or 1.0
    ppct={k:round(v/pt*100,1) for k,v in comb.items()}
    spct={}
    for (p,s),v in sg.items():
        st=sum(v2 for (p2,_),v2 in sg.items() if p2==p) or 1.0
        spct[(p,s)]=round(v/st*comb.get(p,0)/pt*100,1)
    return {"parent_pct":ppct,"subgenre_pct":spct}

def _stability(windows, top):
    scores=[]
    for w in windows:
        total=sum(i["score"] for i in w) or 1.0
        scores.append(sum(i["score"] for i in w if i.get("parent","").lower()==top.lower())/total)
    if len(scores)<2: return 0.5
    mean=np.mean(scores)
    return 0.3 if mean<0.05 else float(np.clip(1.0-np.std(scores)/(mean+1e-6),0.0,1.0))

def classify_genre(waveform):
    pw=_run_model(genre_pipeline_primary, waveform,GENRE_WIN_PRIMARY, _primary_label)
    sw=_run_model(genre_pipeline_secondary,waveform,GENRE_WIN_SECONDARY,_secondary_label)
    pa=_aggregate(pw); sa=_aggregate(sw)
    if not pa["parent_scores"] and not sa["parent_scores"]: return None
    ens=_ensemble(pa,sa); ppct=ens["parent_pct"]; spct=ens["subgenre_pct"]
    if not ppct: return None
    sorted_p=sorted(ppct.items(),key=lambda x:x[1],reverse=True)
    top,top_score=sorted_p[0]
    subs={}
    for (p,s),pct in spct.items():
        if s: subs.setdefault(p,[]).append({"label":s,"score":pct})
    for p in subs: subs[p]=sorted(subs[p],key=lambda x:x["score"],reverse=True)[:GENRE_TOP_SUBGENRES]
    top_sub=None
    if spct:
        bk=max(spct,key=lambda k:spct[k])
        if bk[1]: top_sub={"parent":bk[0],"label":bk[1],"score":round(spct[bk],1)}
    stab=_stability(pw+sw,top); sec=sorted_p[1][1] if len(sorted_p)>1 else 0.0; margin=top_score-sec
    if stab>=0.75 and margin>=15:   cn="High confidence — consistent across the track"
    elif stab>=0.50 and margin>=8:  cn="Moderate confidence — mostly consistent"
    elif margin<5:                  cn="Low confidence — genre boundary is ambiguous"
    else:                           cn="Mixed signal — track may blend multiple genres"
    sources=([f"Primary ({GENRE_MODEL_PRIMARY.split('/')[1]}): {len(pw)} windows"] if pw else [])+\
            ([f"Secondary ({GENRE_MODEL_SECONDARY.split('/')[1]}): {len(sw)} windows"] if sw else [])
    return {"top":{"label":top,"score":round(top_score,1)},"all":[{"label":p,"score":s} for p,s in sorted_p[:10]],
            "subgenres":subs,"top_subgenre":top_sub,"sources":sources,"window_count":len(pw+sw),
            "stability":round(stab*100,1),"confidence_note":cn,"primary_used":bool(pw),"secondary_used":bool(sw)}

# ── Audio conversion ──────────────────────────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=128)

def mel_from_chunk(chunk, mel_transform):
    m = mel_transform(chunk)
    m = torchaudio.functional.amplitude_to_DB(m, 10.0, 1e-10, 0.0, 80.0)
    m = (m - m.mean()) / (m.std() + 1e-9)
    m = m.squeeze(0).transpose(0, 1)
    T = m.shape[0]
    if T < 1024:
        m = torch.nn.functional.pad(m, (0, 0, 0, 1024 - T))
    elif T > 1024:
        m = m[:1024, :]
    return m.unsqueeze(0)

def convert_to_16k_wav(input_path, original_filename=None):
    base="".join(c for c in os.path.splitext(os.path.basename(original_filename or input_path))[0]
                 if c.isalnum() or c in (" ","-","_")).rstrip()
    os.makedirs(CONVERTED_WAV_DIR, exist_ok=True)
    out=os.path.join(CONVERTED_WAV_DIR,f"{base}_16khz.wav")
    try:
        subprocess.run(["ffmpeg","-i",input_path,"-ar","16000","-ac","1","-c:a","pcm_s16le","-y",out],
                       capture_output=True, check=True, encoding='utf-8', errors='replace')
        dur=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                            "-of","default=noprint_wrappers=1:nokey=1",out],
                           capture_output=True, text=True, encoding='utf-8', errors='replace')
        return out, float(dur.stdout.strip()) if dur.stdout.strip() else 0
    except subprocess.CalledProcessError:
        try: wav,sr=torchaudio.load(input_path)
        except Exception:
            data,sr=sf.read(input_path,always_2d=True); wav=torch.from_numpy(data.T).float()
        if wav.shape[0]>1: wav=wav.mean(0,keepdim=True)
        if sr!=SAMPLE_RATE: wav=torchaudio.transforms.Resample(sr,SAMPLE_RATE)(wav)
        peak=wav.abs().max().item()
        if peak>1e-6: wav=wav/peak
        torchaudio.save(out,wav,SAMPLE_RATE,encoding="PCM_S",bits_per_sample=16)
        return out, wav.shape[1]/SAMPLE_RATE

# ── Acoustic features ─────────────────────────────────────────────────────────
def _fft(a,n=1024): return np.abs(np.fft.rfft(a,n=n)), np.fft.rfftfreq(n,d=1/SAMPLE_RATE)

def compute_spectral_flatness(a):
    if np.sqrt(np.mean(a**2))<1e-6: return 0.0
    spec=_fft(a)[0]**2+1e-10
    return float(min(np.exp(np.mean(np.log(spec)))/np.mean(spec),0.70))

def compute_dynamic_range(a,fl=1600):
    f=[a[i:i+fl] for i in range(0,len(a)-fl,fl)]
    return float(np.std([np.sqrt(np.mean(x**2)+1e-9) for x in f])) if f else 0.0

def compute_zero_crossing_rate(a):
    return float(np.sum(np.abs(np.diff(np.sign(a))))/2/len(a))

def compute_spectral_centroid(a):
    mag,frq=_fft(a)
    return float(np.sum(frq*mag)/(np.sum(mag)+1e-10)/(SAMPLE_RATE/2))

def compute_spectral_rolloff(a,rp=0.85):
    mag,frq=_fft(a); cs=np.cumsum(mag**2)
    return float(frq[min(np.searchsorted(cs,rp*cs[-1]),len(frq)-1)]/(SAMPLE_RATE/2))

def compute_temporal_flux(a,fl=1600):
    f=[a[i:i+fl] for i in range(0,len(a)-fl,fl)]
    if len(f)<2: return 0.0
    sp=[np.abs(np.fft.rfft(x,n=min(1024,fl))) for x in f]
    return float(np.mean([np.sum((sp[i+1]-sp[i])**2) for i in range(len(sp)-1)]))

def compute_noise_floor(a,fl=1600):
    f=[a[i:i+fl] for i in range(0,len(a)-fl,fl)]
    if not f: return 0.0
    rms=sorted(float(np.sqrt(np.mean(x**2)+1e-12)) for x in f)
    return float(np.mean(rms[:max(1,len(rms)//10)]))

def compute_harmonic_ratio(a):
    if np.sqrt(np.mean(a**2))<1e-6: return 0.0
    mag=np.abs(np.fft.rfft(a[:min(4096,len(a))],n=min(4096,len(a))))**2
    return float(np.clip(np.sum(mag[mag>=np.percentile(mag,95)])/(np.sum(mag)+1e-12),0,1))

def compute_beat_regularity(a,sr=SAMPLE_RATE):
    fl=512; hop=256
    f=[a[i:i+fl] for i in range(0,len(a)-fl,hop)]
    if len(f)<8: return 0.5
    sp=[np.abs(np.fft.rfft(x,n=fl)) for x in f]
    onset=np.array([max(0,np.sum(sp[i+1]-sp[i])) for i in range(len(sp)-1)])
    onset=onset/(onset.max()+1e-9)
    ac=np.correlate(onset,onset,mode='full')[len(onset)-1:]; ac=ac/(ac[0]+1e-9)
    lo,hi=int(0.3*sr/hop),min(int(2.0*sr/hop),len(ac)-1)
    return float(np.clip(np.max(ac[lo:hi]) if lo<hi else 0.5,0,1))

def compute_pitch_stability(a,sr=SAMPLE_RATE):
    fl=2048; hop=1024
    f=[a[i:i+fl] for i in range(0,len(a)-fl,hop)]
    if len(f)<4: return 0.5
    freqs=[]
    for x in f:
        mag=np.abs(np.fft.rfft(x,n=fl)); frq=np.fft.rfftfreq(fl,d=1/sr)
        mask=(frq>=80)&(frq<=4000)
        if np.any(mask): freqs.append(frq[mask][np.argmax(mag[mask])])
    if len(freqs)<3: return 0.5
    mean=np.mean(freqs)
    return 0.5 if mean<1e-3 else float(np.clip(1.0-np.std(freqs)/(mean+1e-6)*4,0,1))

def compute_chunk_features(chunk_tensor):
    arr=chunk_tensor.squeeze().numpy()
    rms=float(np.sqrt(np.mean(arr**2))+1e-9)
    nrms=np.clip(arr/rms*0.1,-1.0,1.0); namp=np.clip(arr,-1.0,1.0)
    return {
        "spectral_flatness": compute_spectral_flatness(nrms),
        "dynamic_range":     compute_dynamic_range(namp),
        "zero_crossing_rate":compute_zero_crossing_rate(nrms),
        "spectral_centroid": compute_spectral_centroid(namp),
        "spectral_rolloff":  compute_spectral_rolloff(namp),
        "temporal_flux":     compute_temporal_flux(namp),
        "noise_floor":       compute_noise_floor(namp),
        "harmonic_ratio":    compute_harmonic_ratio(namp),
        "beat_regularity":   compute_beat_regularity(namp),
        "pitch_stability":   compute_pitch_stability(namp),
    }

# ── Scoring functions ────────────────────────────────────────────────────────
def _score(v,steps):
    for thr,s in steps:
        if v<thr: return s
    return steps[-1][1]

def sf_score(v): return _score(v,[(0.005,52),(0.04,44),(0.15,48),(0.30,55),(1,62)])
def dr_score(v): return _score(v,[(0.0005,62),(0.001,56),(0.006,46),(0.015,44),(1,52)])
def sc_score(v): return _score(v,[(0.02,58),(0.10,52),(0.30,44),(0.50,48),(1,54)])
def sr_score(v): return _score(v,[(0.01,58),(0.05,52),(0.25,44),(0.50,48),(1,54)])
def nf_score(v): return _score(v,[(0.0002,62),(0.001,56),(0.004,44),(0.015,46),(1,52)])
def hr_score(v): return _score(v,[(0.35,52),(0.55,46),(0.75,44),(0.90,54),(1,62)])
def br_score(v): return _score(v,[(0.25,52),(0.50,47),(0.75,44),(0.90,54),(1,60)])
def ps_score(v): return _score(v,[(0.40,52),(0.60,46),(0.80,44),(0.92,56),(1,62)])

def acoustic_composite_score(feat):
    tfs=_score(feat["temporal_flux"],  [(10,60),(200,54),(2000,44),(8000,48),(30000,52),(1e9,56)])
    zcs=_score(feat["zero_crossing_rate"],[(0.02,58),(0.08,50),(0.20,44),(0.40,48),(1,56)])
    return max(40,min(60,
        0.13*sf_score(feat["spectral_flatness"])+0.13*dr_score(feat["dynamic_range"])+
        0.12*sc_score(feat["spectral_centroid"])+0.12*sr_score(feat["spectral_rolloff"])+
        0.10*tfs+0.10*zcs+0.12*nf_score(feat["noise_floor"])+
        0.10*hr_score(feat["harmonic_ratio"])+0.09*br_score(feat["beat_regularity"])+
        0.09*ps_score(feat["pitch_stability"])))

# ── XAI feature scoring ──────────────────────────────────────────────────────
_AW = {"spectral_flatness":0.13,"dynamic_range":0.13,"tonal_variation":0.12,
       "harmonic_movement":0.12,"section_consistency":0.10,"transient_character":0.10,
       "noise_floor":0.12,"harmonic_ratio":0.10,"beat_regularity":0.09,"pitch_stability":0.09}

def _badge(s): return "AI-like" if s>=55 else ("Human-like" if s<=45 else "Neutral")

def _xfeat(id,label,score,value,wkey,what,why,is_primary=False):
    w = NEURAL_WEIGHT if is_primary else ACOUSTIC_WEIGHT*_AW.get(wkey,0.10)
    return {"id":id,"label":label,"score":max(0,min(100,round(score))),"value":value,
            "badge":_badge(score),"weight":w,"weight_label":f"{round(w*100)}% of final score",
            "is_primary":is_primary,"what":what,"why":why}

def score_features_for_xai(feat, neural_prob, final_ai_probability=None):
    ns=max(0,min(100,round(neural_prob*100)))
    nb="AI-like" if ns>=55 else ("Neutral" if ns>=45 else "Human-like")
    nw=(f"Strong AI signal ({ns}%) — primary driver of classification." if ns>=70 else
        f"Leans AI-generated ({ns}%) — main signal." if ns>=55 else
        f"Borderline score ({ns}%) — near the midpoint." if ns>=45 else
        f"Leans human ({ns}%) — but neural net is uncertain.")
    sfv=feat["spectral_flatness"]; sfs=sf_score(sfv)
    sfw=(f"Very low ({sfv:.3f}) — tonal purity suggests synthesis." if sfv<0.2 else
         f"{sfv:.3f} — natural balance of tonal content." if sfv<=0.45 else
         f"{sfv:.3f} — higher side, dense mix, or synthesised textures." if sfv<=0.7 else
         f"High ({sfv:.3f}) — near noise-like distribution.")
    drv=feat["dynamic_range"]; drs=dr_score(drv)
    drw=(f"Extremely low ({drv:.4f}) — loudness barely changes." if drv<0.01 else
         f"{drv:.4f} — heavily compressed." if drv<0.03 else
         f"{drv:.4f} — typical range for professionally produced music." if drv<0.08 else
         f"{drv:.4f} — fairly high, noticeable contrast." if drv<0.15 else
         f"Very high ({drv:.4f}) — dramatic loudness swings.")
    scv=feat["spectral_centroid"]; scs=sc_score(scv)
    scw=(f"Near zero ({scv:.4f}) — near-silent or very bass-heavy." if scv<0.02 else
         f"{scv:.4f} — bass/low-mids dominant." if scv<0.08 else
         f"{scv:.4f} — balanced spread." if scv<0.15 else
         f"{scv:.4f} — moderately bright." if scv<0.30 else
         f"High ({scv:.4f}) — predominantly treble-heavy.")
    srv=feat["spectral_rolloff"]; srs=sr_score(srv)
    srw=(f"Near zero ({srv:.4f}) — almost no energy above bass." if srv<0.01 else
         f"{srv:.4f} — 85% of energy in deep bass." if srv<0.05 else
         f"{srv:.4f} — typical range, natural treble taper." if srv<0.25 else
         f"{srv:.4f} — moderately high, bright mix." if srv<0.50 else
         f"Very high ({srv:.4f}) — dominant high-frequency content.")
    tfv=feat["temporal_flux"]; tfs2=_score(tfv,[(10,60),(200,54),(2000,44),(8000,48),(30000,52),(1e9,56)])
    tfw=(f"Near-zero ({tfv:,.0f}) — essentially static audio." if tfv<10 else
         f"Low ({tfv:,.0f}) — very slow evolution." if tfv<200 else
         f"Healthy range ({tfv:,.0f}) — natural musical activity." if tfv<2000 else
         f"Fairly active ({tfv:,.0f}) — dense or percussive music." if tfv<8000 else
         f"High ({tfv:,.0f}) — very energetic." if tfv<30000 else
         f"Extremely high ({tfv:,.0f}) — approaching noise-like behaviour.")
    zcrv=feat["zero_crossing_rate"]; zcrs=_score(zcrv,[(0.02,58),(0.08,50),(0.20,44),(0.40,48),(1,56)])
    zcrw=(f"Very low ({zcrv:.3f}) — smooth, low-frequency dominated." if zcrv<0.05 else
          f"{zcrv:.3f} — typical range." if zcrv<0.15 else
          f"{zcrv:.3f} — moderately high." if zcrv<0.35 else
          f"High ({zcrv:.3f}) — dominant high-frequency or noise-like.")
    nfv=feat["noise_floor"]; nfs=nf_score(nfv)
    nfw=(f"Near-zero ({nfv:.5f}) — perfectly silent gaps." if nfv<0.0002 else
         f"Very low ({nfv:.5f}) — quieter than most real recordings." if nfv<0.001 else
         f"{nfv:.5f} — natural range, consistent with studio noise." if nfv<0.004 else
         f"Moderately elevated ({nfv:.5f}) — noticeable background noise." if nfv<0.015 else
         f"High ({nfv:.5f}) — substantial background noise.")
    hrv=feat["harmonic_ratio"]; hrs=hr_score(hrv); hrp=round(hrv*100,1)
    hrw=(f"Very high ({hrp}%) — near-perfect harmonic series." if hrv>0.90 else
         f"High ({hrp}%) — most energy in harmonic peaks." if hrv>0.75 else
         f"{hrp}% — natural balance of harmonic peaks and noise." if hrv>0.55 else
         f"{hrp}% — notable noise energy between harmonics." if hrv>0.35 else
         f"Low ({hrp}%) — predominantly noise-like.")
    brv=feat["beat_regularity"]; brs=br_score(brv); brp=round(brv*100,1)
    brw=(f"Very high ({brp}%) — near-perfectly metronomic." if brv>0.90 else
         f"High ({brp}%) — very consistent rhythm." if brv>0.75 else
         f"{brp}% — clear pulse with human timing variations." if brv>0.50 else
         f"Low ({brp}%) — considerable timing variation." if brv>0.25 else
         f"Very low ({brp}%) — no clear repeating pulse.")
    psv=feat["pitch_stability"]; pss=ps_score(psv); psp=round(psv*100,1)
    psw=(f"Very high ({psp}%) — pitch barely moves." if psv>0.92 else
         f"High ({psp}%) — very consistent pitch." if psv>0.80 else
         f"{psp}% — natural range, moderate variation." if psv>0.60 else
         f"Moderate-low ({psp}%) — significant pitch variation." if psv>0.40 else
         f"Low ({psp}%) — dominant pitch varies widely.")
    feats=[
        _xfeat("neural_score","Neural spectrogram score",ns,f"{neural_prob:.3f}","neural",
               "The deep neural network is the primary classifier.",nw,is_primary=True),
        _xfeat("spectral_flatness","Frequency distribution",sfs,f"{sfv:.3f}","spectral_flatness",
               "Whether energy is concentrated at specific musical pitches (tonal) or spread evenly.",sfw),
        _xfeat("dynamic_range","Loudness variation",drs,f"{drv:.4f}","dynamic_range",
               "How much loudness changes moment-to-moment.",drw),
        _xfeat("tonal_variation","Tonal brightness",scs,f"{scv:.4f}","tonal_variation",
               "Where the spectral 'centre of mass' sits.",scw),
        _xfeat("harmonic_movement","High-frequency activity",srs,f"{srv:.4f}","harmonic_movement",
               "The frequency below which 85% of energy sits.",srw),
        _xfeat("section_consistency","Moment-to-moment change",tfs2,f"{tfv:,.0f}","section_consistency",
               "How rapidly frequency content changes frame-to-frame.",tfw),
        _xfeat("transient_character","Attack sharpness",zcrs,f"{zcrv:.3f}","transient_character",
               "How often the waveform crosses zero per second.",zcrw),
        _xfeat("noise_floor","Background noise floor",nfs,f"{nfv:.5f}","noise_floor",
               "The amplitude level during the quietest 10% of the track.",nfw),
        _xfeat("harmonic_ratio","Harmonic purity",hrs,f"{hrp}%","harmonic_ratio",
               "The proportion of energy at distinct harmonic frequencies.",hrw),
        _xfeat("beat_regularity","Rhythmic regularity",brs,f"{brp}%","beat_regularity",
               "How metronomically consistent the rhythm is.",brw),
        _xfeat("pitch_stability","Pitch consistency",pss,f"{psp}%","pitch_stability",
               "How consistently the dominant pitch holds its frequency.",psw),
    ]
    primary=[f for f in feats if f["is_primary"]]
    secondary=sorted([f for f in feats if not f["is_primary"]],key=lambda x:x["score"],reverse=True)
    return primary+secondary

# ══════════════════════════════════════════════════════════════════════════════
# ── Comprehensive Acoustic Deep Analysis ──────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _interp_label(val, breakpoints):
    """Return a label string by linearly scanning sorted (threshold, label) breakpoints."""
    for thr, lbl in breakpoints:
        if val < thr:
            return lbl
    return breakpoints[-1][1]

def _make_signal(score, weight=1.0, direction="ai", note=""):
    """Helper to build a signal entry for AI/Human indicator lists."""
    return {"score": round(score, 1), "weight": round(weight, 2),
            "direction": direction, "note": note}

def compute_acoustic_deep_analysis(feat, genre_result=None, segment_results=None,
                                   neural_prob=None, ai_probability=None, verdict=None):
    """
    Five-category deep acoustic analysis covering:
      1. Spectral characteristics
      2. Temporal & rhythmic features
      3. Dynamic range & loudness variation
      4. Harmonic vs percussive balance
      5. Genre awareness
      6. Artist/style signature proxies
      7. Production quality indicators
      8. AI vs Human signal lists
    Returns a structured dict consumed by the frontend.
    """

    feat = feat or {}

    # ── Raw feature values (with safe defaults) ──────────────────────────────
    sf_val  = float(feat.get("spectral_flatness",    0.0))
    dr_val  = float(feat.get("dynamic_range",        0.0))
    zcr_val = float(feat.get("zero_crossing_rate",   0.0))
    sc_val  = float(feat.get("spectral_centroid",    0.0))
    sr_val  = float(feat.get("spectral_rolloff",     0.0))
    tf_val  = float(feat.get("temporal_flux",        0.0))
    nf_val  = float(feat.get("noise_floor",          0.0))
    hr_val  = float(feat.get("harmonic_ratio",       0.0))
    br_val  = float(feat.get("beat_regularity",      0.0))
    ps_val  = float(feat.get("pitch_stability",      0.0))

    # ── 1. SPECTRAL CHARACTERISTICS ──────────────────────────────────────────
    # Spectral centroid → brightness
    brightness_label = _interp_label(sc_val, [
        (0.04,  "Sub-bass heavy — very dark tone"),
        (0.10,  "Bass/low-mid dominant — warm and full"),
        (0.20,  "Well-balanced — natural spectral spread"),
        (0.35,  "Moderately bright — upper-mid presence"),
        (1.0,   "Treble-heavy — very bright or thin mix"),
    ])
    # Spectral rolloff → energy distribution
    rolloff_label = _interp_label(sr_val, [
        (0.05,  "Energy concentrated in deep bass"),
        (0.15,  "Most energy in lows/mids — natural taper"),
        (0.35,  "Balanced extension into highs"),
        (0.55,  "Strong high-frequency extension"),
        (1.0,   "Dominant very-high-frequency content"),
    ])
    # Spectral flatness → tonal vs noisy
    tonality_label = _interp_label(sf_val, [
        (0.05,  "Highly tonal — strong pitched content"),
        (0.15,  "Mostly tonal — some noise-like texture"),
        (0.35,  "Balanced tonal/noise texture"),
        (0.60,  "Noise-like — broad spectral distribution"),
        (1.0,   "Near-white noise — very diffuse spectrum"),
    ])
    # Bandwidth proxy: rolloff - centroid gap
    bw_proxy = max(0.0, sr_val - sc_val)
    bandwidth_label = _interp_label(bw_proxy, [
        (0.05,  "Narrow bandwidth — limited frequency range"),
        (0.15,  "Moderate bandwidth"),
        (0.30,  "Wide bandwidth — full-range mix"),
        (1.0,   "Very wide bandwidth — extended highs and lows"),
    ])

    spectral = {
        "spectral_centroid":  {"value": round(sc_val, 4), "label": brightness_label,
                               "interpretation": "Spectral brightness — higher values indicate a brighter, more treble-forward mix."},
        "spectral_bandwidth": {"value": round(bw_proxy, 4), "label": bandwidth_label,
                               "interpretation": "Estimated frequency coverage from bass to highs."},
        "spectral_rolloff":   {"value": round(sr_val, 4), "label": rolloff_label,
                               "interpretation": "Point below which 85% of spectral energy falls."},
        "spectral_flatness":  {"value": round(sf_val, 3),  "label": tonality_label,
                               "interpretation": "Tonal vs noise-like quality — low = pitched/tonal, high = noisy/diffuse."},
        "zero_crossing_rate": {"value": round(zcr_val, 3),
                               "label": _interp_label(zcr_val, [
                                   (0.04, "Very smooth — predominantly low-frequency content"),
                                   (0.10, "Typical — balanced mid-frequency activity"),
                                   (0.20, "Elevated — significant high-frequency or transient content"),
                                   (1.0,  "High — dominant high-frequency or noise-like signal"),
                               ]),
                               "interpretation": "Waveform zero-crossing rate — a proxy for high-frequency content and percussive activity."},
    }

    # ── 2. TEMPORAL & RHYTHMIC FEATURES ──────────────────────────────────────
    # Beat regularity
    br_pct = round(br_val * 100, 1)
    rhythm_label = _interp_label(br_val, [
        (0.30, "Highly irregular — free-time or heavily rubato"),
        (0.55, "Loose groove — natural human timing variations"),
        (0.70, "Moderate consistency — slight expressive drift"),
        (0.85, "High regularity — tight, metronomic feel"),
        (1.0,  "Near-perfect grid — machine-like precision"),
    ])
    # Pitch stability → onset / melodic continuity proxy
    ps_pct = round(ps_val * 100, 1)
    onset_label = _interp_label(ps_val, [
        (0.45, "Very dynamic pitch movement — wide melodic range"),
        (0.65, "Expressive — natural pitch drift and vibrato"),
        (0.80, "Moderate stability — controlled melodic movement"),
        (0.92, "High stability — consistent tonal centre"),
        (1.0,  "Near-static — very little pitch change"),
    ])
    # Temporal flux → onset density / musical activity
    flux_label = _interp_label(tf_val, [
        (10,    "Near-static — very slow-evolving or silence"),
        (200,   "Sparse — gentle, minimal musical activity"),
        (1000,  "Moderate — typical melodic/harmonic content"),
        (5000,  "Active — rhythmically dense or percussive"),
        (15000, "Very active — energetic or complex arrangement"),
        (1e9,   "Extremely dense — noise-like temporal activity"),
    ])

    temporal = {
        "beat_regularity":  {"value": br_pct, "unit": "%", "label": rhythm_label,
                             "interpretation": "Autocorrelation-based rhythmic consistency — how metronomically regular the pulse is."},
        "pitch_stability":  {"value": ps_pct, "unit": "%", "label": onset_label,
                             "interpretation": "Dominant pitch steadiness — captures vibrato, portamento and melodic variation."},
        "temporal_flux":    {"value": round(tf_val, 1), "unit": "flux", "label": flux_label,
                             "interpretation": "Frame-to-frame spectral change rate — correlates with rhythmic density and arrangement complexity."},
    }

    # ── 3. DYNAMIC RANGE & LOUDNESS VARIATION ────────────────────────────────
    dr_pct = round(dr_val * 100, 3)
    dynamics_label = _interp_label(dr_val, [
        (0.005, "Extremely compressed — almost zero dynamic contrast"),
        (0.015, "Heavily limited — typical loud-mastered pop/EDM"),
        (0.035, "Moderately compressed — professional level control"),
        (0.060, "Good dynamic range — natural loudness variation"),
        (0.120, "Wide dynamic range — significant soft/loud contrast"),
        (1.0,   "Very wide — dramatic amplitude swings"),
    ])
    # Noise floor → silence behaviour
    nf_label = _interp_label(nf_val, [
        (0.0001, "Perfectly silent gaps — digital silence"),
        (0.0005, "Near-silent — cleaner than most real recordings"),
        (0.001,  "Very low — typical studio floor"),
        (0.003,  "Natural studio noise — consistent with live tracking"),
        (0.010,  "Elevated background — ambient or room sound present"),
        (1.0,    "Substantial noise floor — noisy environment or analogue tape"),
    ])
    # Compression aggressiveness heuristic
    comp_index = round(max(0.0, 1.0 - dr_val / 0.10) * 100, 1)
    comp_label = _interp_label(comp_index, [
        (20,  "Minimal limiting — very open, uncompressed feel"),
        (40,  "Light compression — natural punch retained"),
        (60,  "Moderate compression — typical commercial production"),
        (80,  "Heavy limiting — very loud, modern mastering"),
        (101, "Extreme limiting — brickwall, zero headroom"),
    ])

    dynamics = {
        "dynamic_range":        {"value": round(dr_val, 4), "label": dynamics_label,
                                 "interpretation": "RMS-based loudness variation — low values suggest heavy limiting or mastering compression."},
        "noise_floor":          {"value": round(nf_val, 6), "label": nf_label,
                                 "interpretation": "Amplitude in the quietest 10% of the track — a proxy for recording environment and post-processing."},
        "compression_index":    {"value": comp_index, "unit": "%", "label": comp_label,
                                 "interpretation": "Estimated aggressiveness of dynamic range compression / limiting (higher = more compressed)."},
    }

    # ── 4. HARMONIC vs PERCUSSIVE BALANCE ────────────────────────────────────
    hr_pct = round(hr_val * 100, 1)
    harmonic_label = _interp_label(hr_val, [
        (0.30, "Predominantly percussive / noise-like — minimal harmonic structure"),
        (0.50, "Balanced — roughly equal harmonic and noise energy"),
        (0.65, "Mostly harmonic — some noise between overtones"),
        (0.80, "Strongly harmonic — clear pitch with overtone series"),
        (0.92, "Near-pure harmonic — very clean overtone stack"),
        (1.0,  "Theoretically pure — synthesised sine-like tones"),
    ])
    # Percussive proxy: inverse of harmonic_ratio weighted by zcr
    perc_index = round((1.0 - hr_val) * 60 + zcr_val * 40, 2)
    perc_label = _interp_label(perc_index, [
        (15, "Low percussive energy — melodic / tonal focus"),
        (30, "Moderate percussive presence — balanced rhythm/melody"),
        (50, "Significant percussive content — rhythm-led arrangement"),
        (70, "High percussive density — drums/transients dominant"),
        (100,"Very high — predominantly percussive / noise-like"),
    ])

    harmonic_percussive = {
        "harmonic_ratio":   {"value": hr_pct, "unit": "%", "label": harmonic_label,
                             "interpretation": "Proportion of spectral energy at distinct harmonic frequencies — high = pitched/tonal, low = percussive/noisy."},
        "percussive_index": {"value": round(perc_index, 1), "unit": "index", "label": perc_label,
                             "interpretation": "Estimated percussive character combining harmonic absence and zero-crossing activity."},
    }

    # ── 5. GENRE AWARENESS ────────────────────────────────────────────────────
    genre_analysis = {}
    if genre_result and genre_result.get("top"):
        top_genre  = genre_result["top"]["label"]
        top_score  = genre_result["top"]["score"]
        stability  = genre_result.get("stability", 0)
        conf_note  = genre_result.get("confidence_note", "")

        # Genre-specific feature interpretation norms
        GENRE_NORMS = {
            "Electronic": {
                "beat_regularity_expected": (0.85, 1.0),
                "dynamic_range_expected":   (0.005, 0.025),
                "harmonic_ratio_expected":  (0.55, 0.90),
                "note": "EDM/Electronic typically shows high rhythmic regularity and heavy limiting.",
            },
            "Hip-Hop": {
                "beat_regularity_expected": (0.70, 0.95),
                "dynamic_range_expected":   (0.010, 0.040),
                "harmonic_ratio_expected":  (0.40, 0.70),
                "note": "Hip-Hop commonly features quantised rhythm with moderate dynamics and sampled textures.",
            },
            "Pop": {
                "beat_regularity_expected": (0.75, 0.95),
                "dynamic_range_expected":   (0.008, 0.030),
                "harmonic_ratio_expected":  (0.50, 0.80),
                "note": "Pop production tends toward high regularity and moderate-to-heavy compression.",
            },
            "Classical": {
                "beat_regularity_expected": (0.30, 0.75),
                "dynamic_range_expected":   (0.04,  0.15),
                "harmonic_ratio_expected":  (0.65, 0.95),
                "note": "Classical music typically has wide dynamic range, expressive timing and rich harmonics.",
            },
            "Jazz": {
                "beat_regularity_expected": (0.35, 0.70),
                "dynamic_range_expected":   (0.03,  0.10),
                "harmonic_ratio_expected":  (0.55, 0.85),
                "note": "Jazz is known for expressive timing, complex harmony, and moderate dynamics.",
            },
            "Metal": {
                "beat_regularity_expected": (0.75, 0.97),
                "dynamic_range_expected":   (0.005, 0.025),
                "harmonic_ratio_expected":  (0.40, 0.75),
                "note": "Metal often features extremely precise timing, heavy limiting, and distorted (lower harmonic ratio) timbres.",
            },
            "Rock": {
                "beat_regularity_expected": (0.60, 0.88),
                "dynamic_range_expected":   (0.015, 0.060),
                "harmonic_ratio_expected":  (0.45, 0.75),
                "note": "Rock balances human groove with studio dynamics and guitar-driven harmonic content.",
            },
            "Blues": {
                "beat_regularity_expected": (0.40, 0.72),
                "dynamic_range_expected":   (0.025, 0.080),
                "harmonic_ratio_expected":  (0.50, 0.78),
                "note": "Blues is characterised by expressive timing, moderate dynamics, and vocal/guitar harmonic richness.",
            },
            "Country": {
                "beat_regularity_expected": (0.60, 0.85),
                "dynamic_range_expected":   (0.020, 0.070),
                "harmonic_ratio_expected":  (0.55, 0.82),
                "note": "Country music typically features tight but human rhythm and clean, open production.",
            },
            "R&B": {
                "beat_regularity_expected": (0.65, 0.90),
                "dynamic_range_expected":   (0.010, 0.040),
                "harmonic_ratio_expected":  (0.50, 0.78),
                "note": "R&B commonly uses programmed or tight rhythm with lush, processed harmonic content.",
            },
            "Reggae": {
                "beat_regularity_expected": (0.55, 0.82),
                "dynamic_range_expected":   (0.020, 0.060),
                "harmonic_ratio_expected":  (0.50, 0.76),
                "note": "Reggae features moderate rhythmic regularity with laid-back groove and warm, mid-heavy tone.",
            },
        }

        norms = GENRE_NORMS.get(top_genre, {})
        alignments = []

        if norms:
            def _check_alignment(val, lo, hi, name, human_dir="none"):
                in_range = lo <= val <= hi
                deviation = 0.0
                if val < lo:
                    deviation = round((lo - val) / max(lo, 1e-6) * 100, 1)
                    status = "Below expected range"
                elif val > hi:
                    deviation = round((val - hi) / max(hi, 1e-6) * 100, 1)
                    status = "Above expected range"
                else:
                    status = "Within expected range"
                return {"feature": name, "status": status,
                        "in_range": in_range, "deviation_pct": deviation,
                        "expected": f"{lo}–{hi}", "actual": val}

            br_lo, br_hi = norms.get("beat_regularity_expected", (0.5, 0.9))
            dr_lo, dr_hi = norms.get("dynamic_range_expected",   (0.01, 0.06))
            hr_lo, hr_hi = norms.get("harmonic_ratio_expected",  (0.45, 0.80))

            alignments = [
                _check_alignment(round(br_val, 3), br_lo, br_hi, "Beat Regularity"),
                _check_alignment(round(dr_val, 4), dr_lo, dr_hi, "Dynamic Range"),
                _check_alignment(round(hr_val, 3), hr_lo, hr_hi, "Harmonic Ratio"),
            ]
            in_range_count = sum(1 for a in alignments if a["in_range"])
            alignment_summary = (
                "Strong genre alignment — all key features match genre norms" if in_range_count == 3 else
                "Partial genre alignment — most features align with genre norms" if in_range_count == 2 else
                "Weak genre alignment — several features diverge from genre norms" if in_range_count == 1 else
                "Genre mismatch — key features conflict with genre norms"
            )
        else:
            alignment_summary = "Genre norms not available for this genre"

        genre_analysis = {
            "detected_genre":    top_genre,
            "confidence":        round(top_score, 1),
            "stability":         round(stability, 1),
            "confidence_note":   conf_note,
            "genre_note":        norms.get("note", ""),
            "feature_alignment": alignments,
            "alignment_summary": alignment_summary,
        }

    # ── 6. ARTIST / STYLE SIGNATURE PROXIES ──────────────────────────────────
    # Vocal processing proxy: pitch_stability × spectral_centroid brightness
    vocal_proc_index = round(ps_val * (1.0 - sf_val) * 100, 1)
    vocal_proc_label = _interp_label(vocal_proc_index, [
        (20, "Minimal vocal/melodic processing detected"),
        (40, "Light processing — natural or lightly treated sound"),
        (60, "Moderate processing — standard studio treatment"),
        (80, "Heavy processing — significant pitch/spectral shaping"),
        (101,"Extreme processing — highly synthesised or auto-tuned"),
    ])

    # Arrangement complexity: combo of temporal flux and spectral variation
    arrange_score = round(min(100, tf_val / 200 * 30 + (1.0 - sf_val) * 40 + br_val * 30), 1)
    arrange_label = _interp_label(arrange_score, [
        (20, "Very sparse — minimal arrangement"),
        (40, "Simple arrangement — few simultaneous elements"),
        (60, "Moderate complexity — layered but restrained"),
        (80, "Complex arrangement — dense, multi-layered production"),
        (101,"Very complex — highly intricate or maximalist arrangement"),
    ])

    # Artistic identity consistency across segments
    if segment_results and len(segment_results) >= 2:
        seg_neural = [s["neural_prob"] for s in segment_results]
        seg_acoustic= [s.get("acoustic_prob", 0.5) for s in segment_results]
        identity_cv = round(float(np.std(seg_neural) / (np.mean(seg_neural) + 1e-6)) * 100, 1)
        identity_label = _interp_label(identity_cv, [
            (8,  "Highly consistent identity — uniform character across track"),
            (18, "Consistent — minor variation between sections"),
            (30, "Moderate variation — some contrast between sections"),
            (50, "High variation — markedly different character by section"),
            (100,"Very inconsistent — sections sound like different recordings"),
        ])
        identity_note = (
            "Low variation may indicate repetitive, AI-like structure." if identity_cv < 8
            else "Natural variation suggests intentional artistic contrast." if identity_cv < 30
            else "High variation may indicate genre-blending or inconsistent production."
        )
    else:
        identity_cv = None
        identity_label = "Insufficient segments for identity analysis"
        identity_note  = ""

    style_signature = {
        "vocal_processing_index": {"value": vocal_proc_index, "label": vocal_proc_label,
                                   "interpretation": "Proxy for the degree of pitch/spectral processing applied to melodic content."},
        "arrangement_complexity": {"value": arrange_score, "label": arrange_label,
                                   "interpretation": "Estimated layering and arrangement density based on temporal activity and spectral character."},
        "identity_consistency":   {"value": identity_cv, "cv_pct": identity_cv, "label": identity_label,
                                   "note": identity_note,
                                   "interpretation": "Cross-segment variation in neural score — low CV may indicate generic/repetitive structure."},
    }

    # ── 7. PRODUCTION QUALITY INDICATORS ────────────────────────────────────
    # Stereo width proxy — not directly measurable from mono features, so we use
    # spectral complexity as a surrogate (centroid spread + flatness)
    mix_clarity_index = round((1.0 - sf_val) * 50 + (1.0 - abs(sc_val - 0.20) / 0.30) * 50, 1)
    mix_clarity_index = max(0.0, min(100.0, mix_clarity_index))
    mix_clarity_label = _interp_label(mix_clarity_index, [
        (30, "Poor mix clarity — congested or unbalanced spectrum"),
        (50, "Below average clarity — some muddiness or harshness"),
        (65, "Average clarity — typical commercial production"),
        (80, "Good clarity — well-balanced and separated mix"),
        (101,"Excellent clarity — very clean, open and professional mix"),
    ])

    # Mastering consistency: low temporal flux variance across segments
    if segment_results and len(segment_results) >= 2:
        seg_feat = [s.get("features", {}) for s in segment_results if s.get("features")]
        if seg_feat:
            dr_vals = [f.get("dynamic_range", dr_val) for f in seg_feat]
            master_cv = round(float(np.std(dr_vals) / (np.mean(dr_vals) + 1e-6)) * 100, 1)
        else:
            master_cv = None
    else:
        master_cv = None

    mastering_label = _interp_label(master_cv if master_cv is not None else 50, [
        (10, "Very consistent mastering — uniform loudness across track"),
        (25, "Consistent mastering — slight variation between sections"),
        (45, "Moderate consistency — some loudness swings"),
        (70, "Inconsistent — noticeable level differences by section"),
        (101,"Very inconsistent — mastering quality varies significantly"),
    ]) if master_cv is not None else "Insufficient data for mastering analysis"

    # Artifact detection heuristics
    artifacts = []
    if sf_val > 0.75:
        artifacts.append({"type": "Over-processing / spectral smearing",
                          "severity": "moderate",
                          "evidence": f"High spectral flatness ({sf_val:.3f}) suggests excessive processing or distortion.",
                          "ai_relevant": True})
    if dr_val < 0.005:
        artifacts.append({"type": "Brickwall limiting",
                          "severity": "high",
                          "evidence": f"Extremely low dynamic range ({dr_val:.4f}) indicates aggressive limiting.",
                          "ai_relevant": False})
    if br_val > 0.95:
        artifacts.append({"type": "Over-quantised rhythm",
                          "severity": "moderate",
                          "evidence": f"Near-perfect beat regularity ({br_val:.3f}) may indicate MIDI/grid quantisation.",
                          "ai_relevant": True})
    if nf_val < 0.00005:
        artifacts.append({"type": "Digital silence in gaps",
                          "severity": "low",
                          "evidence": f"Near-zero noise floor ({nf_val:.6f}) — no room tone or analogue noise present.",
                          "ai_relevant": True})
    if ps_val > 0.95:
        artifacts.append({"type": "Pitch over-correction / heavy auto-tune",
                          "severity": "moderate",
                          "evidence": f"Very high pitch stability ({ps_val:.3f}) may indicate pitch quantisation or auto-tune.",
                          "ai_relevant": True})

    production_quality = {
        "mix_clarity_index":     {"value": round(mix_clarity_index, 1), "label": mix_clarity_label,
                                  "interpretation": "Estimated mix clarity from spectral balance and tonal distribution."},
        "mastering_consistency": {"value": master_cv, "label": mastering_label,
                                  "interpretation": "Cross-segment dynamic range variation — low = consistently mastered."},
        "compression_index":     {"value": comp_index, "label": comp_label,
                                  "interpretation": "Estimated degree of dynamic compression / limiting applied."},
        "artifacts_detected":    artifacts,
        "artifact_count":        len(artifacts),
    }

    # ── 8. AI vs HUMAN INDICATOR LISTS ──────────────────────────────────────
    ai_signals    = []
    human_signals = []

    # Beat regularity
    if br_val > 0.93:
        ai_signals.append({"feature": "Rhythmic Uniformity",
                           "value": f"{br_pct}%",
                           "evidence": "Near-perfect metronomic consistency — typical of sequenced or quantised beats.",
                           "strength": "strong"})
    elif br_val < 0.55:
        human_signals.append({"feature": "Natural Timing Variation",
                              "value": f"{br_pct}%",
                              "evidence": "Loose, expressive timing with human micro-variations in the groove.",
                              "strength": "strong"})

    # Pitch stability
    if ps_val > 0.93:
        ai_signals.append({"feature": "Pitch Quantisation",
                           "value": f"{ps_pct}%",
                           "evidence": "Very stable pitch may indicate heavy auto-tune or synthesised tones.",
                           "strength": "moderate"})
    elif ps_val < 0.60:
        human_signals.append({"feature": "Expressive Pitch Movement",
                              "value": f"{ps_pct}%",
                              "evidence": "Natural pitch drift, vibrato, and melodic expression.",
                              "strength": "strong"})

    # Dynamic range
    if dr_val < 0.005:
        ai_signals.append({"feature": "Brickwall Dynamics",
                           "value": f"{dr_val:.4f}",
                           "evidence": "Extremely compressed loudness — no natural amplitude variation.",
                           "strength": "moderate"})
    elif dr_val > 0.06:
        human_signals.append({"feature": "Wide Dynamic Range",
                              "value": f"{dr_val:.4f}",
                              "evidence": "Natural loudness variation consistent with live or minimally compressed recording.",
                              "strength": "strong"})

    # Noise floor
    if nf_val < 0.00005:
        ai_signals.append({"feature": "Digital Silence",
                           "value": f"{nf_val:.6f}",
                           "evidence": "Perfectly silent background — no room tone or analogue noise floor.",
                           "strength": "moderate"})
    elif nf_val > 0.001:
        human_signals.append({"feature": "Natural Background Noise",
                              "value": f"{nf_val:.6f}",
                              "evidence": "Consistent low-level noise floor characteristic of real recording environments.",
                              "strength": "moderate"})

    # Harmonic ratio extremes
    if hr_val > 0.92:
        ai_signals.append({"feature": "Synthetic Harmonic Purity",
                           "value": f"{hr_pct}%",
                           "evidence": "Near-perfect harmonic series — indicative of synthesised tones without natural overtone decay.",
                           "strength": "moderate"})
    elif hr_val < 0.45:
        human_signals.append({"feature": "Organic Harmonic Content",
                              "value": f"{hr_pct}%",
                              "evidence": "Rich noise between harmonics consistent with acoustic instruments or analog recording.",
                              "strength": "moderate"})

    # Spectral flatness (over-processing)
    if sf_val > 0.65:
        ai_signals.append({"feature": "Spectral Smearing",
                           "value": f"{sf_val:.3f}",
                           "evidence": "High spectral flatness may indicate synthesis, heavy processing, or layered noise textures.",
                           "strength": "weak"})
    elif sf_val < 0.08:
        human_signals.append({"feature": "Tonal Focus",
                              "value": f"{sf_val:.3f}",
                              "evidence": "Strong tonal concentration consistent with acoustic instruments or clean synthesis.",
                              "strength": "weak"})

    # Temporal flux uniformity — if segment results available
    if segment_results and len(segment_results) >= 2:
        seg_feat_list = [s.get("features", {}) for s in segment_results if s.get("features")]
        if len(seg_feat_list) >= 2:
            tf_seg = [f.get("temporal_flux", tf_val) for f in seg_feat_list]
            tf_cv  = float(np.std(tf_seg) / (np.mean(tf_seg) + 1e-6))
            if tf_cv < 0.08:
                ai_signals.append({"feature": "Uniform Temporal Activity",
                                   "value": f"CV={tf_cv:.3f}",
                                   "evidence": "Very low variation in spectral change across sections — typical of repetitive AI generation.",
                                   "strength": "moderate"})
            elif tf_cv > 0.40:
                human_signals.append({"feature": "Varied Section Energy",
                                      "value": f"CV={tf_cv:.3f}",
                                      "evidence": "Strong contrast between sections (verse/chorus/bridge) — consistent with human composition.",
                                      "strength": "moderate"})

    # Neural score alignment
    if neural_prob is not None:
        np_pct = round(neural_prob * 100, 1)
        if neural_prob > 0.80:
            ai_signals.append({"feature": "Neural Network Signal",
                               "value": f"{np_pct}%",
                               "evidence": "The spectrogram pattern strongly matches AI-generated training examples.",
                               "strength": "strong"})
        elif neural_prob < 0.30:
            human_signals.append({"feature": "Neural Network Signal",
                                  "value": f"{100-np_pct}%",
                                  "evidence": "The spectrogram pattern strongly resembles human-recorded audio in the training set.",
                                  "strength": "strong"})

    # ── Final summary ─────────────────────────────────────────────────────────
    ai_strength_map = {"strong": 3, "moderate": 2, "weak": 1}
    ai_weight  = sum(ai_strength_map.get(s["strength"], 1)    for s in ai_signals)
    hum_weight = sum(ai_strength_map.get(s["strength"], 1)    for s in human_signals)
    total_w    = ai_weight + hum_weight + 1e-6
    signal_balance = round(ai_weight / total_w * 100, 1)  # 0=fully human, 100=fully AI
    balance_label = _interp_label(signal_balance, [
        (20,  "Strong human signal dominance"),
        (38,  "Mostly human indicators"),
        (48,  "Slight lean toward human"),
        (52,  "Balanced — near equal AI and human signals"),
        (62,  "Slight lean toward AI"),
        (78,  "Mostly AI indicators"),
        (101, "Strong AI signal dominance"),
    ])

    return {
        "spectral":            spectral,
        "temporal":            temporal,
        "dynamics":            dynamics,
        "harmonic_percussive": harmonic_percussive,
        "genre_awareness":     genre_analysis,
        "style_signature":     style_signature,
        "production_quality":  production_quality,
        "ai_indicators":       ai_signals,
        "human_indicators":    human_signals,
        "signal_balance":      {"score": signal_balance, "label": balance_label,
                                "ai_signal_count":    len(ai_signals),
                                "human_signal_count": len(human_signals)},
    }


def derive_verdict(p):
    if p >= 75:
        return "Likely AI-generated", f"Strong AI patterns detected ({p:.1f}%)"
    elif p >= 60:          # Changed from 55 to 60
        return "Likely AI-generated", f"Patterns lean toward AI generation ({p:.1f}%)"
    elif p <= 25:
        return "Likely Human", f"Strong human characteristics detected ({100-p:.1f}% human)"
    elif p <= 50:          # Changed from 45 to 50
        return "Likely Human", f"Patterns lean toward human performance ({100-p:.1f}% human)"
    else:
        return "Not Sure", f"Signal is ambiguous — could be AI or human ({p:.1f}%)"

# ── Load models ───────────────────────────────────────────────────────────────
print(f"[*] Loading AI detection model on {device}...")
ast_backbone=ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model=HybridASTDetector(ast_backbone).to(device)
if not os.path.exists(CHECKPOINT_PATH): raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
ckpt=torch.load(CHECKPOINT_PATH,map_location=device)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
print(f"[✓] AI detector loaded — epoch={ckpt.get('epoch',0)+1}  val_acc={ckpt.get('val_acc',0):.2%}  device={device}")

print(f"[*] Loading primary genre classifier ({GENRE_MODEL_PRIMARY})...")
try:
    genre_pipeline_primary=pipeline("audio-classification",model=GENRE_MODEL_PRIMARY,
                                    device=0 if torch.cuda.is_available() else -1)
    GENRE_PRIMARY_AVAILABLE=True; print("[✓] Primary genre classifier loaded")
except Exception as e:
    print(f"[!] Primary genre classifier failed: {e}"); genre_pipeline_primary=None; GENRE_PRIMARY_AVAILABLE=False

print(f"[*] Loading secondary genre classifier ({GENRE_MODEL_SECONDARY})...")
try:
    genre_pipeline_secondary=pipeline("audio-classification",model=GENRE_MODEL_SECONDARY,
                                      device=0 if torch.cuda.is_available() else -1,trust_remote_code=True)
    GENRE_SECONDARY_AVAILABLE=True; print("[✓] Secondary genre classifier loaded")
except Exception as e:
    print(f"[!] Secondary genre classifier failed: {e}"); genre_pipeline_secondary=None; GENRE_SECONDARY_AVAILABLE=False

GENRE_AVAILABLE=GENRE_PRIMARY_AVAILABLE or GENRE_SECONDARY_AVAILABLE

# ── Preview clips ─────────────────────────────────────────────────────────────
PREVIEW_CLIP_SEC = DURATION_SEC
PREVIEW_DIR       = "preview_clips"

def _make_previews(wav_path, duration_sec, clip_sec=PREVIEW_CLIP_SEC):
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    dur = max(float(duration_sec), 0.1)
    half = clip_sec / 2.0
    segments = []
    intro_dur = min(clip_sec, dur)
    segments.append(("intro", 0.0, intro_dur))
    if dur > clip_sec:
        chorus_start = max(0.0, dur / 2.0 - half)
        if chorus_start + clip_sec > dur:
            chorus_start = max(0.0, dur - clip_sec)
        chorus_dur = min(clip_sec, dur - chorus_start)
        segments.append(("chorus", chorus_start, chorus_dur))
    else:
        remaining = dur - intro_dur
        if remaining > 0:
            segments.append(("chorus", intro_dur, remaining))
    if dur > clip_sec:
        ending_start = max(0.0, dur - clip_sec)
        ending_dur = min(clip_sec, dur - ending_start)
        segments.append(("ending", ending_start, ending_dur))
    base = os.path.splitext(os.path.basename(wav_path))[0]
    merged_filename = f"{base}_merged_preview_{FRAMES_NEEDED}frames.wav"
    merged_path = os.path.join(PREVIEW_DIR, merged_filename)
    if os.path.exists(merged_path):
        total_dur = sum(s[2] for s in segments if s[2] > 0)
        return {
            "merged_url": f"/preview_clips/{merged_filename}",
            "segments": [{"label": label, "start_sec": round(start, 2), "duration_sec": round(dur_seg, 2), "frames": int(dur_seg * SAMPLE_RATE / HOP_LENGTH)}
                         for label, start, dur_seg in segments if dur_seg > 0],
            "total_duration_sec": round(total_dur, 2),
            "total_frames": int(total_dur * SAMPLE_RATE / HOP_LENGTH),
            "frames_per_clip": FRAMES_NEEDED, "cached": True
        }
    try:
        concat_file = os.path.join(PREVIEW_DIR, f"{base}_concat_list.txt")
        segment_files = []
        with open(concat_file, "w") as f:
            for label, start, seg_dur in segments:
                if seg_dur <= 0: continue
                seg_filename = f"{base}__{label}_{int(seg_dur*100)}ms.wav"
                seg_path = os.path.join(PREVIEW_DIR, seg_filename)
                segment_files.append(seg_path)
                if not os.path.exists(seg_path):
                    subprocess.run(["ffmpeg", "-y", "-i", wav_path,
                                    "-ss", str(round(start, 3)), "-t", str(round(seg_dur, 3)),
                                    "-c:a", "pcm_s16le", seg_path],
                                   capture_output=True, check=True,
                                   encoding='utf-8', errors='replace')
                f.write(f"file '{seg_filename}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_file, "-c:a", "pcm_s16le", merged_path],
                       capture_output=True, check=True,
                       encoding='utf-8', errors='replace')
        total_dur = sum(s[2] for s in segments if s[2] > 0)
        return {
            "merged_url": f"/preview_clips/{merged_filename}",
            "segments": [{"label": label, "start_sec": round(start, 2), "duration_sec": round(seg_dur, 2), "frames": int(seg_dur * SAMPLE_RATE / HOP_LENGTH)}
                         for label, start, seg_dur in segments if seg_dur > 0],
            "total_duration_sec": round(total_dur, 2),
            "total_frames": int(total_dur * SAMPLE_RATE / HOP_LENGTH),
            "frames_per_clip": FRAMES_NEEDED, "cached": False
        }
    except subprocess.CalledProcessError as e:
        return {"error": str(e), "segments": []}

@app.route("/preview_clips/<path:filename>")
def serve_preview_clip(filename): return send_from_directory(PREVIEW_DIR, filename)

@app.route("/")
def index(): return send_from_directory(".", "index.html")

@app.route("/converted_wavs/<path:filename>")
def serve_converted_wav(filename): return send_from_directory(CONVERTED_WAV_DIR, filename)

@app.route("/debug/last_analysis")
def debug_last_analysis():
    import glob
    return jsonify({"converted_wav_dir":CONVERTED_WAV_DIR,
                    "wav_files_found":len(glob.glob(os.path.join(CONVERTED_WAV_DIR,"*.wav"))),
                    "genre_available":GENRE_AVAILABLE,
                    "analysis_duration_sec": DURATION_SEC,
                    "analysis_frames": FRAMES_NEEDED,
                    "ai_threshold": AI_THRESHOLD,
                    "human_threshold": HUMAN_THRESHOLD})

# ── /report ───────────────────────────────────────────────────────────────────
@app.route("/report", methods=["POST"])
def report():
    try:
        data = request.get_json(force=True)
        if not data: return jsonify({"error": "No JSON body received."}), 400
        required = {"filename", "original_verdict", "correct_label", "reason"}
        missing = required - data.keys()
        if missing: return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        report_id = str(uuid.uuid4())[:8].upper()
        record = {
            "report_id": report_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "filename": str(data.get("filename", "unknown"))[:300],
            "original_verdict": str(data.get("original_verdict", ""))[:100],
            "ai_probability": float(data.get("ai_probability", 0)),
            "correct_label": str(data.get("correct_label", ""))[:50],
            "reason": str(data.get("reason", ""))[:100],
            "comment": str(data.get("comment", ""))[:500],
            "genre": data.get("genre"),
            "codec_detected": bool(data.get("codec_detected", False)),
        }
        with open(REPORTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return jsonify({"ok": True, "report_id": report_id}), 200
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Natural-language summary generator ──────────────────────────────────────
def generate_nl_explanation(
    verdict, ai_probability, neural_prob, acoustic_prob,
    shap_acoustic, segment_importance, feat_dict, genre=None, human_score=None, adjustment=None
):
    pct = round(ai_probability, 1)
    lines = []

    if verdict == "Likely AI-generated":
        if pct >= 75:
            opener = f"This track scores {pct}% on the AI similarity scale, indicating strong AI-generated characteristics."
        else:
            opener = f"This track scores {pct}% on the AI similarity scale, suggesting AI-generated content."
    elif verdict == "Not Sure":
        opener = f"The model is uncertain (score: {pct}%), suggesting this track sits in an ambiguous zone between AI and human characteristics."
    else:
        if pct <= 25:
            opener = f"With a score of {pct}%, this track shows strong human acoustic characteristics."
        else:
            opener = f"With a score of {pct}%, this track shows predominantly human acoustic characteristics."
    lines.append(opener)

    # Only mention human indicators when they conflict with an AI verdict
    if human_score is not None and human_score > 70 and verdict == "Likely AI-generated":
        lines.append(f"Acoustic indicators ({human_score:.0f}% human-likelihood) suggest some organic qualities, but the neural network's spectrogram analysis overrides this.")
    elif human_score is not None and human_score > 80 and verdict != "Likely AI-generated":
        lines.append(f"Acoustic indicators ({human_score:.0f}% human-likelihood) reinforce the human classification with natural timing, dynamics, and tonal qualities.")

    disagreement = abs(neural_prob - acoustic_prob)
    if disagreement > 0.25:
        lines.append(f"Note: The neural network ({neural_prob:.0%} AI) is the primary signal, while acoustic features ({acoustic_prob:.0%} AI) provide supplementary context.")

    return " ".join(lines)

# ── /classify (FIXED - Neural dominant) ───────────────────────────────────────
@app.route("/classify", methods=["POST"])
def classify():
    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "No files uploaded."}), 400
    results = []
    for f in files:
        filename = f.filename
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            results.append({"filename": filename, "error": f"Unsupported format: {ext}"})
            continue
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)
        try:
            print(f"\n{'='*60}\n[→] {filename}\n{'='*60}")
            wav_path, total_secs = convert_to_16k_wav(tmp_path, original_filename=filename)
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"WAV not created: {wav_path}")
            data, sr = sf.read(wav_path, always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
                sr = SAMPLE_RATE

            # Use weighted segment analysis
            weighted_neural, weighted_acoustic, agreement_score, segment_results, best_segment = analyze_weighted_segments(
                waveform, total_secs, model, mel_transform, device
            )

            # Get features from the best segment
            chunk_feat = best_segment["features"]
            
            # Get genre classification
            genre_result = None
            if GENRE_AVAILABLE:
                genre_result = classify_genre(waveform)
                if genre_result and genre_result.get("top"):
                    print(f"  Genre: {genre_result['top']['label']} ({genre_result['top']['score']:.1f}%)")
            
            # Detect codec compression
            sample_for_codec = waveform[:, :min(SAMPLES_NEEDED, waveform.shape[1])]
            is_compressed, hf_ratio, codec_conf, codec_details = detect_codec_compression(sample_for_codec, SAMPLE_RATE)
            
            # Calculate calibrated probability (neural dominant now)
            ai_probability, total_adjustment, human_reasons = compute_calibrated_probability(
                weighted_neural, weighted_acoustic, chunk_feat, genre_result, agreement_score
            )
            
            # Apply codec penalty (small)
            codec_penalty_info = {}
            if is_compressed:
                ai_probability, codec_penalty_info = apply_codec_penalty(ai_probability, is_compressed, codec_conf)
                print(f"  Codec penalty: applied -> {ai_probability:.1f}%")
            
            # Determine verdict
            verdict, verdict_explanation = derive_verdict(ai_probability)
            
            # Compute human likelihood score (for display only)
            human_likelihood_score = compute_human_likelihood_score(chunk_feat)
            
            # XAI deep
            run_xai_deep = request.form.get("xai_deep", "false").lower() == "true"
            xai_deep = None
            if run_xai_deep:
                best_segment_waveform = None
                if best_segment["name"] == "chorus":
                    start_sample = int(best_segment["start_sec"] * SAMPLE_RATE)
                    end_sample = int(best_segment["end_sec"] * SAMPLE_RATE)
                    best_segment_waveform = waveform[:, start_sample:end_sample]
                else:
                    best_segment_waveform = waveform[:, :SAMPLES_NEEDED]
                
                if best_segment_waveform.shape[1] < SAMPLES_NEEDED:
                    best_segment_waveform = torch.nn.functional.pad(best_segment_waveform, (0, SAMPLES_NEEDED - best_segment_waveform.shape[1]))
                elif best_segment_waveform.shape[1] > SAMPLES_NEEDED:
                    best_segment_waveform = best_segment_waveform[:, :SAMPLES_NEEDED]
                
                best_mel = mel_from_chunk(best_segment_waveform, mel_transform)
                _, xai_deep = compute_xai_deep(best_mel, model, device)

            # Natural-language explanation
            nl_explanation = generate_nl_explanation(
                verdict, ai_probability, weighted_neural, weighted_acoustic,
                None, None, chunk_feat, genre_result, human_likelihood_score, total_adjustment
            )
            print(f"  [nl] {nl_explanation[:120]}…")

            # Feature scores
            xai_feats = score_features_for_xai(chunk_feat, weighted_neural, ai_probability)

            # Comprehensive acoustic deep analysis
            try:
                acoustic_deep = compute_acoustic_deep_analysis(
                    feat=chunk_feat,
                    genre_result=genre_result,
                    segment_results=segment_results,
                    neural_prob=weighted_neural,
                    ai_probability=ai_probability,
                    verdict=verdict,
                )
                print(f"  [acoustic_deep] signals: AI={acoustic_deep['signal_balance']['ai_signal_count']} Human={acoustic_deep['signal_balance']['human_signal_count']}")
            except Exception as e:
                print(f"  [acoustic_deep] Error: {e}")
                acoustic_deep = {"error": str(e)}

            # Genre response
            gp = {"available": GENRE_AVAILABLE, "primary_available": GENRE_PRIMARY_AVAILABLE,
                  "secondary_available": GENRE_SECONDARY_AVAILABLE,
                  **({k: genre_result.get(k) for k in ["top", "all", "top_subgenre", "sources", "window_count", "stability", "confidence_note", "primary_used", "secondary_used"]}
                     if genre_result else {"top": None, "all": None, "top_subgenre": None, "sources": [], "window_count": 0, "stability": 0, "confidence_note": "", "primary_used": False, "secondary_used": False}),
                  "subgenres": genre_result.get("subgenres", {}) if genre_result else {}}

            # ================================================================
            # PRINT CLEAR VERDICT (MODIFIED SECTION)
            # ================================================================
            print(f"\n  📊 Detailed Analysis:")
            print(f"  {'─' * 50}")
            for seg in segment_results:
                corr_mark = "✓ AGREES" if seg["corroborated"] else "✗ CONFLICT"
                print(f"  {seg['name'].upper():8s} | AI: {seg['neural_prob']*100:5.1f}% | Acoustic: {seg['acoustic_prob']*100:5.1f}% | {corr_mark}")

            print(f"\n  🧠 Weighted Result:")
            print(f"     Neural Network (85% weight): {weighted_neural*100:5.1f}% AI")
            print(f"     Acoustic Features (15% weight): {weighted_acoustic*100:5.1f}% AI")
            print(f"     Adjustment: {total_adjustment:+.1f}%")
            print(f"     {'─' * 40}")
            print(f"     FINAL AI SCORE: {ai_probability:5.1f}%")
            
            # Call the clear verdict display function
            print_clear_verdict(verdict, ai_probability, segment_results, weighted_neural, weighted_acoustic, total_adjustment)

            results.append({
                "original_filename": filename,
                "display_filename": os.path.basename(wav_path),
                "original_format": ext.upper().replace(".", ""),
                "analyzed_format": "WAV (16kHz)",
                "conversion_note": f"Converted from {ext.upper()} to 16kHz WAV",
                "converted_wav_path": f"/converted_wavs/{os.path.basename(wav_path)}" if SAVE_CONVERTED_WAV else None,
                "ai_probability": ai_probability,
                "human_probability": round(100 - ai_probability, 1),
                "verdict": verdict,
                "verdict_explanation": verdict_explanation,
                "nl_explanation": nl_explanation,
                "duration_sec": round(total_secs, 2),
                "analysis_duration_sec": DURATION_SEC,
                "analysis_samples": SAMPLES_NEEDED,
                "analysis_frames": FRAMES_NEEDED,
                "analysis_mode": "weighted_segments_v2_majority_vote",
                "segment_analysis": {
                    "segments": segment_results,
                    "best_segment": best_segment["name"],
                    "agreement_score": round(agreement_score, 3),
                    "weighted_neural": round(weighted_neural, 4),
                    "weighted_acoustic": round(weighted_acoustic, 4),
                    "neural_weight": NEURAL_WEIGHT,
                    "acoustic_weight": ACOUSTIC_WEIGHT,
                },
                "xai": xai_feats,
                "xai_deep": xai_deep,
                "acoustic_deep": acoustic_deep,
                "human_indicators": {
                    "score": round(human_likelihood_score, 1),
                    "reasons": human_reasons,
                    "adjustment_applied": round(total_adjustment, 1),
                    "beat_regularity": round(chunk_feat["beat_regularity"], 3),
                    "pitch_stability": round(chunk_feat["pitch_stability"], 3),
                    "dynamic_range": round(chunk_feat["dynamic_range"], 4),
                    "noise_floor": round(chunk_feat["noise_floor"], 5),
                    "harmonic_ratio": round(chunk_feat["harmonic_ratio"], 3),
                },
                "genre": gp,
                "score_breakdown": {
                    "neural_contribution": round(weighted_neural * NEURAL_WEIGHT * 100, 1),
                    "acoustic_contribution": round(weighted_acoustic * ACOUSTIC_WEIGHT * 100, 1),
                    "adjustment": round(total_adjustment, 1),
                    "neural_raw": round(weighted_neural, 4),
                    "acoustic_raw": round(weighted_acoustic, 4),
                    "agreement_score": round(agreement_score, 3),
                    "codec_compression": {
                        "detected": is_compressed,
                        "hf_ratio": round(hf_ratio, 5),
                        "confidence": round(codec_conf, 3),
                        "details": codec_details,
                        **codec_penalty_info,
                    },
                },
                "diagnostics": {
                    "analysis_source": wav_path,
                    "converted_from": ext,
                    "sample_rate": sr,
                },
                "previews": _make_previews(wav_path, total_secs),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"filename": filename, "error": str(e)})
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return jsonify({"results": results})

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"[*] SoundScan — http://localhost:5000")
    print(f"[*] AI threshold: {AI_THRESHOLD}%")
    print(f"[*] Human threshold: {HUMAN_THRESHOLD}%")
    print(f"[*] Weights: Neural={NEURAL_WEIGHT} Acoustic={ACOUSTIC_WEIGHT}")
    print(f"[*] Analysis mode: Neural-dominant weighted segments v2")
    print(f"[*] Segment aggregation: Majority-vote outlier filtering + energy-based chorus")
    print(f"[*] Adjustment cap: ±8% max, agreement-scaled")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)