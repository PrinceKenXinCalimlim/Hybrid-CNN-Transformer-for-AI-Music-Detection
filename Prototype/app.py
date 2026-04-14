import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import os
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
DURATION        = 6
CHECKPOINT_PATH = "best_model.pth"
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_EXTS  = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
SAVE_CONVERTED_WAV = True
CONVERTED_WAV_DIR  = "converted_wavs"
REPORTS_FILE       = "misclassification_reports.jsonl"   # ← new

LOGIT_TEMPERATURE = 1.0
CLIP_THRESHOLD    = 0.95
CLIP_RATIO_SOFT   = 0.01
CLIP_RATIO_HARD   = 0.10
CLIP_MIN_WEIGHT   = 0.40
PROB_FLOOR        = 0.01
PROB_CEILING      = 0.99
NEURAL_WEIGHT     = 0.75
ACOUSTIC_WEIGHT   = 0.25
AI_THRESHOLD      = 60
PROFESSIONAL_PRODUCTION_BONUS_MIN = 0.92

# ── Codec compression penalty config ─────────────────────────────────────────
CODEC_HF_RATIO_THRESHOLD    = 0.02
CODEC_NEURAL_PENALTY        = 0.12
CODEC_SCORE_DISCOUNT        = 0.88

GENRE_MODEL_PRIMARY   = "dima806/music_genres_classification"
GENRE_MODEL_SECONDARY = "mtg-upf/discogs-maest-10s-pw-129e"
GENRE_SAMPLE_RATE     = 16000
GENRE_WIN_PRIMARY     = 10
GENRE_WIN_SECONDARY   = 10
GENRE_N_WINDOWS       = 3
GENRE_TOP_SUBGENRES   = 3

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect_professional_production(feat):
    conds = [feat["dynamic_range"]<0.04, feat["spectral_flatness"]>0.8, feat["temporal_flux"]<0.8]
    reasons = [r for r,c in zip(["professional compression","full frequency spectrum","consistent energy"],conds) if c]
    bonus = max(PROFESSIONAL_PRODUCTION_BONUS_MIN, 1.0-len(reasons)*0.02) if len(reasons)>=2 else 1.0
    return bonus, reasons

def clipping_weight(chunk_tensor):
    cr = float(np.mean(np.abs(chunk_tensor.squeeze().numpy()) >= CLIP_THRESHOLD))
    if cr <= CLIP_RATIO_SOFT: return 1.0, cr
    if cr >= CLIP_RATIO_HARD: return CLIP_MIN_WEIGHT, cr
    t = (cr-CLIP_RATIO_SOFT)/(CLIP_RATIO_HARD-CLIP_RATIO_SOFT)
    return 1.0 - t*(1.0-CLIP_MIN_WEIGHT), cr

# ── Codec compression detection ───────────────────────────────────────────────
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


def apply_codec_penalty(neural_prob, acoustic_prob, ai_probability,
                        is_compressed, codec_confidence):
    if not is_compressed:
        return ai_probability, NEURAL_WEIGHT, ACOUSTIC_WEIGHT, {}

    scale         = codec_confidence
    nw_penalty    = CODEC_NEURAL_PENALTY * scale
    score_mult    = 1.0 - (1.0 - CODEC_SCORE_DISCOUNT) * scale

    adj_nw        = max(0.40, NEURAL_WEIGHT  - nw_penalty)
    adj_aw        = min(0.60, ACOUSTIC_WEIGHT + nw_penalty)

    blended       = (adj_nw * neural_prob + adj_aw * acoustic_prob) * 100.0
    adj_score     = round(float(np.clip(blended * score_mult, PROB_FLOOR * 100, PROB_CEILING * 100)), 1)

    penalty_info = {
        "applied":          True,
        "confidence":       round(codec_confidence, 3),
        "neural_weight_adj": round(adj_nw, 3),
        "acoustic_weight_adj": round(adj_aw, 3),
        "score_multiplier": round(score_mult, 3),
        "original_score":   ai_probability,
        "adjusted_score":   adj_score,
        "reason": (
            "Lossy codec compression detected (likely YouTube rip / MP3 / AAC). "
            "Codec artefacts can mimic AI-generation signatures (spectral flatness, "
            "clean noise floor). Neural weight reduced and final score discounted."
        ),
    }
    return adj_score, adj_nw, adj_aw, penalty_info

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
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,n_fft=1024,hop_length=160,n_mels=128)

def convert_to_16k_wav(input_path, original_filename=None):
    base="".join(c for c in os.path.splitext(os.path.basename(original_filename or input_path))[0]
                 if c.isalnum() or c in (" ","-","_")).rstrip()
    os.makedirs(CONVERTED_WAV_DIR,exist_ok=True)
    out=os.path.join(CONVERTED_WAV_DIR,f"{base}_16khz.wav")
    try:
        subprocess.run(["ffmpeg","-i",input_path,"-ar","16000","-ac","1","-c:a","pcm_s16le","-y",out],
                       capture_output=True,text=True,check=True)
        dur=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                            "-of","default=noprint_wrappers=1:nokey=1",out],capture_output=True,text=True)
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

def mel_from_chunk(chunk):
    m=mel_transform(chunk)
    m=torchaudio.functional.amplitude_to_DB(m,10.0,1e-10,0.0,80.0)
    m=(m-m.mean())/(m.std()+1e-9); m=m.squeeze(0).transpose(0,1); T=m.shape[0]
    if T<1024: m=torch.nn.functional.pad(m,(0,0,0,1024-T))
    elif T>1024: m=m[:1024,:]
    return m.unsqueeze(0)

def prepare_full_audio(waveform):
    target=SAMPLE_RATE*DURATION; total=waveform.shape[1]
    return (torch.nn.functional.pad(waveform,(0,target-total)) if total<target else waveform[:,:target]), total/SAMPLE_RATE

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

# ── Scoring ───────────────────────────────────────────────────────────────────
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

# ── XAI ───────────────────────────────────────────────────────────────────────
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
    nb="AI-like" if ns>=60 else ("Neutral" if ns>=40 else "Human-like")
    nw=(f"Strong AI signal ({ns}%) — unnaturally smooth transitions, overly regular harmonics, or absent micro-imperfections matching AI training examples." if ns>=80 else
        f"Leans AI-generated ({ns}%) — some spectral patterns match AI audio but not definitively. Acoustic features below give more context." if ns>=60 else
        f"Borderline score ({ns}%) — near the midpoint where AI vs human is ambiguous. Common with heavily produced or unusual-genre music." if ns>=40 else
        f"Leans human ({ns}%) — spectral patterns consistent with real recordings: natural imperfections, organic harmonics, subtle noise floor.")

    sfv=feat["spectral_flatness"]; sfs=sf_score(sfv)
    sfw=(f"Very low ({sfv:.3f}) — tonal purity suggests synthesis or heavy filtering." if sfv<0.2 else
         f"{sfv:.3f} — natural balance of tonal content and background texture. Typical of real instruments." if sfv<=0.45 else
         f"{sfv:.3f} — higher side, energy spread fairly evenly. Dense mix, heavy reverb, or synthesised textures." if sfv<=0.7 else
         f"High ({sfv:.3f}) — near noise-like distribution. More typical of synthesised or heavily processed audio.")

    drv=feat["dynamic_range"]; drs=dr_score(drv)
    drw=(f"Extremely low ({drv:.4f}) — loudness barely changes. Suggests heavy limiting or synthesis." if drv<0.01 else
         f"{drv:.4f} — heavily compressed, little variation between sections." if drv<0.03 else
         f"{drv:.4f} — typical range for professionally produced music." if drv<0.08 else
         f"{drv:.4f} — fairly high, noticeable contrast between quiet and loud passages." if drv<0.15 else
         f"Very high ({drv:.4f}) — dramatic loudness swings, consistent with live or unmastered material.")

    scv=feat["spectral_centroid"]; scs=sc_score(scv)
    scw=(f"Near zero ({scv:.4f}) — near-silent, very bass-heavy, or corrupted signal." if scv<0.02 else
         f"{scv:.4f} — dominated by bass/low-mids with little treble." if scv<0.08 else
         f"{scv:.4f} — balanced spread from bass through treble, typical of real arrangements." if scv<0.15 else
         f"{scv:.4f} — moderately bright with notable treble content (cymbals, presence range)." if scv<0.30 else
         f"High ({scv:.4f}) — predominantly treble-heavy, may indicate noise or distortion.")

    srv=feat["spectral_rolloff"]; srs=sr_score(srv)
    srw=(f"Near zero ({srv:.4f}) — almost no energy above the bass range." if srv<0.01 else
         f"{srv:.4f} — 85% of energy in the deep bass. Heavy low-pass filtering." if srv<0.05 else
         f"{srv:.4f} — typical range, most energy in bass-to-mid zone with natural treble taper." if srv<0.25 else
         f"{srv:.4f} — moderately high, bright or treble-forward mix." if srv<0.50 else
         f"Very high ({srv:.4f}) — dominant high-frequency content, may indicate noise or synthesis.")

    tfv=feat["temporal_flux"]; tfs2=_score(tfv,[(10,60),(200,54),(2000,44),(8000,48),(30000,52),(1e9,56)])
    tfw=(f"Near-zero ({tfv:,.0f}) — essentially static audio with no musical movement." if tfv<10 else
         f"Low ({tfv:,.0f}) — very slow evolution, more typical of synthesised pads or drones." if tfv<200 else
         f"Healthy range ({tfv:,.0f}) — natural musical activity consistent with a real performance." if tfv<2000 else
         f"Fairly active ({tfv:,.0f}) — dense or percussive music with many transient events." if tfv<8000 else
         f"High ({tfv:,.0f}) — very energetic or heavily compressed signal." if tfv<30000 else
         f"Extremely high ({tfv:,.0f}) — approaching noise-like behaviour.")

    zcrv=feat["zero_crossing_rate"]; zcrs=_score(zcrv,[(0.02,58),(0.08,50),(0.20,44),(0.40,48),(1,56)])
    zcrw=(f"Very low ({zcrv:.3f}) — smooth, low-frequency dominated signal." if zcrv<0.05 else
          f"{zcrv:.3f} — typical range, natural balance of frequencies with organic transients." if zcrv<0.15 else
          f"{zcrv:.3f} — moderately high, significant high-frequency or percussive content." if zcrv<0.35 else
          f"High ({zcrv:.3f}) — very frequent crossings, dominant high-frequency or noise-like signal.")

    nfv=feat["noise_floor"]; nfs=nf_score(nfv)
    nfw=(f"Near-zero ({nfv:.5f}) — perfectly silent gaps between notes. Real recordings always carry room/mic/tape noise; an absent floor strongly suggests synthesis." if nfv<0.0002 else
         f"Very low ({nfv:.5f}) — quieter than most real recordings. Well-treated studio, aggressive noise reduction, or synthesised source." if nfv<0.001 else
         f"{nfv:.5f} — natural range for a real recording, consistent with studio mic self-noise or light tape saturation." if nfv<0.004 else
         f"Moderately elevated ({nfv:.5f}) — noticeable background noise from a live environment or older equipment." if nfv<0.015 else
         f"High ({nfv:.5f}) — substantial background noise. Typical of field recordings, live performances, or lo-fi material.")

    hrv=feat["harmonic_ratio"]; hrs=hr_score(hrv); hrp=round(hrv*100,1)
    hrw=(f"Very high ({hrp}%) — near-perfect harmonic series with almost no inter-harmonic noise. Strongly characteristic of synthesised instruments or AI-generated music." if hrv>0.90 else
         f"High ({hrp}%) — most energy in harmonic peaks with limited noise between them. Clean studio or largely synthesised instrumentation." if hrv>0.75 else
         f"{hrp}% — natural balance of harmonic peaks and inter-harmonic noise from resonance, breath, and room reflections." if hrv>0.55 else
         f"{hrp}% — notable noise energy between harmonics, typical of distorted guitars, drums, or percussive material." if hrv>0.35 else
         f"Low ({hrp}%) — predominantly noise-like with weak harmonic structure. Heavy distortion, percussion, or atonal content.")

    brv=feat["beat_regularity"]; brs=br_score(brv); brp=round(brv*100,1)
    brw=(f"Very high ({brp}%) — near-perfectly metronomic. Characteristic of grid-quantised AI or DAW music; human performers always drift slightly." if brv>0.90 else
         f"High ({brp}%) — very consistent rhythm with minor fluctuations. Tight drummer, quantised performance, or drum-machine production." if brv>0.75 else
         f"{brp}% — clear pulse with detectable human timing variations. Typical of live drumming or performances without heavy quantisation." if brv>0.50 else
         f"Low ({brp}%) — considerable timing variation. Free-time playing, rubato, or music without a steady pulse." if brv>0.25 else
         f"Very low ({brp}%) — no clear repeating rhythmic pulse. Free-tempo, ambient, or heavily textural material.")

    psv=feat["pitch_stability"]; pss=ps_score(psv); psp=round(psv*100,1)
    psw=(f"Very high ({psp}%) — pitch barely moves. Synthesised audio locks to exact frequencies with no vibrato or drift; real performers always vary." if psv>0.92 else
         f"High ({psp}%) — very consistent pitch. Well-tuned studio, pitch-corrected vocals, synthesised melody, or a stable instrument like piano or organ." if psv>0.80 else
         f"{psp}% — natural range, moderate variation consistent with vibrato, vocal intonation, or melodic movement." if psv>0.60 else
         f"Moderate-low ({psp}%) — significant pitch variation. Expressive playing (bends, jazz phrasing) or wide melodic leaps." if psv>0.40 else
         f"Low ({psp}%) — dominant pitch varies widely. Percussion-heavy, atonal, or rapid harmonic movement.")

    feats=[
        _xfeat("neural_score","Neural spectrogram score",ns,f"{neural_prob:.3f}","neural",
               "The deep neural network analyses the full mel-spectrogram and returns a probability the audio was AI-generated. Trained on thousands of real and AI-produced tracks.",nw,is_primary=True),
        _xfeat("spectral_flatness","Frequency distribution",sfs,f"{sfv:.3f}","spectral_flatness",
               "Whether energy is concentrated at specific musical pitches (tonal) or spread evenly like noise (flat). Real instruments produce strong harmonics at predictable frequencies.",sfw),
        _xfeat("dynamic_range","Loudness variation",drs,f"{drv:.4f}","dynamic_range",
               "How much loudness changes moment-to-moment. Human performances breathe naturally — verses quieter, choruses louder. AI audio can lack this organic variation.",drw),
        _xfeat("tonal_variation","Tonal brightness",scs,f"{scv:.4f}","tonal_variation",
               "Where the spectral 'centre of mass' sits — bass, mids, or treble. Real music has a predictable brightness range by genre; outliers suggest synthesis or artefacts.",scw),
        _xfeat("harmonic_movement","High-frequency activity",srs,f"{srv:.4f}","harmonic_movement",
               "The frequency below which 85% of energy sits. Low rolloff = bass-heavy; high = treble-prominent. Abnormally low values suggest filtering or synthesis.",srw),
        _xfeat("section_consistency","Moment-to-moment change",tfs2,f"{tfv:,.0f}","section_consistency",
               "How rapidly frequency content changes frame-to-frame. Very low = static/synthesised; moderate = real performance; extreme = noise or heavy processing.",tfw),
        _xfeat("transient_character","Attack sharpness",zcrs,f"{zcrv:.3f}","transient_character",
               "How often the waveform crosses zero per second — proxy for high-frequency and percussive content. Real music falls in a predictable range by genre.",zcrw),
        _xfeat("noise_floor","Background noise floor",nfs,f"{nfv:.5f}","noise_floor",
               "The amplitude level during the quietest 10% of the track. Real recordings always carry residual noise from microphones, rooms, and electronics. AI audio often has a perfectly silent floor.",nfw),
        _xfeat("harmonic_ratio","Harmonic purity",hrs,f"{hrp}%","harmonic_ratio",
               "The proportion of energy at distinct harmonic frequencies vs. noise between them. Synthesised sources produce near-perfect harmonic series; real instruments add inter-harmonic noise from physical resonance.",hrw),
        _xfeat("beat_regularity","Rhythmic regularity",brs,f"{brp}%","beat_regularity",
               "How metronomically consistent the rhythm is. AI and DAW music is quantised to a perfect grid. Human performers always introduce micro-timing variations that give music its feel.",brw),
        _xfeat("pitch_stability","Pitch consistency",pss,f"{psp}%","pitch_stability",
               "How consistently the dominant pitch holds its frequency. AI and synthesised instruments are perfectly stable; humans add vibrato, intonation drift, and portamento.",psw),
    ]
    primary=[f for f in feats if f["is_primary"]]
    secondary=sorted([f for f in feats if not f["is_primary"]],key=lambda x:x["score"],reverse=True)
    return primary+secondary

# ── Verdict ───────────────────────────────────────────────────────────────────
def derive_verdict(p):
    if p>=AI_THRESHOLD: return "Likely AI-generated", f"Patterns lean toward AI generation ({p}%)"
    if p>=50:           return "Not Sure",             f"Signal is ambiguous — could be AI or human ({p}%)"
    return "Likely Human", f"Patterns lean toward human performance ({100-p}%)"

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
    GENRE_SECONDARY_AVAILABLE=True; print("[✓] Secondary genre classifier loaded (MAEST 10s, 400 Discogs styles)")
except Exception as e:
    print(f"[!] Secondary genre classifier failed: {e}"); genre_pipeline_secondary=None; GENRE_SECONDARY_AVAILABLE=False

GENRE_AVAILABLE=GENRE_PRIMARY_AVAILABLE or GENRE_SECONDARY_AVAILABLE

# ── Routes ────────────────────────────────────────────────────────────────────
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
                    "genre_primary":GENRE_MODEL_PRIMARY if GENRE_PRIMARY_AVAILABLE else None,
                    "genre_secondary":GENRE_MODEL_SECONDARY if GENRE_SECONDARY_AVAILABLE else None,
                    "genre_windows_per_model":GENRE_N_WINDOWS})

# ── /report — misclassification feedback ─────────────────────────────────────
@app.route("/report", methods=["POST"])
def report():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received."}), 400

        required = {"filename", "original_verdict", "correct_label", "reason"}
        missing  = required - data.keys()
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        report_id = str(uuid.uuid4())[:8].upper()
        record = {
            "report_id":        report_id,
            "timestamp":        datetime.datetime.utcnow().isoformat() + "Z",
            "filename":         str(data.get("filename",        "unknown"))[:300],
            "original_verdict": str(data.get("original_verdict",""))[:100],
            "ai_probability":   float(data.get("ai_probability", 0)),
            "correct_label":    str(data.get("correct_label",   ""))[:50],
            "reason":           str(data.get("reason",          ""))[:100],
            "comment":          str(data.get("comment",         ""))[:500],
            "genre":            data.get("genre"),
            "codec_detected":   bool(data.get("codec_detected", False)),
        }

        with open(REPORTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        print(f"  [report] {report_id} — {record['filename']} | "
              f"verdict={record['original_verdict']} → {record['correct_label']} | "
              f"reason={record['reason']}")

        return jsonify({"ok": True, "report_id": report_id}), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── /classify ─────────────────────────────────────────────────────────────────
@app.route("/classify", methods=["POST"])
def classify():
    files=request.files.getlist("files")
    if not files or files[0].filename=="":
        return jsonify({"error":"No files uploaded."}),400
    results=[]
    for f in files:
        filename=f.filename; ext=os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            results.append({"filename":filename,"error":f"Unsupported format: {ext}"}); continue
        with tempfile.NamedTemporaryFile(suffix=ext,delete=False) as tmp:
            tmp_path=tmp.name; f.save(tmp_path)
        try:
            print(f"\n{'='*60}\n[→] {filename}\n{'='*60}")
            wav_path,total_secs=convert_to_16k_wav(tmp_path,original_filename=filename)
            if not os.path.exists(wav_path): raise FileNotFoundError(f"WAV not created: {wav_path}")
            data,sr=sf.read(wav_path,always_2d=True)
            waveform=torch.from_numpy(data.T).float()
            if sr!=SAMPLE_RATE: waveform=torchaudio.transforms.Resample(sr,SAMPLE_RATE)(waveform); sr=SAMPLE_RATE
            chunk,_=prepare_full_audio(waveform)
            if chunk.shape[1]<16000: raise ValueError("Audio too short (< 1 second).")

            is_compressed, hf_ratio, codec_conf, codec_details = detect_codec_compression(chunk, SAMPLE_RATE)
            if is_compressed:
                print(f"  [codec] Compression detected — HF ratio={hf_ratio:.5f}  confidence={codec_conf:.2f}  details={codec_details}")

            with torch.no_grad():
                raw_logit=model(mel_from_chunk(chunk).to(device)).squeeze().item()
                _,clip_ratio=clipping_weight(chunk)
                chunk_feat=compute_chunk_features(chunk)

            neural_prob=torch.sigmoid(torch.tensor(raw_logit/LOGIT_TEMPERATURE)).item()
            acoustic_prob=acoustic_composite_score(chunk_feat)/100.0
            pro_bonus,pro_reasons=detect_professional_production(chunk_feat)

            if clip_ratio>0.01: neural_prob*=1.0-(min(clip_ratio,0.10)*0.5)

            disagreement=abs(neural_prob-acoustic_prob)
            penalty=min(disagreement/0.5,1.0)
            eff_nw=NEURAL_WEIGHT-penalty*0.08; eff_aw=ACOUSTIC_WEIGHT+penalty*0.08
            ai_probability=round(max(PROB_FLOOR,min(PROB_CEILING,(eff_nw*neural_prob+eff_aw*acoustic_prob)*pro_bonus))*100,1)

            ai_probability, eff_nw, eff_aw, codec_penalty_info = apply_codec_penalty(
                neural_prob, acoustic_prob, ai_probability, is_compressed, codec_conf
            )

            eff_threshold=AI_THRESHOLD+min(disagreement/0.5,1.0)*4
            verdict,verdict_explanation=derive_verdict(ai_probability)
            if disagreement>=0.4 and AI_THRESHOLD<=ai_probability<eff_threshold:
                verdict="Not Sure"; verdict_explanation=f"Models disagree significantly ({ai_probability}%)"

            print(f"  Neural={neural_prob:.1%}  Acoustic={acoustic_prob:.1%}  Score={ai_probability}%  Verdict={verdict}"
                  + (f"  [codec-adj]" if is_compressed else ""))

            genre_result=classify_genre(waveform) if GENRE_AVAILABLE else None
            if genre_result and genre_result.get("top"):
                ts=genre_result.get("top_subgenre")
                sub_str = f" / {ts['label']}" if ts else ""
                print(f"  [genre] {genre_result['top']['label']} ({genre_result['top']['score']}%){sub_str} — {genre_result.get('confidence_note','')}")

            xai_feats=score_features_for_xai(chunk_feat,neural_prob,ai_probability)
            gp={"available":GENRE_AVAILABLE,"primary_available":GENRE_PRIMARY_AVAILABLE,
                "secondary_available":GENRE_SECONDARY_AVAILABLE,
                **({k:genre_result.get(k) for k in ["top","all","top_subgenre","sources","window_count","stability","confidence_note","primary_used","secondary_used"]}
                   if genre_result else {"top":None,"all":None,"top_subgenre":None,"sources":[],"window_count":0,"stability":0,"confidence_note":"","primary_used":False,"secondary_used":False}),
                "subgenres":genre_result.get("subgenres",{}) if genre_result else {}}

            results.append({
                "original_filename":filename,"display_filename":os.path.basename(wav_path),
                "original_format":ext.upper().replace(".",""),"analyzed_format":"WAV (16kHz)",
                "conversion_note":f"Converted from {ext.upper()} to 16kHz WAV",
                "converted_wav_path":f"/converted_wavs/{os.path.basename(wav_path)}" if SAVE_CONVERTED_WAV else None,
                "ai_probability":ai_probability,"verdict":verdict,"verdict_explanation":verdict_explanation,
                "duration_sec":round(total_secs,2),"xai":xai_feats,"genre":gp,
                "score_breakdown":{
                    "neural_contribution":round(eff_nw*neural_prob*100,1),
                    "acoustic_contribution":round(eff_aw*acoustic_prob*100,1),
                    "neural_weight_pct":round(eff_nw*100),"acoustic_weight_pct":round(eff_aw*100),
                    "disagreement":round(disagreement,3),"professional_bonus":round(pro_bonus,3),
                    "professional_reasons":pro_reasons,
                    "codec_compression": {
                        "detected":       is_compressed,
                        "hf_ratio":       round(hf_ratio, 5),
                        "confidence":     round(codec_conf, 3),
                        "details":        codec_details,
                        **codec_penalty_info,
                    },
                },
                "diagnostics":{"analysis_source":wav_path,"converted_from":ext,"sample_rate":sr,
                               "neural_prob":round(neural_prob,4),"acoustic_prob":round(acoustic_prob,4)},
            })
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"filename":filename,"error":str(e)})
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
    return jsonify({"results":results})

if __name__=="__main__":
    print(f"\n{'='*60}\n[*] SoundScan — http://localhost:5000")
    print(f"[*] AI threshold: {AI_THRESHOLD}%  Neural/Acoustic: {round(NEURAL_WEIGHT*100)}/{round(ACOUSTIC_WEIGHT*100)}")
    print(f"[*] Codec penalty: HF threshold={CODEC_HF_RATIO_THRESHOLD}  neural_penalty={CODEC_NEURAL_PENALTY}  score_discount={CODEC_SCORE_DISCOUNT}")
    print(f"[*] Genre: primary={'ON' if GENRE_PRIMARY_AVAILABLE else 'OFF'} secondary={'ON' if GENRE_SECONDARY_AVAILABLE else 'OFF'}  windows={GENRE_N_WINDOWS}")
    print(f"[*] Reports saved to: {REPORTS_FILE}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0",port=5000,debug=False)