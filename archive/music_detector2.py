#!/usr/bin/env python3
# music_detector2.py  ─ MVP spotting / mood pipeline

VERSION = "0.0.4"

# ── music_detector2.py  (add these very first) ──────────────────────────────
import os, faulthandler
faulthandler.enable(all_threads=True)          # automatic C-level traceback

# Prevent TensorFlow/PyTorch OpenMP clashes → no seg-faults
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # disable MKL oneDNN
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE") # allow duplicate OMP
os.environ.setdefault("OMP_NUM_THREADS",      "1")    # single-thread OpenMP

import torch                                    # import torch **before** tf
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# ────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
import argparse, json, logging, shutil, subprocess, tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from inaSpeechSegmenter import Segmenter
from transformers import AutoModelForAudioClassification, AutoProcessor
import laion_clap
import essentia.standard as es

from music2emo.music2emo import Music2emo

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── load heavy models once ──────────────────────────────────────────────────
seg        = Segmenter()
ast_proc   = AutoProcessor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
ast_model  = AutoModelForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
).to(DEVICE)

clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()

emo_model  = Music2emo()
# ─────────────────────────────────────────────────────────────────────────────


# ─── helpers ─────────────────────────────────────────────────────────────────
def extract_audio(video: str) -> Path:
    """Return 44.1 kHz mono WAV path extracted from *video* with ffmpeg."""
    tmp_dir  = Path(tempfile.mkdtemp(prefix="dcs_audio_"))
    wav_path = tmp_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vn",
         "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(wav_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return wav_path


def ina_regions(wav: Path):
    """
    Call inaSpeechSegmenter.  If it crashes with NumPy's "arrays to stack"
    `TypeError`, fall back to treating the whole clip as one music region.
    """
    try:
        return [r for r in seg(str(wav)) if r[0] == "music"]
    except TypeError:                        # empty-energy edge-case
        dur = sf.info(str(wav)).duration
        logging.warning(
            "inaSpeechSegmenter failed (empty energy); "
            "using full-length region instead."
        )
        return [("music", 0.0, dur)]


def ast_music_prob(chunk, sr):
    # ensure 16 kHz mono for AST
    if sr != 16_000:
        chunk = librosa.resample(
            y=chunk,
            orig_sr=sr,
            target_sr=16_000,   # keyword args avoid TypeError
        )
        sr = 16_000

    inputs = ast_proc(
        chunk,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = ast_model(**inputs).logits
    idx    = ast_model.config.label2id["Music"]
    return torch.sigmoid(logits)[0, idx].item()




def clap_tags(chunk, sr):
    """Return LAION-CLAP similarity scores for a fixed text tag list."""
    if sr != 48_000:                            # <-- target SR for CLAP
        chunk = librosa.resample(
            y=chunk,
            orig_sr=sr,
            target_sr=48_000,                   # use kw-args → no TypeError
        )
        sr = 48_000

    emb  = clap_model.get_audio_embedding_from_data(chunk.reshape(1, -1))[0]
    tags = ["rock", "classical", "contains speech", "lo-fi", "orchestral"]
    temb = clap_model.get_text_embedding(tags)
    sims = (emb @ temb.T) / (np.linalg.norm(emb) * np.linalg.norm(temb, axis=1))
    return dict(zip(tags, sims))



def bpm_track(y, sr):
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    return es.RhythmExtractor2013(method="multifeature")(y)[0]


def seconds_to_tc(sec, fps=25):
    total_frames = int(round(sec * fps))
    h = total_frames // (3600 * fps)
    m = (total_frames % (3600 * fps)) // (60 * fps)
    s = (total_frames % (60 * fps)) // fps
    f = total_frames % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="video or audio file")
    ap.add_argument("--out", default="demo_out", help="output folder")
    ap.add_argument("--fps", type=int, default=25, help="time-code fps")
    ap.add_argument("--thr", type=float, default=0.5,
                    help="mood-tag probability threshold")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    wav      = extract_audio(args.src)
    y, sr    = sf.read(str(wav))
    regions  = ina_regions(wav)
    if not regions:
        logging.warning("No music detected.")
        return

    rows = []
    for _, start, end in regions:
        chunk = y[int(start*sr): int(end*sr)]
        rows.append({
            "start": start,
            "end":   end,
            "prob":  ast_music_prob(chunk, sr),
            "tags":  clap_tags(chunk, sr),
        })

    # ─── export markers CSV ────────────────────────────────────────────────
    df = pd.DataFrame([
        [seconds_to_tc(r["start"], args.fps),
         seconds_to_tc(r["end"],   args.fps),
         seconds_to_tc(r["end"]-r["start"], args.fps),
         f"{r['prob']:.2f}",
         ", ".join(f"{k}:{v:.2f}" for k,v in r["tags"].items())]
        for r in rows
    ], columns=["Start", "End", "Length", "MusicProb", "Tags"])
    df.to_csv(out_dir / "markers.csv", index=False)
    logging.info("Markers saved → %s", out_dir / "markers.csv")

    # ─── global BPM & mood ────────────────────────────────────────────────
    bpm   = bpm_track(y, sr)
    moods = emo_model.predict(str(wav), threshold=args.thr)
    with open(out_dir / "mood.json", "w") as f:
        json.dump({"bpm": bpm, **moods}, f, indent=2)
    logging.info("BPM & mood saved → %s", out_dir / "mood.json")

    shutil.rmtree(wav.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
