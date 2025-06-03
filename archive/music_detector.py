#!/usr/bin/env python3
"""
Music Detection Script v0.0.2
- Integrates advanced music detection pipeline:
  * inaSpeechSegmenter for coarse segmentation
  * AST Transformer for high-precision music probability
  * LAION-CLAP for zero-shot tagging
  * Essentia for BPM
  * music2emo for mood
- Keeps all core functions: ffmpeg extraction, CSV export, timecode handling
- Requires: inaSpeechSegmenter, transformers, torch, laion_clap, essentia, music2emo, soundfile, librosa, numpy, pandas
"""

VERSION = "0.0.2"

import os
import sys
import pandas as pd
import essentia.standard as es
import subprocess
import tempfile
import shutil
import re


def detect_music_segments(audio_file, segment_length=1.0, hop_length=0.5):
    """
    Use Essentia's MusicExtractor on the whole file and print features.
    """
    print(f"Loading audio file: {audio_file}")
    features, _ = es.MusicExtractor()(audio_file)
    print("Extracted features:")
    for k in features.descriptorNames():
        print(f"{k}: {features[k]}")
    # For now, return an empty list (no segments)
    return []


def merge_consecutive_segments(segments, max_gap=2.0):
    """
    Merge consecutive music segments that are close together
    
    Args:
        segments (list): List of (start, end, label, confidence) tuples
        max_gap (float): Maximum gap in seconds to merge segments
    
    Returns:
        list: Merged segments
    """
    if not segments:
        return []
    
    merged = []
    current_start, current_end, current_label, current_conf = segments[0]
    
    for start, end, label, conf in segments[1:]:
        if label == current_label and start - current_end <= max_gap:
            # Merge segments
            current_end = end
            current_conf = max(current_conf, conf)  # Take higher confidence
        else:
            # Add current segment and start new one
            merged.append((current_start, current_end, current_label, current_conf))
            current_start, current_end, current_label, current_conf = start, end, label, conf
    
    # Add the last segment
    merged.append((current_start, current_end, current_label, current_conf))
    
    return merged


def parse_timecode(tc_str, fps):
    """
    Parse a timecode string (hh:mm:ss:ff) into total seconds and frames.
    Returns (seconds, frames)
    """
    m = re.match(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})", tc_str)
    if not m:
        raise ValueError(f"Invalid timecode format: {tc_str}")
    h, m_, s, f = map(int, m.groups())
    total_seconds = h * 3600 + m_ * 60 + s
    return total_seconds, f


def seconds_to_timecode(seconds, fps=25, extra_frames=0):
    """
    Convert seconds (+ extra frames) to timecode format hh:mm:ss:ff
    """
    total_frames = int(round(seconds * fps)) + extra_frames
    h = total_frames // (3600 * fps)
    m = (total_frames % (3600 * fps)) // (60 * fps)
    s = (total_frames % (60 * fps)) // fps
    f = total_frames % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def export_to_csv(segments, output_file="music_segments.csv", offset_tc="00:00:00:00", fps=25):
    """
    Export detected segments to CSV file with timecode format and offset
    """
    offset_sec, offset_frames = parse_timecode(offset_tc, fps)
    timecode_segments = []
    for start, end, label, confidence in segments:
        # Offset in frames
        start_total = start + offset_sec + offset_frames / fps
        end_total = end + offset_sec + offset_frames / fps
        length = end - start
        start_tc = seconds_to_timecode(start_total, fps)
        end_tc = seconds_to_timecode(end_total, fps)
        length_tc = seconds_to_timecode(length, fps)
        timecode_segments.append([start_tc, end_tc, length_tc, label, confidence])
    df = pd.DataFrame(timecode_segments, columns=["Start Timecode", "End Timecode", "Length", "Label", "Confidence"])
    df["Confidence"] = df["Confidence"].round(2)
    df.to_csv(output_file, index=False)
    print(f"✅ Results exported to {output_file}")
    print(f"📊 Found {len(segments)} music segments")


def extract_audio_with_ffmpeg(video_path):
    temp_dir = tempfile.mkdtemp(prefix="musicdetector_")
    wav_path = os.path.join(temp_dir, "temp_audio.wav")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", wav_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return wav_path, temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise RuntimeError(f"ffmpeg audio extraction failed: {e}")


def select_file_from_folder(folder_path):
    """
    List files in the folder and prompt the user to select one.
    Returns the selected file's full path, or None if cancelled.
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist.")
        return None
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        print(f"❌ No files found in '{folder_path}'. Please add files to process.")
        return None
    if len(files) == 1:
        print(f"Found 1 file: '{files[0]}'")
        proceed = input("Proceed? (Y/N): ").strip().lower()
        if proceed == 'y':
            return os.path.join(folder_path, files[0])
        else:
            print("Aborted by user.")
            return None
    else:
        print("Files found for processing:")
        for idx, fname in enumerate(files, 1):
            print(f"  {idx}. {fname}")
        while True:
            choice = input(f"Enter the number of the file to process (1-{len(files)}), or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Aborted by user.")
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(files):
                return os.path.join(folder_path, files[int(choice)-1])
            print("Invalid input. Please try again.")


# --- Advanced Music Detection Pipeline (v0.0.2) ---
if __name__ == "__main__" and VERSION == "0.0.2":
    import soundfile as sf, torch, librosa, numpy as np
    from inaSpeechSegmenter import Segmenter
    from transformers import AutoProcessor, AutoModelForAudioClassification
    import laion_clap, essentia.standard as es
    from music2emo import Music2emo

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    seg = Segmenter()
    proc = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE)
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt()
    emo = Music2emo()

    def speech_music_regions(wav_path):
        regions = seg(wav_path)
        return [r for r in regions if r[0] == 'music']

    def ast_is_music(y, sr):
        inputs = proc(y, sampling_rate=sr, return_tensors="pt", padding=True)
        logits = model(**{k:v.to(DEVICE) for k,v in inputs.items()}).logits
        music_idx = model.config.label2id['Music']
        return torch.sigmoid(logits)[0, music_idx].item()

    def clap_tag(chunk, sr=48000):
        if sr != 48000:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=48000)
        emb = clap_model.get_audio_embedding_from_data(chunk.reshape(1,-1))[0]
        tags = ["rock","classical","contains speech","lo-fi","orchestral"]
        t_emb = clap_model.get_text_embedding(tags)
        sims = (emb @ t_emb.T) / (np.linalg.norm(emb)*np.linalg.norm(t_emb,axis=1))
        return dict(zip(tags, sims))

    def bpm_track(y, sr):
        if y.ndim > 1: y = librosa.to_mono(y.T)
        rhythm = es.RhythmExtractor2013(method="multifeature")
        return rhythm(y)[0]

    def mood_predict(file):
        return emo.predict(file)

    # --- Main v0.0.2 logic ---
    folder_path = os.path.join("Files", "for processing")
    input_file = select_file_from_folder(folder_path)
    if not input_file:
        sys.exit(1)

    offset_tc = input("Enter start timecode for offset (hh:mm:ss:ff, default 00:00:00:00): ").strip() or "00:00:00:00"
    fps_str = input("Enter frame rate (fps, default 25): ").strip() or "25"
    try:
        fps = int(fps_str)
    except ValueError:
        print("Invalid frame rate, using 25.")
        fps = 25
    if not re.match(r"\d{2}:\d{2}:\d{2}:\d{2}", offset_tc):
        print("Invalid timecode format, using 00:00:00:00.")
        offset_tc = "00:00:00:00"

    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg'}
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()
    temp_dir = None
    audio_file = input_file
    try:
        if ext in video_exts:
            print(f"🎬 Detected video file ({ext}), extracting audio with ffmpeg...")
            audio_file, temp_dir = extract_audio_with_ffmpeg(input_file)
            print(f"🔊 Audio extracted to: {audio_file}")
        print("🎵 Starting advanced music detection...")
        y, sr = sf.read(audio_file)
        regions = speech_music_regions(audio_file)
        segments = []
        for lbl, start, end in regions:
            chunk = y[int(start*sr):int(end*sr)]
            c_res = {
                "start": start,
                "end": end,
                "music_prob": ast_is_music(chunk, sr),
                "tags": clap_tag(chunk, sr)
            }
            segments.append(c_res)
        # Export to CSV
        csv_segments = []
        for seg in segments:
            start_tc = seconds_to_timecode(seg["start"], fps)
            end_tc = seconds_to_timecode(seg["end"], fps)
            length_tc = seconds_to_timecode(seg["end"]-seg["start"], fps)
            tags_str = ", ".join(f"{k}:{v:.2f}" for k,v in seg["tags"].items())
            csv_segments.append([start_tc, end_tc, length_tc, seg["music_prob"], tags_str])
        df = pd.DataFrame(csv_segments, columns=["Start Timecode", "End Timecode", "Length", "Music Probability", "Tags"])
        df.to_csv("music_segments_v002.csv", index=False)
        print(f"✅ Results exported to music_segments_v002.csv")
        if segments:
            print(f"🎼 Total music time detected: {sum(seg['end']-seg['start'] for seg in segments):.2f} seconds")
            print(f"BPM: {bpm_track(y, sr)}")
            print(f"Mood: {mood_predict(audio_file)}")
        else:
            print("🔇 No music segments detected")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary files in {temp_dir}")


if __name__ == "__main__":
    main() 
