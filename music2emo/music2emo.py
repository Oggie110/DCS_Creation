# music2emo/music2emo.py
# ────────────────────────────────────────────────────────────────────────────
import os, json, shutil, logging, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch, torchaudio
import torchaudio.transforms as T
import mir_eval, pretty_midi as pm
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from music21 import converter
from tqdm import tqdm

from .utils import logger
from .utils.btc_model         import BTC_model
from .utils.hparams           import HParams
from .utils.mir_eval_modules  import audio_file_to_features, idx2voca_chord
from .utils.mert              import FeatureExtractorMERT
from .model.linear_mt_attn_ck import FeedforwardModelMTAttnCK

# ─── housekeeping ──────────────────────────────────────────────────────────
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

PITCH_CLASS      = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
pitch_num_dic    = {p:i for i,p in enumerate(PITCH_CLASS)}
minor_major_dic2 = {'Db':'C#','Eb':'D#','Gb':'F#','Ab':'G#','Bb':'A#'}

shift_major_dic  = pitch_num_dic
shift_minor_dic  = {
    'A':0,'A#':1,'B':2,'C':3,'C#':4,'D':5,'D#':6,'E':7,'F':8,'F#':9,'G':10,'G#':11
}

segment_duration = 30        # seconds
resample_rate    = 24_000
is_split         = True
# ────────────────────────────────────────────────────────────────────────────
def sanitize_key_signature(key:str)->str:
    return key.replace('-', 'b')

def normalize_chord(lab_path:str, key:str, key_type:str='major')->List[str]:
    if key == "None":
        shift = 0
    else:
        key   = key.title()
        key   = minor_major_dic2.get(key, key)
        shift = shift_major_dic[key] if key_type=="major" else shift_minor_dic[key]

    out=[]
    for line in Path(lab_path).read_text().splitlines():
        if not line.strip(): continue
        s,e,ch=line.split()
        if ch in {"N","X"}:
            nx = ch
        elif ":" in ch:
            root,attr=ch.split(":")
            nx=f"{PITCH_CLASS[(pitch_num_dic[root]-shift)%12]}:{attr}"
        else:
            nx=PITCH_CLASS[(pitch_num_dic[ch]-shift)%12]
        out.append(f"{s} {e} {nx}\n")
    return out

def resample_waveform(wav:torch.Tensor, sr:int, target:int)->Tuple[torch.Tensor,int]:
    if sr==target: return wav, sr
    return T.Resample(sr,target)(wav), target

def split_audio(wav:torch.Tensor, sr:int)->List[torch.Tensor]:
    seg_len=segment_duration*sr
    if wav.shape[-1]<=seg_len: return [wav]
    return [wav[...,i:i+seg_len] for i in range(0,wav.shape[-1],seg_len)]
# ────────────────────────────────────────────────────────────────────────────
class Music2emo:
    def __init__(self, model_weights:str="saved_models/J_all.ckpt"):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        root = Path(__file__).resolve().parent
        self.ckpt_mood = (root/model_weights).resolve()
        self.ckpt_btc  = (root/"inference/data/btc_model_large_voca.pt").resolve()
        self.hparams   = (root/"inference/data/run_config.yaml").resolve()
        for p in (self.ckpt_mood,self.ckpt_btc,self.hparams):
            if not p.exists(): raise FileNotFoundError(p)

        self.feat_ext = FeatureExtractorMERT("m-a-p/MERT-v1-95M",
                                             device=self.device, sr=resample_rate)

        # mood / val-aro model
        self.mood_model = FeedforwardModelMTAttnCK(1536,56,2)
        ckpt=torch.load(self.ckpt_mood,map_location="cpu")["state_dict"]
        self.mood_model.load_state_dict(
            {k.replace("model.",""):v for k,v in ckpt.items()
             if k.replace("model.","") in self.mood_model.state_dict()}
        )
        self.mood_model.to(self.device).eval()

        # BTC chord model (+ meta)
        self.hp        = HParams.load(self.hparams)
        self.btc       = BTC_model(config=self.hp.model).to(self.device).eval()
        btc_sd         = torch.load(self.ckpt_btc,map_location="cpu")
        self.btc.load_state_dict(btc_sd["model"])
        self.btc_mean, self.btc_std = btc_sd["mean"], btc_sd["std"]
        self.n_timestep = self.hp.model["timestep"]   # == 108

        # tags
        tag_file = root/"inference/data/tag_list.npy"
        self.mood_names=[t.replace("mood/theme---","")
                         for t in np.load(tag_file)[127:]]

    # ────────────────────────────────────────────────────────────────────────
    def _mert_embed(self, wav:torch.Tensor, sr:int)->np.ndarray:
        segs = split_audio(wav,sr) if is_split else [wav]
        embs=[]
        for seg in segs:
            tmp=self.feat_ext.extract_features_from_segment(seg,sr)
            if tmp.shape[0]>=7:
                sel=np.concatenate([tmp[:,5,:],tmp[:,6,:]],axis=1)  # (1,1536)
            else:                                                   # fallback
                sel=np.tile(tmp.mean(axis=0,keepdims=True), (1,2))  # (1,1536)
            embs.append(sel.squeeze())
        return np.mean(embs,axis=0).astype(np.float32)

    # ────────────────────────────────────────────────────────────────────────
    def _btc_chord_sequence(self, audio:str)->np.ndarray:
        feat,pps,_ = audio_file_to_features(audio,self.hp)
        feat       = ((feat.T - self.btc_mean) / self.btc_std)    # (T, F)
        # pad to multiple of 108
        pad = (-len(feat)) % self.n_timestep
        if pad: feat = np.pad(feat,((0,pad),(0,0)))
        feat_t = torch.tensor(feat,dtype=torch.float32,device=self.device)
        n_blk  = feat_t.shape[0] // self.n_timestep
        preds  = []

        with torch.no_grad():
            for b in range(n_blk):
                blk = feat_t[b*self.n_timestep:(b+1)*self.n_timestep].unsqueeze(0)
                attn,_  = self.btc.self_attn_layers(blk)
                pred,_  = self.btc.output_layer(attn)          # (1,108)
                preds.extend(pred.squeeze().tolist())
        preds = np.array(preds,dtype=np.int64)
        return preds[:100] if len(preds)>=100 else np.pad(preds,(0,100-len(preds)))

    # ────────────────────────────────────────────────────────────────────────
    def predict(self, audio:str, threshold:float=0.5)->dict:
        # 1) waveform + MERT embedding --------------------------------------
        wav, sr = torchaudio.load(audio)
        wav = wav.mean(0)                    # collapse to mono (time,)
        wav, sr = resample_waveform(wav, sr, resample_rate)
        wav = wav.squeeze()                  # ensure 1D
        print("wav shape before _mert_embed:", wav.shape)  # debug print
        mert = torch.tensor(
            self._mert_embed(wav, sr), dtype=torch.float32, device=self.device
        )

        # 2) chord ids (root/attr simplified = same ids) --------------------
        btc_chord_ids = self._btc_chord_sequence(audio)
        mapped_chord_ids = [idx % 14 for idx in btc_chord_ids]
        chord_ids = torch.tensor(mapped_chord_ids, dtype=torch.long, device=self.device)
        print(f"chord_ids min: {chord_ids.min().item()}, max: {chord_ids.max().item()}, shape: {chord_ids.shape}")
        print(f"mood_model.chord_root_embedding.num_embeddings: {self.mood_model.chord_root_embedding.num_embeddings}")
        mode      = torch.zeros((1,1),dtype=torch.long,device=self.device) # major

        inp = {"x_mert":        mert.unsqueeze(0),
               "x_chord":       chord_ids.unsqueeze(0),
               "x_chord_root":  chord_ids.unsqueeze(0),
               "x_chord_attr":  chord_ids.unsqueeze(0),
               "x_key":         mode}

        with torch.no_grad():
            cls,reg = self.mood_model({k:v.to(self.device) for k,v in inp.items()})
        probs      = torch.sigmoid(cls).squeeze().cpu().numpy()
        moods      = [self.mood_names[i] for i,p in enumerate(probs) if p>threshold]
        val,aro    = reg.squeeze().cpu().tolist()
        return {"valence":val,"arousal":aro,"moods":moods}
