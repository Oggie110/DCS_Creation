# music2emo/utils/mert.py
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModel


class FeatureExtractorMERT:
    """
    Lightweight wrapper around m-a-p/MERT-v1-95M that returns layer-wise
    embeddings (12 × 768 per segment).
    """

    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M",
                 device: str | torch.device | None = None,
                 sr: int = 24_000) -> None:
        self.sr   = sr
        self.name = model_name

        if device is None or device == "None":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = AutoModel.from_pretrained(
            self.name,
            trust_remote_code=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.name, trust_remote_code=True
        )

    # --------------------------------------------------------------------- #
    #                                PUBLIC                                 #
    # --------------------------------------------------------------------- #
    def extract_features_from_segment(
        self,
        segment: torch.Tensor,      # 1-D (time) **or** 2-D (chan, time)
        sample_rate: int,
        save_path: str | None = None,
    ) -> np.ndarray:
        """
        Return array with shape **(1, 12, 768)**.
        If *save_path* is given the array is additionally written to disk.
        """
        # ───  ensure 1-D (samples)  ───────────────────────────────────────
        if segment.ndim == 1:
            pass  # already correct
        elif segment.ndim == 2:
            segment = segment.mean(0)  # collapse to mono, 1D
        elif segment.ndim > 2:
            segment = segment.squeeze()
            if segment.ndim > 1:
                segment = segment.mean(0)
        else:
            raise ValueError(f"Unexpected audio shape {segment.shape}")

        segment = segment.float().to(self.device)

        inputs = self.processor(
            segment, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outs = self.model(**inputs, output_hidden_states=True)

        hidden = torch.stack(outs.hidden_states[1:], dim=0)  # (12, B, T, 768)
        hidden = hidden.mean(dim=2).permute(1, 0, 2)         # (B, 12, 768)
        hidden = hidden.cpu().numpy()

        if save_path:
            np.save(save_path, hidden)

        return hidden
