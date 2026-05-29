"""OpenCLIP per-attribute text embedding.

Each attribute of each frame becomes one natural-language sentence, encoded into a normalized text feature.
Per-sequence tensor is [t, a, d]: t frames, a=7 attributes (ATTR_ORDER), d=768 (ViT-L-14).
"""

import logging

import open_clip
import torch

from gcaption.schema import ATTR_ORDER, FrameCaption

logger = logging.getLogger(__name__)

_GERUND = {"stand": "standing still", "walk": "walking", "run": "running"}


def attr_value(fc: FrameCaption, attr: str) -> str | list[str]:
    """Raw attribute value used for majority voting / labels."""
    if attr == "viewpoint":
        vp = fc.viewpoint
        return f"{vp.roll.value} + {vp.pitch.value} + {vp.yaw.value}"
    v = getattr(fc, attr)
    return v.value if hasattr(v, "value") else v


def attr_sentence(fc: FrameCaption, attr: str) -> str:
    v = attr_value(fc, attr)
    if attr == "age":
        return f"The person appears to be a {v}."
    if attr == "attire":
        return f"The person is wearing {v}."
    if attr == "action":
        return f"The person is {_GERUND.get(v, v)}."
    if attr == "related_item":
        return (
            f"The person is carrying {', '.join(v)}."
            if v
            else "The person is not carrying anything."
        )
    if attr == "location":
        return f"The scene is {v}."
    if attr == "viewpoint":
        return f"The camera viewpoint is {v}."
    if attr == "lighting":
        return f"The lighting is {v}."
    raise KeyError(attr)


class Embedder:
    def __init__(self, clip_model: str, pretrained: str, device: str = "cuda"):
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
        self.model = self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(clip_model)

    # inference-only text encoding; no autograd needed
    @torch.no_grad()
    def encode_sequence(self, frames: list[FrameCaption]) -> torch.Tensor:
        """Encode each frame's 7 attributes into normalized OpenCLIP text features.

        Shapes
        ------
        frames : list of t FrameCaption
        returns: [t, a, d] float32 on CPU, L2-normalized (a=7 attributes in ATTR_ORDER, d=768)
        """
        t, a = len(frames), len(ATTR_ORDER)
        sents = [attr_sentence(fc, attr) for fc in frames for attr in ATTR_ORDER]
        # shape: t*a strings -> [t*a, s] token ids
        tok = self.tokenizer(sents).to(self.device)
        # encode_text returns the model dtype; cast to fp32 for a stable L2 norm
        # shape: [t*a, s] -> [t*a, d]
        feats = self.model.encode_text(tok).float()
        feats = torch.nn.functional.normalize(feats, dim=-1)
        # shape: [t*a, d] -> [t, a, d]
        return feats.reshape(t, a, -1).cpu()
