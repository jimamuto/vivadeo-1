"""Modal deployment for Qwen/Qwen3-VL-Embedding-2B.

Deploy:
  modal deploy sentrysearch/modal_app.py
"""

import os
import subprocess
import tempfile
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DIMENSIONS = 768

app = modal.App("sentrysearch-qwen3-vl-embedding-2b")
model_volume = modal.Volume.from_name("qwen3-vl-embedding-2b-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "accelerate",
        "huggingface_hub",
        "qwen-vl-utils",
        "torch>=2.0",
        "torchvision>=0.15,<0.22",
        "transformers>=5.3",
    )
)


@app.cls(
    image=image,
    gpu="L40S",
    memory=32768,
    timeout=900,
    scaledown_window=300,
    volumes={"/models": model_volume},
)
class QwenEmbedder:
    @modal.enter()
    def load(self):
        import torch
        import torch.nn.functional as F  # noqa: F401
        from transformers.cache_utils import Cache
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLConfig,
            Qwen3VLModel,
            Qwen3VLPreTrainedModel,
        )
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        from transformers.processing_utils import Unpack
        from transformers.utils import TransformersKwargs

        os.environ["HF_HOME"] = "/models/huggingface"

        class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            config: Qwen3VLConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()

            def get_input_embeddings(self):
                return self.model.get_input_embeddings()

            def set_input_embeddings(self, value):
                self.model.set_input_embeddings(value)

            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                cache_position=None,
                **kwargs,
            ):
                return self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    **kwargs,
                )

        self._torch = torch
        self._F = torch.nn.functional
        self._processor = Qwen3VLProcessor.from_pretrained(
            MODEL_ID,
            padding_side="right",
            cache_dir="/models/huggingface",
        )
        self._model = Qwen3VLForEmbedding.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir="/models/huggingface",
        ).to("cuda")
        self._model.eval()

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        import torch

        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @staticmethod
    def _truncate_and_normalize(embedding, target_dims):
        import torch

        truncated = embedding[:target_dims]
        norm = torch.linalg.norm(truncated)
        if norm > 0:
            truncated = truncated / norm
        return truncated.cpu().float().tolist()

    def _embed_conversation(self, conversation):
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversation,
            return_video_metadata=True,
            return_video_kwargs=True,
        )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self._processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pooling_last(
                outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return self._truncate_and_normalize(embeddings[0], DIMENSIONS)

    @modal.method()
    def embed_text(self, text: str) -> list[float]:
        return self._embed_conversation([
            {
                "role": "system",
                "content": [{"type": "text", "text": "Retrieve videos relevant to the query."}],
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ])

    @modal.method()
    def embed_image(self, image_bytes: bytes, filename: str = "query.jpg") -> list[float]:
        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            path = tmp.name
        try:
            return self._embed_conversation([
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Retrieve videos relevant to the query."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "file://" + path}],
                },
            ])
        finally:
            os.unlink(path)

    def _embed_video_bytes(self, video_bytes: bytes, filename: str = "chunk.mp4") -> list[float]:
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            source_path = tmp.name
        frame_path = source_path + ".jpg"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", source_path,
                    "-frames:v", "1",
                    "-vf", "scale=-2:336",
                    "-q:v", "4",
                    frame_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            print("sentrysearch: embedding video chunk as single extracted frame", flush=True)
            return self._embed_conversation([
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Represent the video for retrieval."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "file://" + frame_path}],
                },
            ])
        finally:
            for path in (source_path, frame_path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    @modal.method()
    def embed_video(self, video_bytes: bytes, filename: str = "chunk.mp4") -> list[float]:
        return self._embed_video_bytes(video_bytes, filename)

    @modal.method()
    def embed_videos(self, items: list[tuple[bytes, str]]) -> list[list[float]]:
        print(
            f"sentrysearch: embedding video batch of {len(items)} chunks as extracted frames",
            flush=True,
        )
        return [
            self._embed_video_bytes(video_bytes, filename)
            for video_bytes, filename in items
        ]
