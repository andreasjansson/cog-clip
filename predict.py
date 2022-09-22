import re
import sys
import requests
from io import BytesIO
from typing import List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cog import BasePredictor, Path, Input, BaseModel


class NamedEmbedding(BaseModel):
    input: str
    embedding: List[float]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = CLIPModel.from_pretrained("/weights", local_files_only=True).to(
            "cuda"
        )
        self.processor = CLIPProcessor.from_pretrained(
            "/weights", local_files_only=True
        )

    def predict(
        self,
        inputs: str = Input(
            description="Newline-separated inputs. Can either be strings of text or image URIs starting with http[s]://",
            default="a\nb",
        ),
    ) -> List[NamedEmbedding]:

        lines = []
        texts = []
        image_urls = []
        images = []
        for line in inputs.strip().splitlines():
            line = line.strip()
            lines.append(line)
            if re.match("^https?://", line):
                try:
                    print(f"Downloading {line}", file=sys.stderr)
                    image = Image.open(BytesIO(requests.get(line).content))
                    images.append(image)
                    image_urls.append(line)
                except Exception as e:
                    print(f"Failed to load {line}: {e}", file=sys.stderr)
            else:
                texts.append(line)

        if not images:
            images = None
        if not texts:
            texts = None

        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to("cuda")

        if texts:
            text_embeds = self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            text_outputs = dict(zip(texts, text_embeds))
        else:
            text_outputs = {}
        if images:
            image_embeds = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            image_outputs = dict(zip(image_urls, image_embeds))
        else:
            iamge_outputs = {}

        outputs = []
        for line in lines:
            if line in text_outputs:
                outputs.append(
                    NamedEmbedding(input=line, embedding=text_outputs[line].tolist())
                )
            else:
                outputs.append(
                    NamedEmbedding(input=line, embedding=image_outputs[line].tolist())
                )

        return outputs
