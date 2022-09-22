from typing import List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_id = "openai/clip-vit-large-patch14"  # downloaded to ./weights
        self.model = CLIPModel.from_pretrained(
            f"weights/{model_id}", local_files_only=True
        ).to("cuda")
        self.processor = CLIPProcessor.from_pretrained(
            f"weights/{model_id}", local_files_only=True
        )

    def predict(
        self,
        image: Path = Input(description="Input Image."),
        text: str = Input(
            description='Description of the image, separate different descriptions with "|"'
        ),
    ) -> List[float]:

        image = Image.open(str(image))
        text = [t.strip() for t in text.split("|")]

        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding=True
        ).to("cuda")

        outputs = self.model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        probs = logits_per_image.softmax(
            dim=1
        )  # we can take the softmax to get the label probabilities
        return probs.tolist()[0]
