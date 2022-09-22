# cog-clip

[![Replicate](https://replicate.com/andreasjansson/clip-features)](https://replicate.com/andreasjansson/clip-features)

Cog model that outputs CLIP features for text and images.

Run with the API:

```
import replicate
import numpy as np
from numpy.linalg import norm

def cos_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

inputs = """
a photo of a dog
a cat
two cats with remote controls
https://replicate.com/api/models/cjwbw/clip-vit-large-patch14/files/36b04aec-efe2-4dea-9c9d-a5faca68b2b2/000000039769.jpg
"""

# run prediction
model = replicate.models.get("andreasjansson/clip-features")
outputs = model.predict(inputs)

# output similarity of the three text lines with the image on line 4
for i in range(3):
    print(outputs[i].input)
    print(cos_sim(outputs[i].embedding, outputs[3].embedding))
    print()
```
