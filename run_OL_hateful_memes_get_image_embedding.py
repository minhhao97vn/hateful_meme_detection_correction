import torch
from PIL import Image
import open_clip
import json
import numpy as np

VISION_ENCODER_SIZE = (224, 224)
model_name = 'ViT-g-14'
pretrained_data = 'laion2b_s34b_b88k'

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_data)
tokenizer = open_clip.get_tokenizer(model_name)

model.to('cuda:0')

# for item in open_clip.list_pretrained():
#     print(item)

image_path = "/home/haovan/hateful_memes/"

with open(image_path + "test_seen" + '.jsonl', 'r') as file:
    data_list = list(file)

img_embedding = {}

for idx, json_str in enumerate(data_list):
    data_item = json.loads(json_str)
    image = preprocess(Image.open(image_path + data_item['img']).resize(VISION_ENCODER_SIZE)).unsqueeze(0).to('cuda:0')

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        img_embedding[data_item['id']] = image_features

torch.save(img_embedding, "/home/haovan/hateful_memes/test_seen_meme_embeddings.pt")
print("Done!")