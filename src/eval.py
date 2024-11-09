from sentence_transformers import CrossEncoder
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load the Entailment model
ent_model = CrossEncoder('cross-encoder/nli-roberta-base', device='cpu')

#Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#For CLIP confidence score

def clip_score(image, text):
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True, truncation = True)

    #outputs = clip_model(**inputs)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeddings = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeddings = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)


    similarity = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings)
    score = 2.5 * max(similarity.tolist()[0], 0.0)
    return score

def clip_confidence(image, answer, gen): #clip confidence score
    image_gold_sim = clip_score(image, answer)
    image_generated_sim = clip_score(image, gen)
    confidence_score = image_generated_sim / (image_generated_sim + image_gold_sim)
    return confidence_score


#For entailment score

def entailment(gen, answer):
    sentence_pairs = (gen, answer)
    scores = ent_model.predict(sentence_pairs)

    # Convert logits to probabilities using softmax
    probabilities = []
    probs = torch.nn.functional.softmax(torch.tensor(scores), dim=-1)
    probabilities = probs.tolist()
    # label_mapping = ['contradiction', 'entailment', 'neutral']
    return probabilities[1]
