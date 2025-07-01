import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import gc
import matplotlib.pyplot as plt
from numpy.random import seed

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # For multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Seed worker for DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

#load model and tokenizer
model_path = ".../model"    # load Llama 3.2 model
lama = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model Layers: ", len(lama.model.layers))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lama.to(device)

#get transcript of audiobook for input text to the model
text = []
wordclasses = []
chapter = ["001", "002", "003", "006", "007", "004", "005", "008", "009", "015", "016", "017", "018", "019", "020", "024", "025", "026", "027", "028"]
for i in chapter:
    words = np.array(pd.read_excel(r"Transcript/" + i + "_words.xlsx")["Word"])    #get transcript of each chapter
    classes = np.array(pd.read_excel(r"Transcript/" + i + "_wordclass.xlsx")["Spacy"])
    text.append(words)
    wordclasses.append(classes)
wordclasses = np.concatenate(wordclasses)
text = np.concatenate(text) #concatenate texts
text = " ".join(text)

#get probability of next word
def get_probs(german_text, specific_word):
    # Tokenize the input text
    input_tokens = tokenizer(german_text, return_tensors="pt").to(device)

    # Tokenize the specific word and handle subword tokenization
    specific_token_ids = tokenizer.encode(specific_word, add_special_tokens=False)
    # print(f"Tokenized IDs for '{specific_word}': {specific_token_ids}")

    # Pass the input through the model
    with torch.no_grad():
        outputs = lama(**input_tokens)

    # Extract logits
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Initialize variables for combined probability
    combined_probability = []

    # Compute probabilities for each token in the specific word
    current_input_tokens = input_tokens["input_ids"]  # Current sequence of tokens
    for token_id in specific_token_ids:
        # Get the logits for the next token prediction
        next_token_logits = lama(input_ids=current_input_tokens)["logits"][0, -1, :]

        # Compute probabilities
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        token_probability = next_token_probs[token_id].item()
        combined_probability.append(token_probability)

        # Add the predicted token to the input sequence for the next step
        current_input_tokens = torch.cat([current_input_tokens, torch.tensor([[token_id]], device=device)], dim=-1)
    prob = np.mean(combined_probability)
    return prob


# loop through text, get the probability of the next word and save them
probs = []
for i in range(1, len(text)-1):
    pred = 0
    if classes[i+1] == "NOUN" or classes[i+1] == "VERB" or classes[i+1] == "ADJ" or classes[i+1] == "PROPN":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        # limit the amount of words to 200 for computational efficiency
        if i > 200:
            t = text[i-200:i+1]
        else:
            t = text[:i+1]
        t = " ".join(t)
        predword = text[i+1]
        probs.append((i+1, predword, get_probs(t, predword)))


# save the probabilities for each word class (NOUN, VERB, ADJ, PROPN)
nouns = []
verbs = []
adj = []
propn = []
for i,word,prob in probs:
    if classes[i] == "NOUN":
        nouns.append(prob)
    elif classes[i] == "VERB":
        verbs.append(prob)
    elif classes[i] == "ADJ":
        adj.append(prob)
    elif classes[i] == "PROPN":
        propn.append(prob)

# plot the distribution
plt.figure(figsize=(10,6))
plt.bar(["Nouns", "Verbs", "Adjectives", "Proper Nouns"], [np.mean(nouns), np.mean(verbs), np.mean(adj), np.mean(propn)], color=["red", "blue", "orange", "green"], alpha=0.8)
plt.xticks(fontsize=20)
plt.ylabel("Average Predictability", fontsize=20)
plt.yticks(ticks=[0, 0.1, 0.2, 0.3], fontsize=20)
plt.savefig("semantic_predictability.pdf", bbox_inches="tight", dpi=200)
