import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from sklearn.utils import resample
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import concatenate, zeros
from numpy import isnan, isinf, sum, sqrt, triu
from numpy.random import seed
from scipy.spatial import distance

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
# tokenize text
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

# Map tokens to words
words = text.split()
token_to_word_mapping = []
current_word_idx = 0
decoded_text = ""
for token_id in inputs["input_ids"][0]:
    token = tokenizer.decode([token_id])
    decoded_text += token
    token_to_word_mapping.append(current_word_idx)
    if decoded_text.strip() == words[current_word_idx]:
        current_word_idx += 1
        decoded_text = ""

# Initialize storage for embeddings
input_embeddings = None  # To store the input embeddings
layer_embeddings = {}  # To store the embeddings for each transformer layer

# Hook function for embedding layer
def embedding_hook(module, input, output):
    global input_embeddings
    input_embeddings = output.detach().cpu()  # Move embeddings to CPU for storage

# Register hook on embedding layer
embedding_layer = lama.model.embed_tokens
embedding_layer.register_forward_hook(embedding_hook)

# Hook function for layer embeddings
def layer_hook(layer_id):
    def hook(module, input, output):
        layer_embeddings[layer_id] = output[0].detach().cpu()  # Move embeddings to CPU for storage
    return hook

# Attach hooks to all transformer layers
hooks = []
for layer_id, layer in enumerate(lama.model.layers):
    hooks.append(layer.register_forward_hook(layer_hook(layer_id)))

# Forward pass to capture embeddings
with torch.no_grad():
    _ = lama(**inputs)

# Remove hooks
for hook in hooks:
    hook.remove()

# Aggregate embeddings for each word - input + 16 layers embeddings
word_embeddings = {layer_id: [] for layer_id in range(len(lama.model.layers) + 1)}  # Include input embeddings
for word_idx in range(len(words)):
    # Indices of tokens for this word
    token_indices = [i for i, w_idx in enumerate(token_to_word_mapping) if w_idx == word_idx]

    if token_indices:
        # Aggregate input embeddings for the word
        input_word_embedding = input_embeddings[0, token_indices[0], :]
        word_embeddings[0].append(input_word_embedding)

        # Aggregate layer embeddings for the word
        for layer_id in range(len(lama.model.layers)):
            layer_word_embedding = layer_embeddings[layer_id][0][token_indices[0], :]
            word_embeddings[layer_id + 1].append(layer_word_embedding)


print("Nouns: ", len(classes[classes == "NOUN"]), ", Verbs: ", len(classes[classes == "VERB"]), ", Proper Nouns: ",
      len(classes[classes == "PROPN"]), ", Adjectives: ", len(classes[classes == "ADJ"]))
#shift classes one to the left
classes_shift = classes[1:]
classes_shift = np.concatenate((classes_shift, ["X"]))
len(classes_shift), classes_shift


# Define MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# Training loop
def train_model(model, train_loader, val_loader, epochs=20):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    print("Training complete. Best validation accuracy:", best_val_acc)

def plot_confusion_matrix(y_true, y_pred, classes, layer, accuracy):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3], normalize="true")
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, vmin=0, vmax=1, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("Confusion Matrix for Layer " + str(layer) + ", Accuracy:" + str(accuracy))
    plt.savefig("Prediction_Layer_" + str(layer) + ".pdf", bbox_inches="tight", dpi=200)
    plt.show()


n_pred = []
v_pred = []
a_pred = []
p_pred = []
# loop through Llama layers
for i in range(len(word_embeddings)):
    # get embeddings for each wordclass (NOUN, VERB, ADJ, PROPN)
    word_emb = word_embeddings.copy()
    print(type(word_emb[i]), type(classes), type(classes_shift))
    mask = (classes_shift == "NOUN")
    nouns_x = np.array(word_emb[i])[mask]
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "VERB")
    verbs_x = np.array(word_emb[i])[mask]
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "ADJ")
    adj_x = np.array(word_emb[i])[mask]
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "PROPN")
    propn_x = np.array(word_emb[i])[mask]

    # downsample other wordclasses to the amount of the smallest class for data balance
    min_amount = min(len(nouns_x), len(verbs_x), len(adj_x), len(propn_x))
    print(min_amount)
    nouns_x = resample(nouns_x, replace=False, n_samples=min_amount, random_state=42)
    verbs_x = resample(verbs_x, replace=False, n_samples=min_amount, random_state=42)
    adj_x = resample(adj_x, replace=False, n_samples=min_amount, random_state=42)
    propn_x = resample(propn_x, replace=False, n_samples=min_amount, random_state=42)

    # concatenate resulting embeddings for the four wordclasses
    X = np.concatenate((nouns_x, verbs_x, adj_x, propn_x))
    # generate label list y
    nouns_y = [0] * len(nouns_x)
    verbs_y = [1] * len(verbs_x)
    adj_y = [2] * len(adj_x)
    propn_y = [3] * len(propn_x)
    y = np.concatenate((nouns_y, verbs_y, adj_y, propn_y))

    print("Total X:", len(X), ", Total y: ", len(y))

    # Split data into train, validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    input_dim = 2048
    hidden_dim = 512
    num_classes = 4
    model = MLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    train_model(model, train_loader, val_loader, epochs=20)

    # Test the model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    test_acc = test_correct / test_total
    print("Test Accuracy:", test_acc)
    print(classification_report(all_labels, all_preds,
                                target_names=["Noun", "Verb", "Adjective", "Proper Noun"]))
    # plot confusion matrices
    plot_confusion_matrix(all_labels, all_preds, classes=["Noun", "Verb", "Adjective", "Proper Noun"], layer=i,
                          accuracy=round(test_acc, 2))

    # save progress of accurate classified classes
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    n_pred.append(len(all_preds[(all_labels == 0) & (all_preds == 0)]) / len(all_labels[all_labels == 0]))
    v_pred.append(len(all_preds[(all_labels == 1) & (all_preds == 1)]) / len(all_labels[all_labels == 1]))
    a_pred.append(len(all_preds[(all_labels == 2) & (all_preds == 2)]) / len(all_labels[all_labels == 2]))
    p_pred.append(len(all_preds[(all_labels == 3) & (all_preds == 3)]) / len(all_labels[all_labels == 3]))

save_v = np.concatenate((n_pred, v_pred, a_pred, p_pred))

plt.figure(figsize=(10,6))
plt.plot(n_pred, color="red", label="Noun", linewidth=2)
plt.plot(v_pred, color="blue", label="Verb", linewidth=2)
plt.plot(a_pred, color="orange", label="Adj", linewidth=2)
plt.plot(p_pred, color="green", label="Propn", linewidth=2)
plt.ylim(0,1)
plt.xlabel("Layer", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.savefig("Process_Wordclasses.pdf", bbox_inches="tight", dpi=200)
plt.show()

#compute GDV of embeddings

def zScoreSpecial(data):

  # get parameters
  NC = len(data) # nr. of clusters
  ND = data[0].shape[1] # nr. of dimensions

  # copy data --> zData
  zData = []
  for C in range(NC):
    arr = data[C].copy()
    zData.append(arr)

  # compute means and STDs for each dimension, over ALL data
  all = concatenate(zData)
  mu =  zeros(shape=ND, dtype=float)
  sig = zeros(shape=ND, dtype=float)
  for D in range(ND):
    mu[D]  = all[:,D].mean()
    sig[D] = all[:,D].std()

  # z-score the data in each cluster
  for C in range(NC):
    for D in range(ND):
      zData[C][:,D] = ( zData[C][:,D] - mu[D] ) / ( 2 * sig[D] )

  # replace nan and inf by 0
  for C in range(NC):
    nanORinf = isnan(zData[C]) | isinf(zData[C])
    zData[C][ nanORinf ] = 0.0

  return zData

def computeGDV(data):

  '''
  Returns the Generalized Discrimination Value
  as well as intraMean and interMean

  data is expected to be a list of label-sorted point 'clusters':
  data = [cluster1, cluster2, ...]

  Each cluster is a NumPy matrix,
  and the rows of this matrix
  are n-dimensional data vectors,
  each belonging to the same label.
  '''

  # get parameters
  NC = len(data) # nr. of clusters
  ND = data[0].shape[1] # nr. of dimensions

  # copy data --> zData
  zData = []
  for C in range(NC):
    arr = data[C].copy()
    zData.append(arr)

  # dimension-wise z-scoring
  zData = zScoreSpecial(zData)

  # intra-cluster distances
  dIntra = zeros(shape=NC, dtype=float)
  for C in range(NC):
    NP = zData[C].shape[0]
    dis = distance.cdist(zData[C], zData[C], 'euclidean')
    # dis is symmetric with zero diagonal
    dIntra[C] = sum(dis) / (NP*(NP-1)) # divide by nr. of non-zero el.
  #print('dIntra = ',dIntra)

  # inter-cluster distances
  dInter = zeros(shape=(NC,NC), dtype=float)
  for C1 in range(NC):
    NP1 = zData[C1].shape[0]
    for C2 in range(NC):
      NP2 = zData[C2].shape[0]
      dis = distance.cdist(zData[C1], zData[C2], 'euclidean')
      dInter[C1][C2] = sum(dis) / (NP1*NP2) # divide by nr. of non-zero el.
  #print('dInter =\n',dInter)

  # compute GDV
  pre = 1.0 / sqrt(float(ND))
  intraMean = dIntra.mean()
  interMean = sum( triu(dInter,k=1) ) / (NC*(NC-1)/2) # divide by nr. of non-zero el.
  #print('intraMean=',intraMean,'\ninterMean=',interMean)
  gdv = pre * (intraMean - interMean)

  return pre*intraMean, pre*interMean,gdv

gdv = []
for i in range(len(word_embeddings)):
    # get embeddings for each wordclass
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "NOUN")
    nouns_x = np.array(word_emb[i])[mask]
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "VERB")
    verbs_x = np.array(word_emb[i])[mask]
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "ADJ")
    adj_x = np.array(word_emb[i])[mask]
    word_emb = word_embeddings.copy()
    mask = (classes_shift == "PROPN")
    propn_x =np.array(word_emb[i])[mask]

    # downsample each class to the amount of the smallest class
    min_amount = min(len(nouns_x), len(adj_x))
    print(min_amount)
    nouns_x = resample(nouns_x, replace=False, n_samples=min_amount, random_state=42)
    verbs_x = resample(verbs_x, replace=False, n_samples=min_amount, random_state=42)
    adj_x = resample(adj_x, replace=False, n_samples=min_amount, random_state=42)
    propn_x = resample(propn_x, replace=False, n_samples=min_amount, random_state=42)

    print("Nouns: ", len(nouns_x), ", Verbs: ", len(verbs_x), ", Adjectives: ", len(adj_x), ", Proper Nouns: ", len(propn_x))

    _, _, gdv_ = computeGDV([np.array(nouns_x), np.array(verbs_x), np.array(adj_x), np.array(propn_x)])
    gdv.append(gdv_)

plt.figure(figsize=(10,6))
plt.plot(gdv, linewidth=2)
plt.xlabel("Layer", fontsize=20)
plt.ylabel("GDV", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("GDV_plot.pdf", bbox_inches="tight", dpi=200)

