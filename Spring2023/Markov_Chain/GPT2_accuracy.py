import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.auto import tqdm
import math
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()  # Set the model to evaluation mode

def calculate_perplexity(model, test_corpus):
    total_loss = 0
    total_tokens = 0

    for sentence in tqdm(test_corpus, desc="Calculating perplexity"):
        tokens = tokenizer.encode(sentence, return_tensors="pt")
        n_tokens = tokens.shape[1] - 1  # Remove the last token (end-of-sequence)
        
        if n_tokens < 1:
            continue

        with torch.no_grad():
            outputs = model(tokens[:, :-1])  # Remove the last token
            logits = outputs.logits
            
            # Flatten the logits and target tensors before computing the cross-entropy loss
            logits = logits.view(-1, logits.size(-1))
            target = tokens[:, 1:].view(-1)

            loss = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
            total_loss += loss.item()
            total_tokens += n_tokens

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def calculate_kl_divergence_gpt2(model, test_corpus, order=2):
    kl_divergence = 0
    total_tokens = 0
    test_model = defaultdict(Counter)

    for sentence in tqdm(test_corpus, desc="Calculating KL-divergence", ncols=80):
        tokens = word_tokenize(sentence)
        for i in range(len(tokens) - order):
            state = tuple(tokens[i:i + order])
            next_word = tokens[i + order]
            test_model[state][next_word] += 1
            total_tokens += 1

    for state in tqdm(test_model, desc="Processing states", ncols=80):
        state_tensor = tokenizer.encode(" ".join(state), return_tensors="pt")
        with torch.no_grad():
            output = model(state_tensor)
            logits = output.logits[:, -1, :]  # Get the logits for the last token only
            probs = torch.softmax(logits, dim=-1).squeeze()

        for next_word in test_model[state]:
            test_prob = test_model[state][next_word] / total_tokens
            next_word_id = tokenizer.encode(next_word, add_special_tokens=False)[0]

            model_prob = probs[next_word_id].item()
            kl_divergence += test_prob * math.log(test_prob / model_prob)

    return kl_divergence

if __name__ == '__main__':
    # Read the datasets
    with open("wikitext-2/wiki.train.tokens", "r") as f:
        train_data = f.read().split("\n")

    with open("wikitext-2/wiki.test.tokens", "r") as f:
        test_data = f.read().split("\n")

    # Calculate the perplexity
    # print("Calculating perplexity...")
    # perplexity = calculate_perplexity(model, test_data)
    # print("Perplexity:", perplexity)

    # Calculate the KL-divergence
    print("Calculating KL-divergence...")
    kl_divergence = calculate_kl_divergence_gpt2(model, test_data)
    print("KL-divergence:", kl_divergence)