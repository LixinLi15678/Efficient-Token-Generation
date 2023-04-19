from collections import defaultdict, Counter
import random
import json
import nltk
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
import math
from transformers import GPT2Tokenizer

nltk.download('punkt')  # Download the tokenizer's data

def train_markov_chain(corpus, order=2):
    print("Training the Markov chain...")
    model = defaultdict(Counter)

    for sentence in tqdm(corpus, desc="Training"):
        tokens = word_tokenize(sentence)  # Replace GPT-2 tokenizer with word_tokenize
        for i in range(len(tokens) - order):
            state = tuple(tokens[i:i + order])
            next_word = tokens[i + order]
            model[state][next_word] += 1

    return model

def generate_text(model, seed, length=50, order=2):
    current_state = seed
    output = list(current_state)
    for _ in range(length):
        next_word = random.choices(
            list(model[current_state].keys()),
            list(model[current_state].values())
        )[0]
        output.append(next_word)
        current_state = tuple(output[-order:])
    return " ".join(output)

def calculate_accuracy(model, test_corpus, order=2):
    correct_predictions = 0
    total_predictions = 0
    for sentence in tqdm(test_corpus, desc="Calculating accuracy"):
        tokens = word_tokenize(sentence)  # Replace GPT-2 tokenizer with word_tokenize
        for i in range(len(tokens) - order):
            state = tuple(tokens[i:i + order])
            true_next_word = tokens[i + order]

            # Check if the state exists in the model
            if state in model:
                predicted_next_word = random.choices(
                    list(model[state].keys()),
                    list(model[state].values())
                )[0]
                if predicted_next_word == true_next_word:
                    correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy

def calculate_accuracy_gpt2(model, test_corpus, order=2):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    correct_predictions = 0
    total_predictions = 0
    for sentence in tqdm(test_corpus, desc="Calculating accuracy"):
        tokens = tokenizer.tokenize(sentence)
        for i in range(len(tokens) - order):
            state = tuple(tokens[i:i + order])
            true_next_word = tokens[i + order]

            # Check if the state exists in the model
            if state in model:
                predicted_next_word = random.choices(
                    list(model[state].keys()),
                    list(model[state].values())
                )[0]
                if predicted_next_word == true_next_word:
                    correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    return accuracy

def calculate_kl_divergence(model, test_corpus, order=2):
    kl_divergence = 0
    total_tokens = 0
    test_model = defaultdict(Counter)

    for sentence in tqdm(test_corpus, desc="Calculating KL-divergence"):
        tokens = word_tokenize(sentence)
        for i in range(len(tokens) - order):
            state = tuple(tokens[i:i + order])
            next_word = tokens[i + order]
            test_model[state][next_word] += 1
            total_tokens += 1

    for state in test_model:
        if state not in model:
            continue

        for next_word in test_model[state]:
            test_prob = test_model[state][next_word] / total_tokens
            if next_word in model[state]:
                model_prob = model[state][next_word] / sum(model[state].values())
            else:
                model_prob = 1e-10  # Smoothing factor to avoid division by zero

            kl_divergence += test_prob * math.log(test_prob / model_prob) 

    return kl_divergence # KL(P || Q) = Σ P(x) * log(P(x) / Q(x))


def save_model_to_json(model, filename):
    # Convert defaultdict to a regular dict before saving
    model_to_save = {json.dumps(k): dict(v) for k, v in model.items()}
    
    with open(filename, "w") as f:
        json.dump(model_to_save, f)
        print(f"Model saved to {filename}")

def load_model_from_json(filename):
    with open(filename, "r") as f:
        loaded_model = json.load(f)

    # Convert the loaded dict back to a defaultdict
    model = defaultdict(Counter)
    for k, v in loaded_model.items():
        model[tuple(json.loads(k))] = Counter(v)

    return model

if __name__ == '__main__':
    # Read the datasets
    with open("wikitext-2/wiki.train.tokens", "r") as f:
        train_data = f.read().split("\n")

    with open("wikitext-2/wiki.test.tokens", "r") as f:
        test_data = f.read().split("\n")

    order = 2
    # Train the model
    # markov_model = train_markov_chain(train_data, order=order)

    # Save the model
    # save_model_to_json(markov_model, "markov_model_nltk.json")

    # Load the model (NLTK)
    markov_model = load_model_from_json("markov_model_nltk.json")
    print("Model loaded successfully!")

    # Load the model (GPT-2)
    # markov_model = load_model_from_json("markov_model_gpt2.json")
    # print("Model loaded successfully! (GPT2)")


    # Calculate the accuracy using NLTK tokenizer
    print("Calculating accuracy...")
    accuracy = calculate_accuracy(markov_model, test_data, order=order)
    print(f"Accuracy: {accuracy}%")

    # Calculate the accuracy using GPT-2 tokenizer
    # print("Calculating accuracy...")
    # accuracy = calculate_accuracy_gpt2(markov_model, test_data, order=order)
    # print(f"Accuracy: {accuracy}%")

    # Calculate the KL-divergence
    print("Calculating KL-divergence...")
    kl_divergence = calculate_kl_divergence(markov_model, test_data, order=order)
    print(f"KL-divergence: {kl_divergence}")

    