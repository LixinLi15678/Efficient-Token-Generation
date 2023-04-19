import random
import json

def train() -> dict:
    # read the data from the file
    with open('wikitext-2/wiki.train.tokens', 'r') as f:
        train_data = f.read().split()

    transitions = {}

    # create the transition matrix
    for i in range(len(train_data)-1):
        current_token = train_data[i]
        next_token = train_data[i+1]
        if current_token not in transitions:
            transitions[current_token] = {}
        if next_token not in transitions[current_token]:
            transitions[current_token][next_token] = 1
        else:
            transitions[current_token][next_token] += 1

    # normalize the transition probabilities
    for token in transitions:
        total = float(sum(transitions[token].values()))
        for next_token in transitions[token]:
            transitions[token][next_token] /= total

    return transitions

def write_json(data_list: dict, name: str) -> None:
    # writing json file
    with open(f"{name}.json", 'w') as f:
        json.dump(data_list, f)

if __name__ == '__main__':
    result = train()
    write_json(result, 'transitions_matrix')