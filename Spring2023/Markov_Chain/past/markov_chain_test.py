import random
import json
from sklearn.metrics import f1_score


def mc_test():
    with open('wikitext-2/wiki.test.tokens', 'r') as f:
        test_data = f.read().split()

    with open('transitions_matrix.json', 'r') as f:
        transitions = json.load(f)

    # generate some sample text
    current_token = random.choice(test_data)
    print(current_token)
    # current_token = "Hello"
    result = current_token.capitalize()

    while len(result) < 20:
        if current_token not in transitions:
            # randomly choose a token from the whole list
            current_token = random.choice(list(transitions.keys()))
            result += ' ' + current_token
            continue
        
        # get the probabilities for the next tokens
        probs = transitions[current_token]
        
        # sort by probability (descending)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # get the top 3 most probable next tokens
        top_next_tokens = [x[0] for x in sorted_probs[:3]]
        print(top_next_tokens)

        # choose next token
        if top_next_tokens[0] == '<unk>':
            next_token = top_next_tokens[1]
        else:
            next_token = top_next_tokens[0]
        
        current_token = next_token
        result += ' ' + current_token
        
    print(result)

if __name__ == '__main__':
    mc_test()