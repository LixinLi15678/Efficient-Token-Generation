# Efficient Token Generation with Mixed Model Approach

## Overview
This project, conducted under the guidance of Professor McDanel, explores enhancing token generation efficiency in natural language processing (NLP). By integrating various models such as Jedi, Markov Chain, RNN, and GPT-2, the study aims to improve prediction efficiency without significantly sacrificing accuracy.

## Objective
The main objective is to streamline the token generation process by combining the strengths of smaller models like Jedi and Markov Chain with more complex ones like RNN and GPT-2, thereby achieving higher efficiency in prediction tasks.

## Methodology
- **Jedi**: Utilized as an auto-complete server to suggest possible next tokens.
- **Markov Chain**: Employs a probabilistic approach to predict the next token based on the current state.
- **RNN (Recurrent Neural Network)**: Captures temporal dependencies in sequences for better prediction accuracy.
- **GPT-2**: A more advanced model that leverages deep learning to predict token sequences with high accuracy.

## Results
The study demonstrates that by setting a threshold for prediction probabilities and sequentially using models from simplest to most complex, efficiency can be significantly improved. The approach allows for quicker predictions with acceptable accuracy levels, particularly in applications where speed is crucial.

## How to Use
1. Clone this repository to your local machine.
2. Install required dependencies listed in `requirements.txt`.
3. Explore the Python scripts (`histogram.py`, `python_client.py`, `GPT2_accuracy.py`, and `MC_accuracy.py`) to understand the implementation of each model.
4. You can modify the scripts to test different datasets or model configurations.

## Future Directions
Inspired by the "Big Little Transformer Decoder," future work could involve creating a hierarchy of models ranging from small to large, further optimizing the balance between efficiency and accuracy.

## Acknowledgments
Special thanks to Professor McDanel for guidance and inspiration throughout this research project.

## References
- Vaswani et al., "Attention is All You Need"
- Various resources on tokenization, Transformer architecture, and KL-Divergence

