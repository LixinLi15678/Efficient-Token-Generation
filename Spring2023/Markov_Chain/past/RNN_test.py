import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

# 1. Preprocess the dataset
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

train_iter = WikiText2(split='train')
test_iter = WikiText2(split='test')

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, text_iter, tokenizer, vocab):
        self.data = [vocab[token] for token in tokenizer(" ".join(text_iter))]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 2. Create the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden

# Hyperparameters
embed_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.5
num_epochs = 2
batch_size = 32


# Instantiate the model
model = RNNModel(len(vocab), embed_size, hidden_size, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TextDataset(train_iter, tokenizer, vocab)
test_dataset = TextDataset(test_iter, tokenizer, vocab)

train_data = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
test_data = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

model_path = "./past/rnn_model.pth"

def train():
    print("Training the model...")
    for epoch in range(num_epochs):
        model.train()
        # Wrap the train_data DataLoader with tqdm
        progress_bar = tqdm(enumerate(train_data), total=len(train_data), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, data in progress_bar:
            inputs = data[:-1]
            targets = data[1:]
            hidden = None

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()

            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())
            if i == 100:
                break

    # Save the model's state_dict
    torch.save(model.state_dict(), model_path)
    print("Model saved!")

train()

# Load the saved model's state_dict
loaded_model = RNNModel(len(vocab), embed_size, hidden_size, num_layers, dropout)
loaded_model.load_state_dict(torch.load(model_path))
print("Model loaded!")

# 4. Test the model on the "wiki.test.tokens" dataset
loaded_model.eval()
total = 0
correct = 0
hidden = None

# Wrap the test_data DataLoader with tqdm
progress_bar = tqdm(enumerate(test_data), total=len(test_data), desc="Testing")

for i, data in progress_bar:
    inputs = data[:-1]
    targets = data[1:]
    outputs, hidden = loaded_model(inputs, hidden)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)  # Use targets.size(0) instead of targets.size(0) * targets.size(1)
    correct += (predicted == targets.view(-1)).sum().item()

    # Update the progress bar
    progress_bar.set_postfix(correct=correct, total=total)

# Calculate the accuracy of the model
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}%")
