import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, activation_function, dropout_rate):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation_function
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        x = self.activation(self.conv1(embedded))
        x = self.activation(self.conv2(x))
        x = nn.functional.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x



































