### THIS FILE CONTAINS THE DEFINITION OF THE LSTM CLASS ###
import torch
import torch.nn as nn

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout_prob):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        out, lstm_hidden = self.lstm(x, hidden)  # Size [batch_size, n_days, n_features]
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out[:, :, 0].clone()
        return out, lstm_hidden

    def init_hidden_zero(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

    def init_hidden_normal(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).normal_(0, 1).to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_size).normal_(0, 1).to(device))
        return hidden



if __name__ == '__main__':
    pass