import os
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader

'''
Setting up parameters and paths
Configure data path here

"data\simba_simulation\1_Datafile12_size100"
"data\simba_simulation\2_Datafile12_size1000"
"data\simba_simulation\3_Datafile13_size100"
"data\simba_simulation\4_Datafile13_size1000"
'''

data_root = "data\simba_simulation\2_Datafile12_size1000"
input_length = 20 # length of each input chunk = context size
output_length = input_length  # predict this amount of steps per chunk

# defining the dataset class as per pytorch conventions
class TimeSeriesDataset(Dataset):
    def __init__(self, signals):
        #trim signals to full chunks
        total_length = signals.shape[1]
        num_full_chunks = total_length // input_length
        trimmed_length = num_full_chunks * input_length
        signals_trimmed = signals[:, :trimmed_length]

        # reshape into (num_signals, num_full_chunks, input_length) and create input-output pairs
        chunks = signals_trimmed.reshape(signals.shape[0], num_full_chunks, input_length)
        X = chunks[:, :-1, :]
        Y = chunks[:, 1:, :]

        # flatten to (num_signals*(num_full_chunks-1), input_length)
        self.X = X.reshape(-1, input_length)
        self.Y = Y.reshape(-1, input_length)

    #how many samples are in the dataset
    def __len__(self):
        return len(self.X)
    
    #getting tuple input and output in the form of tensors
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx]).unsqueeze(-1)
        y = torch.FloatTensor(self.Y[idx])                 
        return x, y

#rnn Model
#one feature per time point, hidden size of 64, 15 stacked layers
class RNNRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=15, output_length=output_length):
        super(RNNRegressor, self).__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.output_length = output_length

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1) # only predict one step at a time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = x.size()
        device = x.device

        #encode context
        _, (hn, cn) = self.rnn(x)
        # starting decoder at last true timestep
        input_t = x[:, -1:, :]
        preds = torch.zeros(batch_size, self.output_length, device=device)

        # autoregressive loop
        for t in range(self.output_length):
            out, (hn, cn) = self.rnn(input_t, (hn, cn))
            h_t = out.squeeze(1)
            y_t = self.fc(h_t)
            preds[:, t] = y_t.squeeze(1)
            # next input is the predicted value
            input_t = y_t.unsqueeze(1)

        return preds

#Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNRegressor().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 200

# training loop
def train():
    train_losses = []
    test_losses  = []
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # training
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x_batch.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # evaluation
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                total_test_loss += loss.item() * x_batch.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)

        print(f"Epoch {epoch}/{num_epochs} — train_loss: {avg_train_loss:.6f}, test_loss: {avg_test_loss:.6f}, time: {time.time()-epoch_start:.2f}s")
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

    '''
    save model and loss history
    Adjust filenames to reflect model parameters

    "[1]rnn_model_chunks_100e_4l_5_lr_32hidden.pth"
    "[2]rnn_model_chunks_100e_4l_5_lr_32hidden.pth"
    "[3]rnn_model_chunks_100e_4l_5_lr_32hidden.pth"
    "[4]rnn_model_chunks_100e_4l_5_lr_32

    "[1]loss_history_chunks_100e_4l_5_lr_32hidden.npz"
    "[2]loss_history_chunks_100e_4l_5_lr_32hidden.npz"
    "[3]loss_history_chunks_100e_4l_5_lr_32hidden.npz"
    "[4]loss_history_chunks_100e_4l_5_lr_32hidden.npz"    
    '''

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "[2]rnn_model_chunks_100e_4l_5_lr_32hidden.pth"))
    np.savez(os.path.join("models", "[2]loss_history_chunks_100e_4l_5_lr_32hidden.npz"),
             train_losses=np.array(train_losses),
             test_losses=np.array(test_losses))
    print("Training complete.")

if __name__ == "__main__":
    train_signals = np.load(os.path.join(data_root, "train_signals.npy"))
    test_signals  = np.load(os.path.join(data_root, "test_signals.npy"))

    # DataLoaders
    train_dataset = TimeSeriesDataset(train_signals)
    test_dataset  = TimeSeriesDataset(test_signals)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)
    train()