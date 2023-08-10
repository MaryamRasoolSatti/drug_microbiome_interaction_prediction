#%%********************************OPTUNA gincovnet************************************
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool

class GINConvNet(torch.nn.Module):
    def __init__(self, num_features=114, conv_hidden_sizes=[256], linear_hidden_sizes=[256], dropout=0.1):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Create convolutional layers based on conv_hidden_sizes
        self.conv_layers = nn.ModuleList()
        in_channels = num_features
        for out_channels in conv_hidden_sizes:
            self.conv_layers.append(Sequential(Linear(in_channels, out_channels), ReLU(), Linear(out_channels, out_channels)))
            in_channels = out_channels

        # Combined layers
        self.fc_adjust = nn.Linear(in_channels, 95)  # Modify the input size of fc_adjust
        linear_hidden_sizes.sort(reverse=True)  # Sort the linear_hidden_sizes in descending order
        self.linear_layers = nn.ModuleList()
        in_dim = 95
        for out_dim in linear_hidden_sizes:
            self.linear_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        self.out = nn.Linear(in_dim, 2)  # Output layer with 2 units for binary classification

    def forward(self, data):
        x1, edge_index_1, batch1 = data.x.float(), data.edge_index, data.batch

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x1 = F.relu(conv_layer(x1))
            x1 = self.dropout(x1)

        x1 = global_add_pool(x1, batch1)

        # Apply fc_adjust layer
        x1 = self.fc_adjust(x1)

        # Apply linear layers with dropout and ReLU activation
        for linear_layer in self.linear_layers:
            x1 = linear_layer(x1)
            x1 = self.relu(x1)
            x1 = self.dropout(x1)

        out = self.out(x1)
        return F.softmax(out, dim=1)  # Softmax activation for binary classification
#%%
import optuna
import torch.optim as optim

def objective(trial):
    # Define the hyperparameter search space
    num_features = 114
    conv_hidden_sizes = [trial.suggest_int('conv_hidden_size_{}'.format(i), 114, 256) for i in range(trial.suggest_int('num_conv_layers', 1, 5))]
    conv_hidden_sizes.sort(reverse=True)  # Sort the convolutional filter sizes in descending order
    print("conv hidden sizes:", conv_hidden_sizes)  # Add this line to observe the sorted linear hidden sizes

    num_linear_layers = trial.suggest_int('num_linear_layers', 1, 5)
    linear_hidden_sizes = [trial.suggest_int('linear_hidden_size_{}'.format(i), 8, 128) for i in range(num_linear_layers)]
    linear_hidden_sizes.sort(reverse=True)  # Sort the convolutional filter sizes in descending order
    print("Linear hidden sizes:", linear_hidden_sizes)  # Add this line to observe the sorted linear hidden sizes

    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)  # Add weight decay as a hyperparameter

    # Create the model
    model = GINConvNet(num_features=num_features, conv_hidden_sizes=conv_hidden_sizes,
                       linear_hidden_sizes=linear_hidden_sizes, dropout=dropout)
    model.to(device)
    # Define the optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for data in train_loader:
            data = data.to(device)
            y = data.y.view(-1, 1).long().to(device)
            y = y.squeeze(1)
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # Calculate validation accuracy
    y_true = []
    y_pred = []
    for data in val_loader:
        data = data.to(device)
        y = data.y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        output = model(data)
        preds = output.argmax(dim=1)

        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    val_accuracy = accuracy_score(y_true, y_pred)

    return val_accuracy
    
#%%
# Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
best_trial = study.best_trial
print("Number of finished trials:", len(study.trials))
print("Best trial:")
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
