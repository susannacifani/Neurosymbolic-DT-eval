import torch.nn as nn
import torch.nn.functional as F

class CNN_grounder(nn.Module):
    def __init__(self, num_symbols):
        super(CNN_grounder, self).__init__()
        
        # Blocco Convoluzionale Moderno
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 16 filtri
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 filtri
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 filtri
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dopo la Flatten, la dimensione è (64 * 8 * 8) = 4096
        # Inseriamo il tuo "128" neuroni qui:
        self.classifier = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_symbols),
            nn.Softmax(dim=-1) # Softmax finale!
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




class Linear_grounder_no_droput(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Linear_grounder_no_droput, self).__init__()
        self.grounder = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.BatchNorm1d(hidden_size), # <-- SCUDO 1
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size), # <-- SCUDO 2
            nn.ReLU(),
            nn.Linear(hidden_size, num_output),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
         return self.grounder(x)

class Linear_grounder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_output):
        super(Linear_grounder, self).__init__()
        self.grounder = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LayerNorm(hidden_size), # <-- SCUDO 1
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), # <-- SCUDO 2
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_output),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
         return self.grounder(x)


