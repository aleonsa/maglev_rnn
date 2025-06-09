import torch
import torch.nn as nn

# Recrear arquitectura
class MagLevRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# Cargar pesos
checkpoint = torch.load('maglev_rnn_model.pth', weights_only=False)
model = MagLevRNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Trace y exportar
dummy_input = torch.randn(1, 20, 4)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('maglev_rnn_traced.pt')

print("¡Listo para MATLAB!")