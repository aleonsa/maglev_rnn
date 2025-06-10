import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 0. LYAP
P = torch.tensor([
    [3.5488e4, 948.0, -213.0],
    [948.0, 25.0, -6.0], 
    [-213.0, -6.0, 1.0]
], dtype=torch.float32) # calculada con el lqr implementado

# Matrices del sistema
m = 0.068
Ke = 6.53e-5
R = 10
L = 0.4125
g = 9.81
a0 = 0.007
i0 = np.sqrt((m * g * a0**2) / Ke)

# Matrices del sistema
A = np.array([
    [0, 1, 0],
    [(Ke*i0**2)/(m*a0**3), 0, -(Ke*i0)/(m*a0**2)],
    [0, 0, -R/L]
])
B = np.array([[0], [0], [1/L]])
C = np.array([[1, 0, 0]])

# 1. CARGAR DATASET
print("Cargando dataset...")
# with open('maglev_rnn_dataset.pkl', 'rb') as f:
with open('maglev_mixed_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# 2. PREPARAR DATOS PARA RNN
def create_sequences(trajectories, controls, references, seq_len=20):
    """Crea secuencias de entrada y salida para la RNN"""
    X, y = [], []
    
    for traj, ctrl, ref in zip(trajectories, controls, references):
        for i in range(seq_len, len(traj)):
            # Entrada: últimos seq_len puntos de [estados + referencia]
            states_seq = traj[i-seq_len:i]  # (seq_len, 3)
            ref_seq = ref[i-seq_len:i].reshape(-1, 1)  # (seq_len, 1)
            input_seq = np.hstack([states_seq, ref_seq])  # (seq_len, 4)
            
            # Salida: control actual
            target = ctrl[i]
            
            X.append(input_seq)
            y.append(target)
    
    return np.array(X), np.array(y)

print("Preparando secuencias...")
X, y = create_sequences(data['trajectories'], data['controls'], data['references'])
print(f"Shape de entrada: {X.shape}")  # (n_samples, seq_len, 4)
print(f"Shape de salida: {y.shape}")   # (n_samples,)

# 3. NORMALIZAR DATOS
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Reshape para normalizar
X_flat = X.reshape(-1, X.shape[-1])
X_normalized = scaler_X.fit_transform(X_flat).reshape(X.shape)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 4. DIVIDIR DATOS
split = int(0.8 * len(X))
X_train, X_test = X_normalized[:split], X_normalized[split:]
y_train, y_test = y_normalized[:split], y_normalized[split:]

# Convertir a tensores
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4.5 derivada de Lyapunov
def lyapunov_derivative(states, controls, P, A, B):
    # states: (batch, 3), controls: (batch,)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    B_tensor = torch.tensor(B.flatten(), dtype=torch.float32)
    
    # x_dot = Ax + Bu
    x_dot = torch.matmul(states, A_tensor.T) + controls.unsqueeze(1) * B_tensor
    
    # dV/dt = 2 * x^T * P * x_dot
    dV_dt = 2 * torch.sum(torch.matmul(states, P) * x_dot, dim=1)
    return dV_dt

# 5. DEFINIR RNN
class MagLevRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Solo última salida temporal
        return out.squeeze()

# 6. ENTRENAR
model = MagLevRNN()
criterion = nn.MSELoss() # agregar lyapunov
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # integrar estabilidad con lyapunov

print("Entrenando...")
losses = []
batch_size = 32
epochs = 50

# Barra de progreso para epochs
epoch_bar = tqdm(range(epochs), desc="Entrenamiento")

for epoch in epoch_bar:
    model.train()
    epoch_loss = 0
    num_batches = len(range(0, len(X_train), batch_size))
    
    # Barra de progreso para batches
    batch_bar = tqdm(range(0, len(X_train), batch_size), 
                     desc=f"Epoch {epoch+1}/{epochs}", 
                     leave=False)
    
    for i in batch_bar:
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        
        loss_mse = criterion(outputs, batch_y)
        # Penalización Lyapunov
        # Desnormalizar los estados para usar con P original
        states_batch = batch_X[:, -1, :3]
        states_orig = scaler_X.inverse_transform(
            torch.cat([states_batch, torch.zeros(states_batch.shape[0], 1)], dim=1).numpy()
            )[:, :3]
        states_batch = torch.tensor(states_orig, dtype=torch.float32)
        dV_dt = lyapunov_derivative(states_batch, outputs, P, A, B)
        lyapunov_penalty = torch.mean(torch.relu(dV_dt + 1e-3))
        loss = loss_mse + 0.001 * lyapunov_penalty  # lambda = 0.001
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Actualizar descripción del batch
        batch_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
    
    # Promedio de la pérdida del epoch
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    
    # Actualizar descripción del epoch
    epoch_bar.set_postfix({'Avg Loss': f'{avg_loss:.6f}'})

print("Entrenamiento completado!")

# 7. EVALUAR
model.eval()
with torch.no_grad():
    # En lugar de todo de una vez, evaluar por lotes
    train_pred = []
    for i in range(0, len(X_train), batch_size):
        batch_pred = model(X_train[i:i+batch_size])
        train_pred.append(batch_pred)
    train_pred = torch.cat(train_pred)
    
    test_pred = []
    for i in range(0, len(X_test), batch_size):
        batch_pred = model(X_test[i:i+batch_size])
        test_pred.append(batch_pred)
    test_pred = torch.cat(test_pred)
    
    train_loss = criterion(train_pred, y_train).item()
    test_loss = criterion(test_pred, y_test).item()

# 8. GRÁFICAS RÁPIDAS
plt.figure(figsize=(12, 4))

# Pérdida durante entrenamiento
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Pérdida de Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# Predicción vs Real (muestra)
plt.subplot(1, 2, 2)
sample_size = 500
y_true = scaler_y.inverse_transform(y_test[:sample_size].numpy().reshape(-1, 1)).flatten()
y_pred = scaler_y.inverse_transform(test_pred[:sample_size].numpy().reshape(-1, 1)).flatten()

plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Control Real (LQR)')
plt.ylabel('Control Predicho (RNN)')
plt.title('RNN vs LQR')
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. GUARDAR MODELO
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'model_config': {'input_size': 4, 'hidden_size': 32, 'num_layers': 2}
}, 'maglev_rnn_model.pth')

print("Modelo guardado como 'maglev_rnn_model.pth'")
print("¡RNN entrenada! Próximo paso: probar en simulación.")