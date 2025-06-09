import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from collections import deque

# Cargar modelo entrenado
print("Cargando modelo RNN...")
checkpoint = torch.load('maglev_rnn_model.pth', weights_only=False)
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Recrear arquitectura RNN
class MagLevRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = MagLevRNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Modelo RNN cargado exitosamente!")

# Parámetros del sistema (mismos de tu código original)
m = 0.068
Ke = 6.53e-5
R = 10
L = 0.4125
g = 9.81
a0 = 0.007
i0 = np.sqrt((m * g * a0**2) / Ke)

A = np.array([
    [0, 1, 0],
    [(Ke*i0**2)/(m*a0**3), 0, -(Ke*i0)/(m*a0**2)],
    [0, 0, -R/L]
])
B = np.array([[0], [0], [1/L]])

# Ganancias LQR (para comparación)
K = np.array([[-5154.64457011539, -137.727154414640, 30.9448066898419]])
Kr = np.array([[-1016.25668098860]])

class RNNController:
    def __init__(self, model, scaler_X, scaler_y, seq_len=20):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.seq_len = seq_len
        self.history = deque(maxlen=seq_len)
        
    def reset(self):
        """Reinicia el historial para nueva simulación"""
        self.history.clear()
        
    def get_control(self, state, reference):
        """Calcula control usando RNN"""
        # Agregar punto actual al historial
        current_point = np.array([state[0], state[1], state[2], reference])
        self.history.append(current_point)
        
        # Si no tenemos suficiente historial, usar control cero
        if len(self.history) < self.seq_len:
            return 0.0
        
        # Crear secuencia de entrada
        input_seq = np.array(list(self.history)).reshape(1, self.seq_len, 4)
        
        # Normalizar
        input_flat = input_seq.reshape(-1, 4)
        input_normalized = self.scaler_X.transform(input_flat).reshape(1, self.seq_len, 4)
        
        # Predicción
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_normalized)
            control_normalized = self.model(input_tensor).item()
            
        # Desnormalizar
        control = self.scaler_y.inverse_transform([[control_normalized]])[0, 0]
        
        return control

def lqr_control(state, reference):
    """Control LQR original"""
    return Kr[0,0] * (reference - a0) - K @ state.reshape(-1, 1)

def system_dynamics(state, control):
    """Dinámicas del sistema"""
    dx = A @ state.reshape(-1, 1) + B * control
    return dx.flatten()

def simulate_system(controller_type, ref_signal, t_span, x0=None):
    """Simula el sistema con controlador especificado"""
    if x0 is None:
        x0 = np.array([0, 0, 0])
    
    dt = t_span[1] - t_span[0]
    n_steps = len(t_span)
    
    x_traj = np.zeros((n_steps, 3))
    u_traj = np.zeros(n_steps)
    x_traj[0] = x0
    
    # Inicializar controlador RNN si es necesario
    if controller_type == 'RNN':
        rnn_controller = RNNController(model, scaler_X, scaler_y)
        rnn_controller.reset()
    
    for i in range(n_steps - 1):
        if controller_type == 'LQR':
            u = lqr_control(x_traj[i], ref_signal[i])
            u_traj[i] = u[0, 0]
        elif controller_type == 'RNN':
            u_traj[i] = rnn_controller.get_control(x_traj[i], ref_signal[i])
        
        # Integrar dinámicas
        dx = system_dynamics(x_traj[i], u_traj[i])
        x_traj[i+1] = x_traj[i] + dt * dx
    
    # Último control
    if controller_type == 'LQR':
        u = lqr_control(x_traj[-1], ref_signal[-1])
        u_traj[-1] = u[0, 0]
    elif controller_type == 'RNN':
        u_traj[-1] = rnn_controller.get_control(x_traj[-1], ref_signal[-1])
    
    return x_traj, u_traj

# SIMULACIONES DE PRUEBA
print("\n=== COMPARANDO RNN vs LQR ===")

# Parámetros de simulación
dt = 0.001
t_span = np.arange(0, 15 + dt, dt)

# Definir señales de referencia de prueba
def test_references():
    tests = []
    
    # Test 1: Escalón
    ref1 = np.ones_like(t_span) * a0
    ref1[t_span >= 5] = a0 + 0.003  # +3mm escalón
    tests.append(("Escalón +3mm", ref1))
    
    # Test 2: Senoidal
    ref2 = a0 + 0.004 * np.sin(2*np.pi*0.1*t_span)
    tests.append(("Senoidal 0.1Hz", ref2))
    
    # Test 3: Escalones múltiples
    ref3 = np.ones_like(t_span) * a0
    ref3[t_span >= 3] = a0 + 0.002
    ref3[t_span >= 6] = a0 - 0.001
    ref3[t_span >= 9] = a0 + 0.003
    ref3[t_span >= 12] = a0
    tests.append(("Escalones múltiples", ref3))
    
    return tests

# Ejecutar pruebas
test_cases = test_references()

fig, axes = plt.subplots(len(test_cases), 2, figsize=(15, 4*len(test_cases)))
if len(test_cases) == 1:
    axes = axes.reshape(1, -1)

for i, (test_name, ref_signal) in enumerate(test_cases):
    print(f"\nPrueba {i+1}: {test_name}")
    
    # Simular LQR
    print("  Simulando LQR...")
    x_lqr, u_lqr = simulate_system('LQR', ref_signal, t_span)
    y_lqr = x_lqr[:, 0] + a0
    
    # Simular RNN
    print("  Simulando RNN...")
    x_rnn, u_rnn = simulate_system('RNN', ref_signal, t_span)
    y_rnn = x_rnn[:, 0] + a0
    
    # Calcular errores
    error_lqr = np.sqrt(np.mean((ref_signal - y_lqr)**2))
    error_rnn = np.sqrt(np.mean((ref_signal - y_rnn)**2))
    
    print(f"  Error RMS LQR: {error_lqr*1000:.3f} mm")
    print(f"  Error RMS RNN: {error_rnn*1000:.3f} mm")
    print(f"  Ratio RNN/LQR: {error_rnn/error_lqr:.3f}")
    
    # Gráficas
    # Posición
    axes[i, 0].plot(t_span, ref_signal*1000, 'k--', linewidth=2, label='Referencia')
    axes[i, 0].plot(t_span, y_lqr*1000, 'b-', linewidth=1.5, label=f'LQR (RMS: {error_lqr*1000:.2f}mm)')
    axes[i, 0].plot(t_span, y_rnn*1000, 'r-', linewidth=1.5, label=f'RNN (RMS: {error_rnn*1000:.2f}mm)')
    axes[i, 0].set_ylabel('Posición [mm]')
    axes[i, 0].set_title(f'{test_name} - Posición')
    axes[i, 0].legend()
    axes[i, 0].grid(True)
    
    # Control
    axes[i, 1].plot(t_span, u_lqr, 'b-', linewidth=1.5, label='LQR')
    axes[i, 1].plot(t_span, u_rnn, 'r-', linewidth=1.5, label='RNN')
    axes[i, 1].set_ylabel('Control [V]')
    axes[i, 1].set_xlabel('Tiempo [s]')
    axes[i, 1].set_title(f'{test_name} - Control')
    axes[i, 1].legend()
    axes[i, 1].grid(True)

plt.tight_layout()
plt.show()

# RESUMEN FINAL
print("\n" + "="*50)
print("RESUMEN DE COMPARACIÓN RNN vs LQR")
print("="*50)

for i, (test_name, ref_signal) in enumerate(test_cases):
    x_lqr, u_lqr = simulate_system('LQR', ref_signal, t_span)
    x_rnn, u_rnn = simulate_system('RNN', ref_signal, t_span)
    
    y_lqr = x_lqr[:, 0] + a0
    y_rnn = x_rnn[:, 0] + a0
    
    error_lqr = np.sqrt(np.mean((ref_signal - y_lqr)**2))
    error_rnn = np.sqrt(np.mean((ref_signal - y_rnn)**2))
    
    print(f"{test_name}:")
    print(f"  LQR: {error_lqr*1000:.3f} mm RMS")
    print(f"  RNN: {error_rnn*1000:.3f} mm RMS")
    print(f"  Mejora: {((error_lqr-error_rnn)/error_lqr)*100:+.1f}%")
    print()

print("🎯 ¡Simulación completada! Analiza los resultados.")