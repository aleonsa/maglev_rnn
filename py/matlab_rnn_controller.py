# matlab_rnn_controller.py
# Archivo para ser llamado desde MATLAB

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import sys
import os

# Clase RNN (misma que antes)
class MagLevRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# Controlador global (se inicializa una vez)
class GlobalRNNController:
    def __init__(self):
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = deque(maxlen=20)
        self.seq_len = 20
        self.initialized = False
    
    def initialize(self, model_path='maglev_rnn_model.pth'):
        """Inicializa el modelo (llamar una sola vez)"""
        if self.initialized:
            return
            
        # Cargar modelo
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        self.model = MagLevRNN()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.initialized = True
        print("RNN Controller inicializado exitosamente")
    
    def reset(self):
        """Reinicia el historial (llamar al inicio de simulación)"""
        self.history.clear()
    
    def get_control(self, states, reference):
        """Calcula control usando RNN"""
        if not self.initialized:
            self.initialize()
        
        # Convertir entrada a numpy si es necesario
        if isinstance(states, (list, tuple)):
            states = np.array(states)
        if isinstance(reference, (list, tuple)):
            reference = reference[0] if len(reference) > 0 else reference
        
        # Agregar punto actual al historial
        current_point = np.array([states[0], states[1], states[2], reference])
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
        
        return float(control)

# Instancia global del controlador
controller = GlobalRNNController()

# Funciones para MATLAB
def init_rnn_controller():
    """Inicializa el controlador RNN"""
    controller.initialize()
    return "Initialized"

def reset_rnn_controller():
    """Reinicia el historial del controlador"""
    controller.reset()
    return "Reset"

def get_rnn_control(states, reference):
    """Obtiene control de la RNN"""
    try:
        control = controller.get_control(states, reference)
        return control
    except Exception as e:
        print(f"Error en get_rnn_control: {e}")
        return 0.0

# Función de prueba
def test_controller():
    """Prueba rápida del controlador"""
    init_rnn_controller()
    reset_rnn_controller()
    
    # Simular algunos puntos
    for i in range(25):  # Más que seq_len para probar
        states = [0.001 * i, 0.0, 0.1]
        reference = 0.007 + 0.002 * np.sin(0.1 * i)
        control = get_rnn_control(states, reference)
        print(f"Step {i}: states={states}, ref={reference:.6f}, control={control:.6f}")

if __name__ == "__main__":
    test_controller()