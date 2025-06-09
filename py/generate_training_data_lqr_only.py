import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

class MagLevDatasetGenerator:
    def __init__(self):
        # Parámetros del sistema (mismos que tu código)
        self.m = 0.068
        self.Ke = 6.53e-5
        self.R = 10
        self.L = 0.4125
        self.g = 9.81
        self.a0 = 0.007
        self.i0 = np.sqrt((self.m * self.g * self.a0**2) / self.Ke)
        
        # Matrices del sistema
        self.A = np.array([
            [0, 1, 0],
            [(self.Ke*self.i0**2)/(self.m*self.a0**3), 0, -(self.Ke*self.i0)/(self.m*self.a0**2)],
            [0, 0, -self.R/self.L]
        ])
        self.B = np.array([[0], [0], [1/self.L]])
        self.C = np.array([[1, 0, 0]])
        
        # Ganancias LQR (mismas que funcionaron)
        self.K = np.array([[-5154.64457011539, -137.727154414640, 30.9448066898419]])
        self.Kr = np.array([[-1016.25668098860]])
        
        # Parámetros de simulación
        self.dt = 0.001
        
    def generate_reference_signals(self, t_span, ref_type, params=None):
        """Genera diferentes tipos de señales de referencia"""
        if ref_type == 'step':
            # Escalón con diferentes amplitudes
            amp = params.get('amplitude', 0.002)  # ±2mm por defecto
            step_time = params.get('step_time', 5.0)
            ref = np.ones_like(t_span) * self.a0
            ref[t_span >= step_time] = self.a0 + amp
            
        elif ref_type == 'sine':
            # Senoidal (como tu código original)
            freq = params.get('frequency', 0.1)
            amp = params.get('amplitude', 0.004)
            ref = self.a0 + amp * np.sin(2*np.pi*freq*t_span)
            
        elif ref_type == 'square':
            # Onda cuadrada
            freq = params.get('frequency', 0.05)
            amp = params.get('amplitude', 0.003)
            ref = self.a0 + amp * np.sign(np.sin(2*np.pi*freq*t_span))
            
        elif ref_type == 'multistep':
            # Múltiples escalones
            ref = np.ones_like(t_span) * self.a0
            steps = params.get('steps', [2, 6, 10, 14])  # tiempos de escalón
            amplitudes = params.get('amplitudes', [0.002, -0.001, 0.003, -0.002])
            
            for step_time, amp in zip(steps, amplitudes):
                ref[t_span >= step_time] = self.a0 + amp
                
        elif ref_type == 'chirp':
            # Frecuencia variable (chirp)
            f0 = params.get('f_start', 0.01)
            f1 = params.get('f_end', 0.2)
            amp = params.get('amplitude', 0.002)
            t_end = t_span[-1]
            ref = self.a0 + amp * np.sin(2*np.pi*(f0 + (f1-f0)*t_span/(2*t_end))*t_span)
            
        elif ref_type == 'random_steps':
            # Escalones aleatorios
            n_steps = params.get('n_steps', 8)
            max_amp = params.get('max_amplitude', 0.004)
            
            ref = np.ones_like(t_span) * self.a0
            step_times = np.sort(np.random.uniform(1, t_span[-1]-1, n_steps))
            amplitudes = np.random.uniform(-max_amp, max_amp, n_steps)
            
            for step_time, amp in zip(step_times, amplitudes):
                ref[t_span >= step_time] = self.a0 + amp
        
        return ref
    
    def system_dynamics(self, t, x, ref_val):
        """Dinámicas del sistema con LQR"""
        u = self.Kr[0,0] * (ref_val - self.a0) - self.K @ x.reshape(-1, 1)
        dx = self.A @ x.reshape(-1, 1) + self.B * u[0,0]
        return dx.flatten(), u[0,0]
    
    def simulate_trajectory(self, t_span, ref_signal, x0=None, add_noise=False):
        """Simula una trayectoria completa"""
        if x0 is None:
            x0 = np.array([0, 0, 0])
            
        n_steps = len(t_span)
        x_traj = np.zeros((n_steps, 3))
        u_traj = np.zeros(n_steps)
        x_traj[0] = x0
        
        # Ruido para robustez (opcional)
        process_noise = 0.0001 if add_noise else 0.0
        
        for i in range(n_steps - 1):
            dx, u = self.system_dynamics(t_span[i], x_traj[i], ref_signal[i])
            x_traj[i+1] = x_traj[i] + self.dt * dx
            
            # Agregar ruido de proceso si se especifica
            if add_noise:
                x_traj[i+1] += np.random.normal(0, process_noise, 3)
            
            u_traj[i] = u
        
        # Último control
        _, u_traj[-1] = self.system_dynamics(t_span[-1], x_traj[-1], ref_signal[-1])
        
        return x_traj, u_traj
    
    def generate_dataset(self, n_trajectories=100, trajectory_length=20, save_path='maglev_dataset.pkl'):
        """Genera el dataset completo para entrenar la RNN"""
        
        print(f"Generando {n_trajectories} trayectorias de {trajectory_length}s cada una...")
        
        all_trajectories = []
        all_controls = []
        all_references = []
        
        # Definir tipos de referencias y sus parámetros
        ref_configs = [
            ('step', {'amplitude': 0.002}),
            ('step', {'amplitude': 0.004}),
            ('step', {'amplitude': -0.002}),
            ('sine', {'frequency': 0.1, 'amplitude': 0.003}),
            ('sine', {'frequency': 0.05, 'amplitude': 0.004}),
            ('sine', {'frequency': 0.2, 'amplitude': 0.002}),
            ('square', {'frequency': 0.05, 'amplitude': 0.003}),
            ('multistep', {}),
            ('chirp', {'amplitude': 0.002}),
            ('random_steps', {'n_steps': 6})
        ]
        
        for i in tqdm(range(n_trajectories)):
            # Tiempo de simulación
            t_span = np.arange(0, trajectory_length + self.dt, self.dt)
            
            # Seleccionar tipo de referencia aleatoriamente
            ref_type, params = ref_configs[i % len(ref_configs)]
            
            # Variar parámetros ligeramente
            if ref_type == 'sine':
                params['frequency'] *= np.random.uniform(0.7, 1.3)
                params['amplitude'] *= np.random.uniform(0.8, 1.2)
            elif ref_type == 'step':
                params['amplitude'] *= np.random.uniform(0.8, 1.2)
                params['step_time'] = np.random.uniform(3, 8)
            
            # Generar referencia
            ref_signal = self.generate_reference_signals(t_span, ref_type, params)
            
            # Condición inicial aleatoria (pequeña)
            x0 = np.random.normal(0, 0.0005, 3)
            
            # Simular trayectoria
            add_noise = i > n_trajectories // 2  # Mitad con ruido, mitad sin ruido
            x_traj, u_traj = self.simulate_trajectory(t_span, ref_signal, x0, add_noise)
            
            all_trajectories.append(x_traj)
            all_controls.append(u_traj)
            all_references.append(ref_signal)
        
        # Crear dataset
        dataset = {
            'trajectories': all_trajectories,  # Estados [posición, velocidad, corriente]
            'controls': all_controls,          # Acciones de control
            'references': all_references,      # Señales de referencia
            'dt': self.dt,
            'system_params': {
                'A': self.A, 'B': self.B, 'C': self.C,
                'K': self.K, 'Kr': self.Kr, 'a0': self.a0
            }
        }
        
        # Guardar dataset
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset guardado en: {save_path}")
        print(f"Total de trayectorias: {len(all_trajectories)}")
        print(f"Longitud promedio: {np.mean([len(traj) for traj in all_trajectories]):.0f} puntos")
        
        return dataset
    
    def visualize_sample_trajectories(self, dataset, n_samples=4):
        """Visualiza algunas trayectorias del dataset"""
        trajectories = dataset['trajectories']
        controls = dataset['controls']
        references = dataset['references']
        dt = dataset['dt']
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            idx = np.random.randint(0, len(trajectories))
            
            x_traj = trajectories[idx]
            u_traj = controls[idx]
            ref_traj = references[idx]
            t_span = np.arange(len(x_traj)) * dt
            
            # Posición vs referencia
            axes[i, 0].plot(t_span, (x_traj[:, 0] + self.a0)*1000, 'b-', label='Posición')
            axes[i, 0].plot(t_span, ref_traj*1000, 'r--', label='Referencia')
            axes[i, 0].set_ylabel('Posición [mm]')
            axes[i, 0].set_title(f'Trayectoria {idx+1}')
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            
            # Control
            axes[i, 1].plot(t_span, u_traj, 'k-', label='Control')
            axes[i, 1].set_ylabel('Control [V]')
            axes[i, 1].set_xlabel('Tiempo [s]')
            axes[i, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear generador
    generator = MagLevDatasetGenerator()
    
    # Generar dataset
    print("=== GENERANDO DATASET PARA RNN ===")
    dataset = generator.generate_dataset(
        n_trajectories=200,      # Número de trayectorias
        trajectory_length=20,    # Duración en segundos
        save_path='maglev_rnn_dataset.pkl'
    )
    
    # Visualizar algunas muestras
    print("\n=== VISUALIZANDO MUESTRAS ===")
    generator.visualize_sample_trajectories(dataset, n_samples=3)
    
    print("\n=== ESTADÍSTICAS DEL DATASET ===")
    trajectories = dataset['trajectories']
    controls = dataset['controls']
    
    # Estadísticas de los estados
    all_states = np.vstack(trajectories)
    print(f"Estados - Posición: [{np.min(all_states[:,0]*1000):.1f}, {np.max(all_states[:,0]*1000):.1f}] mm")
    print(f"Estados - Velocidad: [{np.min(all_states[:,1]):.3f}, {np.max(all_states[:,1]):.3f}] m/s")
    print(f"Estados - Corriente: [{np.min(all_states[:,2]):.3f}, {np.max(all_states[:,2]):.3f}] A")
    
    # Estadísticas del control
    all_controls = np.concatenate(controls)
    print(f"Control: [{np.min(all_controls):.1f}, {np.max(all_controls):.1f}] V")
    
    print(f"\nDataset listo para entrenar la RNN!")