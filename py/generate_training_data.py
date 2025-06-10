import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class MagLevDatasetGenerator:
    def __init__(self):
        # Parámetros del sistema
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
        
        # Ganancias LQR
        self.K = np.array([[-5154.64457011539, -137.727154414640, 30.9448066898419]])
        self.Kr = np.array([[-1016.25668098860]])
        
        # Parámetros Super-Twisting
        self.S_aug = 0.1*np.array([-874.09, -41.846, 0.4125, 4169.2])
        self.Kue_aug = 0.1*np.array([-62813, -874.09, 570.22, 0])
        self.L_smc = 500
        
        self.dt = 0.001
        
    def csign(self, x, m):
        """Función Csign para super-twisting"""
        if abs(x) < 1e-12:
            return 0.0
        return np.abs(x)**m * np.sign(x)
    
    def generate_reference_signals(self, t_span, ref_type, params=None):
        """Genera señales de referencia"""
        if ref_type == 'step':
            amp = params.get('amplitude', 0.002)
            step_time = params.get('step_time', 5.0)
            ref = np.ones_like(t_span) * self.a0
            ref[t_span >= step_time] = self.a0 + amp
            
        elif ref_type == 'sine':
            freq = params.get('frequency', 0.1)
            amp = params.get('amplitude', 0.004)
            ref = self.a0 + amp * np.sin(2*np.pi*freq*t_span)
            
        elif ref_type == 'square':
            freq = params.get('frequency', 0.05)
            amp = params.get('amplitude', 0.003)
            ref = self.a0 + amp * np.sign(np.sin(2*np.pi*freq*t_span))
            
        elif ref_type == 'multistep':
            ref = np.ones_like(t_span) * self.a0
            steps = params.get('steps', [2, 6, 10, 14])
            amplitudes = params.get('amplitudes', [0.002, -0.001, 0.003, -0.002])
            for step_time, amp in zip(steps, amplitudes):
                ref[t_span >= step_time] = self.a0 + amp
                
        elif ref_type == 'random_steps':
            n_steps = params.get('n_steps', 8)
            max_amp = params.get('max_amplitude', 0.004)
            ref = np.ones_like(t_span) * self.a0
            step_times = np.sort(np.random.uniform(1, t_span[-1]-1, n_steps))
            amplitudes = np.random.uniform(-max_amp, max_amp, n_steps)
            for step_time, amp in zip(step_times, amplitudes):
                ref[t_span >= step_time] = self.a0 + amp
        
        return ref
    
    def simulate_lqr(self, t_span, ref_signal, x0, add_noise=False):
        """Simula con controlador LQR"""
        n_steps = len(t_span)
        x_traj = np.zeros((n_steps, 3))
        u_traj = np.zeros(n_steps)
        x_traj[0] = x0
        
        noise = 0.0001 if add_noise else 0.0
        
        for i in range(n_steps - 1):
            u = self.Kr[0,0] * (ref_signal[i] - self.a0) - self.K @ x_traj[i]
            dx = self.A @ x_traj[i] + self.B.flatten() * u[0]
            x_traj[i+1] = x_traj[i] + self.dt * dx
            
            if add_noise:
                x_traj[i+1] += np.random.normal(0, noise, 3)
            
            u_traj[i] = u[0]
        
        # Último control
        u_traj[-1] = self.Kr[0,0] * (ref_signal[-1] - self.a0) - self.K @ x_traj[-1]
        return x_traj, u_traj
    
    def simulate_supertwisting(self, t_span, ref_signal, x0, add_noise=False):
        """Simula con Super-Twisting"""
        n_steps = len(t_span)
        x_traj = np.zeros((n_steps, 3))
        u_traj = np.zeros(n_steps)
        x_traj[0] = x0
        
        # Estados adicionales del SMC
        z_state = 0.0
        xi = 0.0
        
        noise = 0.00001 if add_noise else 0.0
        
        for i in range(n_steps - 1):
            # Estados aumentados
            x_aug = np.array([x_traj[i, 0], x_traj[i, 1], x_traj[i, 2], xi])
            
            # Superficie deslizante
            s = self.S_aug @ x_aug
            
            # Control
            ueqp = -self.Kue_aug @ x_aug
            alpha = 1.5 * np.sqrt(self.L_smc)
            beta = 1.1 * self.L_smc
            u = -alpha * self.csign(s, 0.5) + z_state + ueqp
            
            # Dinámicas
            dx = self.A @ x_traj[i] + self.B.flatten() * u
            z_dot = -beta * np.sign(s)
            
            # Integrar
            x_traj[i+1] = x_traj[i] + self.dt * dx
            z_state = z_state + self.dt * z_dot
            
            # Estado integral
            y = x_traj[i, 0] + self.a0
            error = ref_signal[i] - y
            xi = xi + self.dt * error
            
            if add_noise:
                x_traj[i+1] += np.random.normal(0, noise, 3)
            
            u_traj[i] = u
        
        # Último control
        x_aug = np.array([x_traj[-1, 0], x_traj[-1, 1], x_traj[-1, 2], xi])
        s = self.S_aug @ x_aug
        ueqp = -self.Kue_aug @ x_aug
        u_traj[-1] = -1.5 * np.sqrt(self.L_smc) * self.csign(s, 0.5) + z_state + ueqp
        
        return x_traj, u_traj
    
    def generate_dataset(self, n_experiments=100, trajectory_length=20, save_path='maglev_mixed_dataset.pkl'):
        """Genera dataset con ambos controladores para los mismos experimentos"""
        
        print(f"Generando {n_experiments} experimentos con ambos controladores...")
        
        all_trajectories = []
        all_controls = []
        all_references = []
        all_controller_types = []
        
        # Tipos de referencias
        ref_configs = [
            ('step', {'amplitude': 0.002}),
            ('step', {'amplitude': 0.004}),
            ('sine', {'frequency': 0.1, 'amplitude': 0.003}),
            ('sine', {'frequency': 0.05, 'amplitude': 0.004}),
            ('square', {'frequency': 0.05, 'amplitude': 0.003}),
            ('multistep', {}),
            ('random_steps', {'n_steps': 6})
        ]
        
        for i in tqdm(range(n_experiments)):
            # Definir experimento
            t_span = np.arange(0, trajectory_length + self.dt, self.dt)
            ref_type, params = ref_configs[i % len(ref_configs)]
            
            # Variar parámetros
            if ref_type == 'sine':
                params = params.copy()
                params['frequency'] *= np.random.uniform(0.7, 1.3)
                params['amplitude'] *= np.random.uniform(0.8, 1.2)
            elif ref_type == 'step':
                params = params.copy()
                params['amplitude'] *= np.random.uniform(0.8, 1.2)
                params['step_time'] = np.random.uniform(3, 8)
            
            # Generar referencia y condición inicial (iguales para ambos)
            ref_signal = self.generate_reference_signals(t_span, ref_type, params)
            x0 = np.random.normal(0, 0.0005, 3)
            add_noise = i > n_experiments // 2
            
            # Simular con LQR
            x_lqr, u_lqr = self.simulate_lqr(t_span, ref_signal, x0, add_noise)
            all_trajectories.append(x_lqr)
            all_controls.append(u_lqr)
            all_references.append(ref_signal)
            all_controller_types.append('LQR')
            
            # Simular con Super-Twisting (mismo experimento)
            x_smc, u_smc = self.simulate_supertwisting(t_span, ref_signal, x0, add_noise)
            all_trajectories.append(x_smc)
            all_controls.append(u_smc)
            all_references.append(ref_signal)  # Misma referencia
            all_controller_types.append('SMC')
        
        # Crear dataset
        dataset = {
            'trajectories': all_trajectories,
            'controls': all_controls,
            'references': all_references,
            'controller_types': all_controller_types,
            'dt': self.dt,
            'system_params': {
                'A': self.A, 'B': self.B, 'K': self.K, 'Kr': self.Kr, 'a0': self.a0
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset guardado: {save_path}")
        print(f"Total: {len(all_trajectories)} trayectorias ({n_experiments} experimentos × 2 controladores)")
        
        return dataset

# Uso
if __name__ == "__main__":
    generator = MagLevDatasetGenerator()
    dataset = generator.generate_dataset(
        n_experiments=100,
        trajectory_length=20,
        save_path='maglev_mixed_dataset.pkl'
    )