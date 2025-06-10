import numpy as np
import matplotlib.pyplot as plt
import pickle

def analyze_dataset(dataset_path='maglev_mixed_dataset.pkl'):
    """Analiza el dataset para detectar problemas con SMC"""
    
    # Cargar dataset
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    trajectories = data['trajectories']
    controls = data['controls']
    references = data['references']
    controller_types = data['controller_types']
    dt = data['dt']
    a0 = data['system_params']['a0']
    
    # Separar por controlador
    lqr_indices = [i for i, ct in enumerate(controller_types) if ct == 'LQR']
    smc_indices = [i for i, ct in enumerate(controller_types) if ct == 'SMC']
    
    print(f"Total trayectorias: {len(trajectories)}")
    print(f"LQR: {len(lqr_indices)}, SMC: {len(smc_indices)}")
    
    # Analizar controles
    lqr_controls = [controls[i] for i in lqr_indices]
    smc_controls = [controls[i] for i in smc_indices]
    
    all_lqr_u = np.concatenate(lqr_controls)
    all_smc_u = np.concatenate(smc_controls)
    
    print("\n=== ESTADÍSTICAS DE CONTROL ===")
    print(f"LQR - Min: {np.min(all_lqr_u):.2f}, Max: {np.max(all_lqr_u):.2f}, Std: {np.std(all_lqr_u):.2f}")
    print(f"SMC - Min: {np.min(all_smc_u):.2f}, Max: {np.max(all_smc_u):.2f}, Std: {np.std(all_smc_u):.2f}")
    
    # Detectar experimentos "explosivos"
    def analyze_trajectory(traj, ctrl, ref, ctrl_type):
        """Analiza una trayectoria individual"""
        # Error de seguimiento
        y = traj[:, 0] + a0  # Posición absoluta
        tracking_error = np.sqrt(np.mean((ref - y)**2)) * 1000  # RMS en mm
        
        # Control extremo
        max_control = np.max(np.abs(ctrl))
        
        # Oscilaciones (derivada de control)
        control_variation = np.std(np.diff(ctrl))
        
        # Estados extremos
        max_position = np.max(np.abs(traj[:, 0])) * 1000  # mm
        max_velocity = np.max(np.abs(traj[:, 1]))  # m/s
        max_current = np.max(np.abs(traj[:, 2]))   # A
        
        return {
            'tracking_error': tracking_error,
            'max_control': max_control,
            'control_variation': control_variation,
            'max_position': max_position,
            'max_velocity': max_velocity,
            'max_current': max_current,
            'type': ctrl_type
        }
    
    # Analizar todas las trayectorias
    analysis_results = []
    for i, (traj, ctrl, ref, ctrl_type) in enumerate(zip(trajectories, controls, references, controller_types)):
        result = analyze_trajectory(traj, ctrl, ref, ctrl_type)
        result['index'] = i
        analysis_results.append(result)
    
    # Separar resultados por controlador
    lqr_results = [r for r in analysis_results if r['type'] == 'LQR']
    smc_results = [r for r in analysis_results if r['type'] == 'SMC']
    
    # Detectar experimentos problemáticos
    print("\n=== EXPERIMENTOS PROBLEMÁTICOS ===")
    
    # Umbral para detectar problemas
    error_threshold = 5.0  # mm
    control_threshold = 100  # V
    
    problematic_smc = [r for r in smc_results if 
                       r['tracking_error'] > error_threshold or 
                       r['max_control'] > control_threshold]
    
    problematic_lqr = [r for r in lqr_results if 
                       r['tracking_error'] > error_threshold or 
                       r['max_control'] > control_threshold]
    
    print(f"SMC problemáticos: {len(problematic_smc)}/{len(smc_results)}")
    print(f"LQR problemáticos: {len(problematic_lqr)}/{len(lqr_results)}")
    
    if problematic_smc:
        print("\nPeores casos SMC:")
        for r in sorted(problematic_smc, key=lambda x: x['max_control'], reverse=True)[:3]:
            print(f"  Índice {r['index']}: Error {r['tracking_error']:.2f}mm, Control {r['max_control']:.1f}V")
    
    # Visualizar algunos experimentos
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Seleccionar experimentos para visualizar
    experiments_to_plot = []
    
    # 1. Mejor SMC
    best_smc = min(smc_results, key=lambda x: x['tracking_error'])
    experiments_to_plot.append(('Mejor SMC', best_smc['index']))
    
    # 2. Peor SMC
    worst_smc = max(smc_results, key=lambda x: x['max_control'])
    experiments_to_plot.append(('Peor SMC', worst_smc['index']))
    
    # 3. SMC típico (mediano)
    median_smc = sorted(smc_results, key=lambda x: x['tracking_error'])[len(smc_results)//2]
    experiments_to_plot.append(('SMC Típico', median_smc['index']))
    
    for plot_idx, (title, traj_idx) in enumerate(experiments_to_plot):
        traj = trajectories[traj_idx]
        ctrl = controls[traj_idx]
        ref = references[traj_idx]
        ctrl_type = controller_types[traj_idx]
        
        t_span = np.arange(len(traj)) * dt
        y = traj[:, 0] + a0
        
        # Posición
        axes[plot_idx, 0].plot(t_span, ref*1000, 'k--', label='Referencia', linewidth=2)
        axes[plot_idx, 0].plot(t_span, y*1000, 'r-', label=f'{ctrl_type}', linewidth=1.5)
        axes[plot_idx, 0].set_ylabel('Posición [mm]')
        axes[plot_idx, 0].set_title(f'{title} - Seguimiento')
        axes[plot_idx, 0].legend()
        axes[plot_idx, 0].grid(True)
        
        # Control
        axes[plot_idx, 1].plot(t_span, ctrl, 'b-', linewidth=1.5)
        axes[plot_idx, 1].set_ylabel('Control [V]')
        axes[plot_idx, 1].set_title(f'{title} - Control')
        axes[plot_idx, 1].grid(True)
        
        # Estados
        axes[plot_idx, 2].plot(t_span, traj[:, 0]*1000, 'r-', label='Posición', linewidth=1.5)
        axes[plot_idx, 2].plot(t_span, traj[:, 1]*100, 'g-', label='Velocidad×100', linewidth=1.5)
        axes[plot_idx, 2].plot(t_span, traj[:, 2]*10, 'b-', label='Corriente×10', linewidth=1.5)
        axes[plot_idx, 2].set_ylabel('Estados')
        axes[plot_idx, 2].set_xlabel('Tiempo [s]')
        axes[plot_idx, 2].set_title(f'{title} - Estados')
        axes[plot_idx, 2].legend()
        axes[plot_idx, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Gráficas de comparación estadística
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Error de seguimiento
    lqr_errors = [r['tracking_error'] for r in lqr_results]
    smc_errors = [r['tracking_error'] for r in smc_results]
    
    axes[0, 0].hist(lqr_errors, bins=20, alpha=0.7, label='LQR', color='blue')
    axes[0, 0].hist(smc_errors, bins=20, alpha=0.7, label='SMC', color='red')
    axes[0, 0].set_xlabel('Error RMS [mm]')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].set_title('Distribución de Errores de Seguimiento')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Control máximo
    lqr_max_ctrl = [r['max_control'] for r in lqr_results]
    smc_max_ctrl = [r['max_control'] for r in smc_results]
    
    axes[0, 1].hist(lqr_max_ctrl, bins=20, alpha=0.7, label='LQR', color='blue')
    axes[0, 1].hist(smc_max_ctrl, bins=20, alpha=0.7, label='SMC', color='red')
    axes[0, 1].set_xlabel('Control Máximo [V]')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Control Máximo')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Scatter: Error vs Control
    axes[1, 0].scatter(lqr_errors, lqr_max_ctrl, alpha=0.6, label='LQR', color='blue')
    axes[1, 0].scatter(smc_errors, smc_max_ctrl, alpha=0.6, label='SMC', color='red')
    axes[1, 0].set_xlabel('Error RMS [mm]')
    axes[1, 0].set_ylabel('Control Máximo [V]')
    axes[1, 0].set_title('Error vs Control Máximo')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Variación de control
    lqr_var = [r['control_variation'] for r in lqr_results]
    smc_var = [r['control_variation'] for r in smc_results]
    
    axes[1, 1].hist(lqr_var, bins=20, alpha=0.7, label='LQR', color='blue')
    axes[1, 1].hist(smc_var, bins=20, alpha=0.7, label='SMC', color='red')
    axes[1, 1].set_xlabel('Variación de Control')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Variación de Control')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Resumen final
    print("\n=== RESUMEN ESTADÍSTICO ===")
    print(f"Error promedio LQR: {np.mean(lqr_errors):.2f} ± {np.std(lqr_errors):.2f} mm")
    print(f"Error promedio SMC: {np.mean(smc_errors):.2f} ± {np.std(smc_errors):.2f} mm")
    print(f"Control máximo promedio LQR: {np.mean(lqr_max_ctrl):.1f} ± {np.std(lqr_max_ctrl):.1f} V")
    print(f"Control máximo promedio SMC: {np.mean(smc_max_ctrl):.1f} ± {np.std(smc_max_ctrl):.1f} V")
    
    return analysis_results, problematic_smc, problematic_lqr

# Ejecutar análisis
if __name__ == "__main__":
    results, prob_smc, prob_lqr = analyze_dataset()