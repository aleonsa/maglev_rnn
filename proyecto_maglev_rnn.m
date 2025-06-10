%% LQR TRACKING - IMPLEMENTACIÓN SIMPLIFICADA
clear; clc;

%% 1. PARÁMETROS DEL SISTEMA
m = 0.068; Ke = 6.53e-5; R = 10; L = 0.4125; g = 9.81;
a0 = 0.007; i0 = sqrt((m * g * a0^2) / Ke);

A = [0, 1, 0;
     (Ke*i0^2)/(m*a0^3), 0, -(Ke*i0)/(m*a0^2);
     0, 0, -R/L];
B = [0; 0; 1/L];
C = [1, 0, 0];
% C = eye(3);
D = zeros(size(C,1), size(B,2));

%% 2. DISEÑO LQR
Q = diag([100, 1, 0.1]);
R_lqr = 0.1;
K = lqr(A, B, Q, R_lqr);

[P, ~, ~] = care(A, B, Q, R_lqr);

%% 3. GANANCIAS DE SEGUIMIENTO
M = [A, B; C, D];
N = inv(M) * [zeros(3,1); 1];
Nx = N(1:3);
Nu = N(4);
Kr = Nu + K * Nx;

%% 4. SIMULACIÓN
ref_func = @(t) a0 + 0.004 * sin(2*pi*0.1*t);  % Senoidal ±4mm centrada en a0
tspan = 0:0.01:20;
x0 = [0; 0; 0];

sys_cl = @(t, x) A*x + B*(Kr*(ref_func(t) - a0) - K*x);
[t_sim, x_sim] = ode45(sys_cl, tspan, x0);

ref_sim = arrayfun(ref_func, tspan);
y_sim = (C * x_sim')' + a0;

u_sim = zeros(size(tspan));
for i = 1:length(tspan)
    u_sim(i) = Kr * (ref_sim(i) - a0) - K * x_sim(i,:)';
end

%% 5. GRÁFICAS
figure;
subplot(2,1,1);
plot(tspan, y_sim*1000, 'b', tspan, ref_sim*1000, 'r--', 'LineWidth', 1.5);
ylabel('Posición [mm]'); legend('Salida','Referencia'); grid on;

subplot(2,1,2);
plot(tspan, u_sim, 'k', 'LineWidth', 1.5);
ylabel('Control [V]'); xlabel('Tiempo [s]'); grid on;

%% 6. ERROR RMS
error_rms = sqrt(mean((ref_sim' - y_sim).^2));
fprintf('Error RMS: %.3f mm\n', error_rms*1000);
