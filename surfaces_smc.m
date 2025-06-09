%% Modelo lineal Maglev
clear; clc; close all;
m = 0.068; Ke = 6.53e-5; R = 10; L = 0.4125; g = 9.81;
a0 = 0.007; i0 = sqrt((m * g * a0^2) / Ke);
A = [0, 1, 0;
 (Ke*i0^2)/(m*a0^3), 0, -(Ke*i0)/(m*a0^2);
 0, 0, -R/L];
B = [0; 0; 1/L];
C = [1, 0, 0];
D = zeros(size(C,1), size(B,2));

%% observador
po = [-100; -120; -140];  % Polos deseados para el observador

L = place(A', C', po)';  % Transponemos A y C porque place trabaja con sistemas controlables

%% SISTEMA AUMENTADO PARA SEGUIMIENTO
% Agregar estado integral del error de salida
% x_aug = [x; xi] donde xi = integral(r - y)
% Sistema aumentado: [x_dot; xi_dot] = [A, 0; -C, 0]*[x; xi] + [B; 0]*u + [0; 1]*r

A_aug = [A, zeros(3,1);
         -C, 0];           % 4x4
B_aug = [B; 0];            % 4x1  
Br_aug = [zeros(3,1); 1];  % 4x1 (entrada de referencia)

% Verificar controlabilidad del sistema aumentado
F_aug = ctrb(A_aug, B_aug);
disp(['Rango sistema aumentado: ', num2str(rank(F_aug))]);

%% Forma canónica de controlador (sistema aumentado)
% 1. pol. car. de A_aug
pcA_aug = charpoly(A_aug);
pcA_aug = flip(pcA_aug); % voltearlo

% construimos matriz triangular (4x4 ahora)
T1_aug = [pcA_aug(2:end);
          pcA_aug(3:end), 0;
          pcA_aug(4:end), 0, 0;
          pcA_aug(5:end), 0, 0, 0];

% la transformacion final es T = F*T1
T_aug = F_aug * T1_aug;

% encontramos el sistema transformado
A_bar_aug = T_aug \ A_aug * T_aug; 
A_bar_aug(abs(A_bar_aug) < 1e-9) = 0;
B_bar_aug = T_aug \ B_aug; 
B_bar_aug(abs(B_bar_aug) < 1e-9) = 0;
F_bar_aug = ctrb(A_bar_aug, B_bar_aug);

%% diseñamos superficie con A-U (sistema aumentado)
eT_aug = [0 0 0 1] / F_bar_aug; 
eT_aug(abs(eT_aug) < 1e-9) = 0;

% polos deseados de la superficie (ahora 3 polos porque es 4x4)
des_poles = [-50, -30, -40];  % Polos más rápidos para buena respuesta
gamma_aug = poly(des_poles);
gamma_A_aug = polyvalm(gamma_aug, A_bar_aug); % gamma(A)

% finalmente, la superficie es
S_bar_aug = eT_aug * gamma_A_aug;

% transformada a coordenadas originales
S_aug = S_bar_aug / T_aug;

% Extraer ganancias
% S_aug = [S_x, S_xi] donde S_x son las ganancias de los estados originales
% y S_xi es la ganancia del estado integral
S_x = S_aug(1:3);    % Ganancias de [posición, velocidad, corriente]
S_xi = S_aug(4);     % Ganancia del estado integral

disp('Superficie de deslizamiento para seguimiento:');
disp(['S_aug = [', num2str(S_aug), ']']);

%% Control equivalente para seguimiento
% Para sistema: x_dot = A*x + B*u + Br*r
% Superficie: sigma = S_x*x + S_xi*xi = 0
% sigma_dot = S_x*(A*x + B*u) + S_xi*(r - C*x) = 0
% Despejando u_eq:
% S_x*B*u_eq = -S_x*A*x - S_xi*(r - C*x)
% u_eq = -(S_x*A*x + S_xi*(r - C*x)) / (S_x*B)
Kue_aug = (S_aug*A_aug) / (S_aug*B_aug);
Kue_x = S_x * A / (S_x * B);      % Ganancia de retroalimentación de estados
Kue_r = S_xi / (S_x * B);         % Ganancia de referencia
Kue_xi = -S_xi * C / (S_x * B);   % Ganancia del estado integral

disp('Ganancias del control equivalente:');
disp(['Kue_aug = [', num2str(Kue_aug), '] (estados)']);

%% LQ-singular 1er Orden
% elijase Q
Q = blkdiag(1e2, 10, 0.1, 1e6);
% encuentre Qbar
Qbar = inv(T_aug)' * Q * inv(T_aug);
% Qbar = blkdiag(1e5, 100, 1e5, 0.1);
Q11 = Qbar(1:3, 1:3);
Q12 = Qbar(1:3, end);
Q21 = Qbar(4, 1:3);
Q22 = Qbar(4, 4);
A11 = A_bar_aug(1:3, 1:3);
A12 = A_bar_aug(1:3, 4);
Qh = Q11 - Q12*inv(Q22)*Q12';
Rh = Q22;
Ah = A11 - A12*inv(Q22)*Q12';
Bh = A12;
[P] = icare(Ah, [], Qh, [], [], [], -Bh*inv(Rh)*Bh');
K1 = inv(Rh)*(Bh'*P + Q12');
S1_bar = [K1, 1];
S_lq = S1_bar / T_aug;
Kue1_bar = S1_bar * A_bar_aug;
Kue_lq = S_lq * A_aug; % / S1*B_aug pero es = 1

% Displays de las ganancias LQ-singular
disp('=== DISEÑO LQ-SINGULAR ===');
disp(['S_lq = [', num2str(S_lq), ']']);
disp(['Kue_lq = [', num2str(Kue_lq), ']']);
