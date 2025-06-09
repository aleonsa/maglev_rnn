function animarMaglev(x_vector, dt)
% ANIMARMAGLEV - Anima el sistema de levitación magnética
% 
% Inputs:
%   x_vector - vector de posiciones de la pelota [cm] (desde techo hacia abajo)
%   dt - tiempo entre frames [s] (opcional, default = 0.05s)
%
% Ejemplo:
%   t = 0:0.1:5;
%   x = 1 + 0.5*sin(2*pi*t);  % oscilación
%   animarMaglev(x, 0.1);

if nargin < 2
    dt = 0.05;  % tiempo por defecto
end

% Parámetros del sistema (en cm)
Tb = 1.4;        % Ball travel (cm)
db = 2.54;       % Ball diameter (cm)
rb = db/2;       % Ball radius (cm)
hp = 10;         % Pedestal height (cm)
techo_ancho = 8; % Ancho del electroimán
techo_alto = 4;  % Alto del electroimán
pedestal_ancho = 3; % Ancho del pedestal

% Configurar coordenadas del sistema
y_pedestal_base = 0;
y_pedestal_top = hp;
y_pelota_min = y_pedestal_top + 2*rb;
y_pelota_max = y_pelota_min + Tb;
y_techo_base = y_pelota_max;
y_techo_top = y_techo_base + techo_alto;
y_origen = y_techo_base;  % x=0 aquí

% Crear la figura
figure('Position', [100, 100, 600, 800]);
hold on; axis equal;

%% 1. DIBUJAR EL TECHO (ELECTROIMÁN)
x_techo = [-techo_ancho/2, techo_ancho/2, techo_ancho/2, -techo_ancho/2, -techo_ancho/2];
y_techo = [y_techo_base, y_techo_base, y_techo_top, y_techo_top, y_techo_base];
fill(x_techo, y_techo, [0.7 0.7 0.7], 'EdgeColor', 'black', 'LineWidth', 2);

% Rayas del electroimán
for i = 1:8
    x_start = -techo_ancho/2 + (i-1)*techo_ancho/8;
    x_end = x_start + techo_ancho/8;
    line([x_start, x_end], [y_techo_top, y_techo_base], 'Color', 'black', 'LineWidth', 1);
end

%% 2. DIBUJAR EL PEDESTAL
x_pedestal = [-pedestal_ancho/2, pedestal_ancho/2, pedestal_ancho/2, -pedestal_ancho/2, -pedestal_ancho/2];
y_pedestal = [y_pedestal_base, y_pedestal_base, y_pedestal_top, y_pedestal_top, y_pedestal_base];
fill(x_pedestal, y_pedestal, [0.8 0.8 0.8], 'EdgeColor', 'black', 'LineWidth', 2);

%% 3. CONFIGURACIÓN INICIAL
xlim([-6, 6]);
ylim([-1, y_techo_top + 1]);
xlabel('Posición Horizontal (cm)', 'FontSize', 12);
ylabel('Posición Vertical (cm)', 'FontSize', 12);
grid on; grid minor;

% Coordenadas del círculo
theta = linspace(0, 2*pi, 100);
x_pelota_coords = rb * cos(theta);

%% 4. ANIMACIÓN
h_pelota = [];
for i = 1:length(x_vector)
    % Posición actual
    x = rb + x_vector(i);  % rb es offset mínimo
    
    % Convertir a coordenadas y
    y_pelota_centro = y_origen - x;
    y_pelota_coords = rb * sin(theta) + y_pelota_centro;
    
    % Borrar pelota anterior
    if ~isempty(h_pelota)
        delete(h_pelota);
    end
    
    % Dibujar nueva pelota
    h_pelota = fill(x_pelota_coords, y_pelota_coords, [0.3 0.3 0.8], ...
                   'EdgeColor', 'blue', 'LineWidth', 2);
    
    % Actualizar título
    title(sprintf('Sistema Maglev - x=%.2fcm (desde techo)', x-rb), ...
          'FontSize', 14, 'FontWeight', 'bold');
    
    % Pausa y actualización
    pause(dt);
    drawnow;
end

hold off;
end