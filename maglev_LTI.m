function [ A, B, C, D, u0, x0 ] = maglev_LTI(x10)
    % Linealizacion alrededor de x_eq
    % x1 = posicion de la pelota
    % x2 = velocidad de la pelota
    % x3 = corriente

    % Definición de parámetros
    m = 0.0571; % kg
    g = 9.81; % m/s^2
    Femp1 = 0.017521; % H
    Femp2 = 0.0058231; % m
    fip1 = 0.00014142; % ms
    fip2 = 0.0045626; % m
    ci = 0.0243; % A
    ki = 2.5165; % A
    x3min = 0.03884; % A
    umin = 0.00498;

    % calcule x3, en función de x1
    x30 = sqrt((m*g*Femp2)/Femp1)*exp(x10 / (2*Femp2));

    x0 = [x10, 0, x30]';
    
    % calcule la u0 (valor nominal de u para mantener la pelota en x10)
    u0 = (x30 - ci) / ki;

    fi = fip1/fip2 * exp(-x10/fip2);

    a21 = ((x30^2)/m) * (Femp1/(Femp2^2)) * exp(-x10/Femp2);
    a23 = -(2*x30 / m) * (Femp1/Femp2) * exp(-x10/Femp2);
    a31 = -(ki*u0 + ci - x30) * (1 / (fi*fip2));
    a33 = -1/fi;

    b3 = ki*(1/fi);

    A = [  0, 1,   0;
         a21, 0, a23;
         a31, 0, a33];
    B = [0 0 b3]';

    C = eye(size(A));
    D = zeros(size(C,1), size(B,2));

end