clear all; close all; clc;

a = 12;
b = 6;
epsilon = 100; % depth of potential well, V_LJ(r_target) = epsilon
gamma = 100; % force gain

r_target = 390;
r = linspace(320,600,1000);

V_LJ = epsilon*((r_target./r).^a - 2*(r_target./r).^b);
F_LJ = -gamma*epsilon./r .* (a*(r_target./r).^a - 2*b*(r_target./r).^b);

LJ_pot = figure;
font = 20;
plot(r,V_LJ, 'LineWidth', 2)
hold on
plot(r,F_LJ, 'LineWidth', 2)
hold on
scatter(390, 0, 100, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k')
legend('potential', 'force', 'Location', 'SouthEast')
xlabel('distance [mm]')
ylabel('(potential, force) [-]')
xlim([320, 610])
ylim([-600, 200])
xticks([350, r_target, round(r(F_LJ==max(F_LJ))), max(r)])
yticks([-600, round(min(V_LJ)), 0, round(max(F_LJ)), 200])
set(gca,'FontSize', font)
orient(LJ_pot, 'landscape')
grid on
box on
hold off

% SAVE FIGURE
set(LJ_pot, 'PaperPosition', [0 0 10 5]);
set(LJ_pot, 'PaperSize', [10 5]);
saveas(LJ_pot, 'LJ_potential', 'pdf')


