

weights = 15:0.01:35;

% https://int-brain-lab.slack.com/archives/C8Q01DNN4/p1560185305052300
proposal_matteo = 1.5* (1 + 0.05*(weights-25));
proposal_matteo(proposal_matteo < 1) = 1;
proposal_matteo(proposal_matteo > 3) = 3;


proposal_anne = -1 + 0.1*weights;
proposal_anne(proposal_anne < 1) = 1;
proposal_anne(proposal_anne > 3) = 3;

proposal_nate = weights * 0.06;
proposal_nate(proposal_nate < 1) = 1;
proposal_nate(proposal_nate > 3) = 3;

proposal_nate_2 = weights * 0.07;
proposal_nate_2(proposal_nate_2 < 1) = 1;
proposal_nate_2(proposal_nate_2 > 3) = 3;

proposal_nate_3 = weights * 0.05;
proposal_nate_3(proposal_nate_3 < 1) = 1;
proposal_nate_3(proposal_nate_3 > 3) = 3;


% plot
close all;
plot(weights, proposal_matteo, weights, proposal_anne, weights, proposal_nate, ...
    weights, proposal_nate_2, weights, proposal_nate_3);
legend({'Matteo', 'Anne', 'Nate (0.06)', 'Nate (0.07)', 'Nate (0.05)'}, 'location', 'northwest');
legend boxoff;
xlabel('Mouse weight (g)'); ylabel('Reward size (uL)');
grid on;
print(gcf, '-dpdf', '~/Data/Figures_IBL/water_volume_by_weight.pdf');
