data = load("data2.txt");
load prediction
time = data(:,1)/0.0173;
ind = 6:5:5*size(prediction,2)+5;
figure()
plot(time(ind), prediction', '.','markersize', 12)
hold on
plot(time([1,ind]), data([1,ind], 2:8))
xlim([0 120])
ylim([0 0.4])
legend('m=-3','m=-2','m=-1','m=0','m=1','m=2','m=3', 'fontsize', 12)
xlabel('Time (ms)', 'fontsize', 12)
ylabel('P_{ms}', 'fontsize', 12)
set(gca, 'fontsize', 12)
title('(b)', 'fontweight', 'normal')

