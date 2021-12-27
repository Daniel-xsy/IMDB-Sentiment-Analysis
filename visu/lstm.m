%dropout
x = linspace(0,0.9,10);
train = [0.9623 0.9696 0.9652 0.9589 0.9576 0.9544 0.9242 0.9356 0.9204 0.8918];
test =  [0.8704 0.8780 0.8825 0.8844 0.8844 0.8868 0.8865 0.8826 0.8826 0.8584];

%embedding size
% x = [32 64 128 256 512];
% train = [0.9127 0.9306 0.9385 0.9582 0.9668];
% test = [0.8707 0.8832 0.8871 0.8825 0.8838];
%semilogx(x, train, '-*')
plot(x, train, '-*');
hold on;
%semilogx(x, test, '-*')
plot(x, test, '-*');
xlabel('dropout rate');
xticks(x);
ylabel('accuracy');
ylim([0.85,0.97]);
legend('train acc','test acc');
saveas(gcf, 'figure/dp.pdf')