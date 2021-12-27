%emb size
% x = [8 16 32 64 128 256 512 1024];
% train_acc = [0.9679 0.9791 0.9824 0.9891 0.9915 0.9909 0.9920 0.9959];
% test_acc = [0.8743 0.8767 0.8811 0.8848 0.8824 0.8827 0.8841 0.8833];

% filter number
% x = [1 2 3 4 5 6];
% test_acc = [0.8748 0.8827 0.8850 0.8848 0.8842 0.8836];

% filter scale
x = [2 4 6 8 10 12 14];
test_acc = [0.8816 0.8816 0.8834 0.8836 0.8848 0.8812 0.8809];


%semilogx(x, train, '-*')
% plot(x, train_acc, '-*');
% hold on;
% semilogx(x, test_acc, '-*')
plot(x, test_acc, '-*');
xlabel('scale step');
xticks(x);
ylabel('accuracy');
%ylim([0.85,0.97]);
legend('test acc');
saveas(gcf, 'figure/cnn_scale_step.pdf')