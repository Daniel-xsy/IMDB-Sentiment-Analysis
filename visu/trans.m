%embedding size
x = [1 2 3 4 5 6 7 8];
train_loss = [0.702101 0.639835 0.349782 0.262969 0.218488 0.156562 0.144558 0.139117];
train_acc = [0.5044 0.5997 0.8553 0.8993 0.9206 0.9508 0.9541 0.9563];
test_loss = [0.697687 0.440276 0.326035 0.317794 0.343168 0.379830 0.397803 0.418404];
test_acc = [0.5000 0.8072 0.8608 0.8696 0.8650 0.8644 0.8605 0.8568];
%semilogx(x, train, '-*')
plot(x, train_acc, '-*');
hold on;
%semilogx(x, test, '-*')
plot(x, test_acc, '-*');
xlabel('epoch');
xticks(x);
ylabel('accuracy');
%ylim([0.85,0.97]);
legend('train acc','test acc');
saveas(gcf, 'figure/trans_train_acc.pdf')