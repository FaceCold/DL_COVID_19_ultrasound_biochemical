close all
clear all
clc

load DrawScatter.mat

tp = zeros(1,10);
tn = zeros(1,10);
fp = zeros(1,10);
fn = zeros(1,10);
acc = zeros(1,10);
pre = zeros(1,10);
recall = zeros(1,10);
f1score = zeros(1,10);

tp2 = 0;
tn2 = 0;
fp2 = 0;
fn2 = 0;
acc2 = 0;
pre2 = 0;
recall2 = 0;
f1score2 = 0;

train_f = P_train(2:126,:); 
train_l = T_train(2:126,:);

test_f = [P_train(1,:); P_test; P_train(127:131, :)];
test_l = [T_train(1,:); T_test; T_train(127:131, :)];

test_f1 = [test_f(3:21,:)];
test_l1 = [test_l(3:21,:)];

data_external = test_f1;
label_external = test_l1;

test_f2 = [test_f(1:2,:); test_f(22:39,:)];
test_l2 = [test_l(1:2,:); test_l(22:39,:)];

data_internal = [train_f(1:5, :); test_f2(1:2, :); train_f(6:125, :); test_f2(3:20, :)];
label_internal = [train_l(1:5, :); test_l2(1:2, :); train_l(6:125, :); test_l2(3:20, :)];

for j = 1: 10


a1 = randperm(138) + 7;
a2 = randperm(7);

train_index = [a2(1: 5) a1(1: 118)];
test_index = [a2(6: 7) a1(119: 138)];

train_data = zeros(123, 6);
train_label = zeros(123,1);

test_data = zeros(22, 6);
test_label = zeros(22,1);

num_train = 1;
num_test = 1;

for i = 1: length(data_internal')
    if ismember(i, train_index)
        train_data(num_train, :) = data_internal(i, :);
        train_label(num_train, :) = label_internal(i, :);
        num_train = num_train + 1;
    else
        test_data(num_test,:) = data_internal(i, :);
        test_label(num_test, :) = label_internal(i, :);
        num_test = num_test + 1;
    end
end

mdl = ClassificationKNN.fit(train_data, train_label, 'NumNeighbors', 2);
pred1 = predict(mdl, test_data);


for i = 1: length(pred1)
    if pred1(i, 1) == 0 && test_label(i, 1) == 0
        tp(:, j) = tp(:, j) + 1;
    end
    if pred1(i, 1) == 0 && test_label(i, 1) == 1
        fp(:, j) = fp(:, j) + 1;
    end
    if pred1(i, 1) == 1 && test_label(i, 1) == 0
        fn(:, j) = fn(:, j) + 1;
    end
    if pred1(i, 1) == 1 && test_label(i, 1) == 1
        tn(:, j) = tn(:, j) + 1;
    end
end


pre(:, j) = tp(:, j)/ (tp(:, j) + fp(:, j));
recall(:, j) = tp(:, j) / (tp(:, j) + fn(:, j));

% sensi(:,1) = tp(:,1) / (tp(:,1) + fn(:,1));
% spe(:,1) = tn(:,1) / (tn(:,1) + fp(:,1));

acc(:, j) = (tp(:, j) + tn(:, j)) / (tp(:, j) + tn(:, j) + fp(:, j) + fn(:, j));
f1score(:, j) = 2 * recall(:, j) * pre(:, j) / (recall(:, j) + pre(:, j));

end


acc_mean = mean(acc);
pre_mean = mean(pre);
recall_mean = mean(recall);
f1score_mean = mean(f1score);

pred2 = predict(mdl, test_f1);

for i = 1: length(pred2)
    if pred2(i, 1) == 0 && test_l1(i, 1) == 0
        tp2 = tp2 + 1;
    end
    if pred2(i, 1) == 0 && test_l1(i, 1) == 1
        fp2 = fp2 + 1;
    end
    if pred2(i, 1) == 1 && test_l1(i, 1) == 0
        fn2 = fn2 + 1;
    end
    if pred2(i, 1) == 1 && test_l1(i, 1) == 1
        tn2 = tn2 + 1;
    end
end


pre2 = tp2 / (tp2 + fp2);
recall2 = tp2 / (tp2 + fn2);

% sensi2 = tp2 / (tp2 + fn2);
% spe2 = tn2 / (tn2 + fp2);

acc2 = (tp2 + tn2) / (tp2 + tn2 + fp2 + fn2);
f1score2 = 2 * recall2 * pre2 / (recall2 + pre2);


