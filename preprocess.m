clc;
clear all;
close all;

k = 370;        % num of test images

gt = zeros(k, 4);
pred = zeros(k, 4);

for count = 1 : k
  gt_selected = textread('gt2.txt', '%s', 4 * count);
  pred_selected = textread('pred2.txt', '%s', 4 * count);
  
  gt(count, 1) = str2double(gt_selected{(4 * count) - 3 ,1});
  gt(count, 2) = str2double(gt_selected{(4 * count) - 2 ,1});
  gt(count, 3) = str2double(gt_selected{(4 * count) - 1 ,1});
  gt(count, 4) = str2double(gt_selected{(4 * count) - 0 ,1});
  
  pred(count, 1) = str2double(pred_selected{(4 * count) - 3 ,1});
  pred(count, 2) = str2double(pred_selected{(4 * count) - 2 ,1});
  pred(count, 3) = str2double(pred_selected{(4 * count) - 1 ,1});
  pred(count, 4) = str2double(pred_selected{(4 * count) - 0 ,1});
end

pred_final = zeros(k, 4);
pred_value = zeros(k, 4);
pred_after = zeros(k, 4);

before_softmax2 = zeros(k, 4);
before_softmax = pred;

for count = 1 : k
    [m, index] = max(pred(count ,:));
    switch index
        case 1
            pred_after(count, :) = [1 0 0 0];
        case 2
            pred_after(count, :) = [0 1 0 0];
        case 3
            pred_after(count, :) = [0 0 1 0];
        case 4
            pred_after(count, :) = [0 0 0 1];
    end
end

out_pred = reshape(pred_after', [1, k * 4]);
out_gt = reshape(gt', [1, k * 4]);
