clear all
close all
clc

load pred.mat

testIndex = zeros(370, 2);
score_selected = zeros(370, 4);

for count = 1: 370
    index_selected = textread('score_test_evaluation_included2.txt', '%s', 5 * count);   % txt file containing file names and labels 
    index_tmp = index_selected{(5 * count) - 4 ,1};
    
 
    score_count = pred_after(count, 1)*0 + pred_after(count, 2)*1 + pred_after(count, 3)*2 + pred_after(count, 4)*3;
    
    index_tmp2 = split(index_tmp, '_');
    testIndex(count, 1) = str2double(index_tmp2(1));
    testIndex(count, 2) = score_count;
end

annIndex = unique(testIndex(:,1));
%% load data

filePath_tmp = '.\score_database';
filePath = strcat(filePath_tmp, '\');
folderInfo = dir(strcat(filePath, '*.jpg'));

scoreSum = 0;
patientInfo = zeros(1712,2);

patientScore = zeros(164,2);
annTestScore = zeros(33,3);

for i = 1 : length(folderInfo)
    
    imgName = char(folderInfo(i).name);
    imgElement = split(imgName, '_');
    patientID = str2double(imgElement(1));
    scoreTmp = split(imgElement(3), '.');
    score = str2double(scoreTmp(1));
    patientInfo(i, 1) = patientID;
    patientInfo(i, 2) = score;
end  

for i = 1 : 164
    [m,n] = find(patientInfo(:,1) == i);
    
    patientScore(i, 1) = i;
    patientScore(i, 2) = mean(patientInfo(m(1):m(end),2));  
end

for i = 1 : length(annIndex)
    [m,n] = find(testIndex(:,1) == annIndex(i,1));
    
    annTestScore(i, 1) = annIndex(i,1);
    annTestScore(i, 2) = mean(testIndex(m(1):m(end),2));  
end

severeIndex = [4;7;9;50;52;54;67;95;103;107;133;135;136;137;138;139;140;141;142;143;144;145;146;147;148;149;150;151;152;154;155;156;157;158;159;161;162;163;164];
severeScore = zeros(39,3);

for i = 1 : length(severeIndex)
    severeScore(i, 1:2) = patientScore(severeIndex(i), :);
    severeScore(i, 3) = 1;
end

mildScore = zeros(125,3);
num = 1;

for i = 1 : 164
    tmp = ismember(i, severeIndex);
    if tmp == 1
    else
        mildScore(num, 1:2) = patientScore(i, :);
        mildScore(num, 3) = 0;  
        num = num + 1;
    end
end


for i = 1 : length(annIndex)
    tmp = ismember(annTestScore(i, 1), severeIndex);
    if tmp == 1
        annTestScore(i,3) = 1;
    else
        annTestScore(i,3) = 0;
    end
end

tp = zeros(10,1);
tn = zeros(10,1);
fp = zeros(10,1);
fn = zeros(10,1);
acc = zeros(10,1);
pre = zeros(10,1);
recall = zeros(10,1);
f1score = zeros(10,1);
sensi = zeros(10,1);
spe = zeros(10,1);

for j = 1 : 10

idxSevere = randperm(39)';
idxMild = randperm(125)';

train_mild_idx = idxMild(1:100);
test_mild_idx = idxMild(101:125);
train_severe_idx = idxSevere(1:31);
test_severe_idx = idxSevere(32:39);

train_mild = zeros(100, 3);
train_severe = zeros(31, 3);
test_mild = zeros(25, 3);
test_severe = zeros(8, 3);


for i = 1 : length(train_mild_idx)
    train_mild(i, :) = mildScore(train_mild_idx(i), :);
end

for i = 1 : length(train_severe_idx)
    train_severe(i, :) = severeScore(train_severe_idx(i), :);
end

for i = 1 : length(test_mild_idx)
    test_mild(i, :) = mildScore(test_mild_idx(i), :);
end

for i = 1 : length(test_severe_idx)
    test_severe(i, :) = severeScore(test_severe_idx(i), :);
end



train_data = [train_mild; train_severe];
test_data = [test_mild; test_severe];

tmp1 = train_data';
train_normalized = mapminmax(tmp1(2,:),0,1);
train_normalized = train_normalized';
tmp3 = train_normalized;
train_normalized = [tmp3 tmp3];


tmp2= annTestScore';
test_normalized = mapminmax(tmp2(2,:),0,1);
test_normalized = test_normalized';
tmp4 = test_normalized;
test_normalized = [tmp4 tmp4];

%% SVM train

SVMModel = fitcsvm(train_normalized ,train_data(:,3), 'Standardize',true, 'BoxConstraint',3,'KernelFunction','rbf','KernelScale', 'auto');

[ans_test,~]=predict(SVMModel, test_normalized);

x1 = SVMModel.Alpha;
x2 = SVMModel.Mu;
x3 = SVMModel.Sigma;

for i = 1 : length(ans_test)
    if ans_test(i, 1) == 1 && annTestScore(i,3) == 1
        tn(j,1) = tn(j,1) + 1;
    end
    if ans_test(i, 1) == 1 && annTestScore(i,3) == 0
        fn(j,1) = fn(j,1) + 1;
    end
    if ans_test(i, 1) == 0 && annTestScore(i,3) == 1
        fp(j,1) = fp(j,1) + 1;
    end
    if ans_test(i, 1) == 0 && annTestScore(i,3) == 0
        tp(j,1) = tp(j,1) + 1;
    end
end

pre(j,1) = tp(j,1)/ (tp(j,1) + fp(j,1));
recall(j,1) = tp(j,1) / (tp(j,1) + fn(j,1));

sensi(j,1) = tp(j,1) / (tp(j,1) + fn(j,1));
spe(j,1) = tn(j,1) / (tn(j,1) + fp(j,1));

acc(j,1) = (tp(j,1) + tn(j,1)) / (tp(j,1) + tn(j,1) + fp(j,1) + fn(j,1));
f1score(j,1) = 2 * recall(j,1) * pre(j,1) / (recall(j,1) + pre(j,1));

end

mean_acc = mean(acc);
mean_pre = mean(pre);
mean_recall = mean(recall);
mean_f1 = mean(f1score);
mean_sensi = mean(sensi);
mean_spe = mean(spe);
