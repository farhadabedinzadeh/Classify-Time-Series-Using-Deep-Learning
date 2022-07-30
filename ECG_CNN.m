clc;clear;close
%% '================ Written by Farhad AbedinZadeh ================'
%                                                                 %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
%% 

load 'ECGData.mat'
data=ECGData.Data;
label=ECGData.Labels;
for i=1:size(data,1)
trainD(:,:,:,i)=data(i,1:256);
% trainD(:,:,:,i)=data(i,:);

end

targetD=categorical(label);

%% Define Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([size(trainD,1) 1 1]) % 22X1X1 refers to number of features per sample
    convolution2dLayer(3,16,'Padding','same')
    reluLayer
    fullyConnectedLayer(384) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(384) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(3) % 2 refers to number of neurons in next output layer (number of output classes)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',500, ...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainD,targetD',layers,options);

predictedLabels = classify(net,trainD)';

plotconfusion(targetD',predictedLabels)
