function [matrix,unassign,NetClass,noise] = ProduceMatrix (target,output)
%ProduceMatrix - generates a 6*6 confusion matrix of testData
%
%IN:  trainData:   data of the training examples [N x 45]
%     trainLabel:  labels of the training examples [N x 1]
%     testData:    data of the test examples [N x 45]
%     testLabel:   labels of the test examples [N x 1]
%OUT: matrix: 6*6 confusion matrix [6 x 6]
NetClass=[];
unassign=0;
matrix=zeros(max(target),max(output));
noise=zeros(max(target),1);
s=size(output,1);
for i = 1:s % test each example
    if output(i)>0
    matrix(target(i),output(i))=matrix(target(i),output(i))+1;
    NetClass=[NetClass;[target(i) output(i)]];
    else
        noise(target(i))=noise(target(i))+1;
    end
end
unassign=sum(noise);
% 
