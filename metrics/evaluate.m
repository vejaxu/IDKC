function [ matrix,unassign,match,nmiscore,amiscore,accurateScore,NetFmeasure,recall,precision,Fmeasure,indF] = evaluate(class,Tclass)
%EVALUATE Summary of this function goes here
%   Detailed explanation goes here
[matrix,unassign,NetClass,noise] = ProduceMatrix (class,Tclass);
indF=zeros(max(class),1);
if size(matrix,2)~=0
    nmiscore=nmi(NetClass(:,1), NetClass(:,2));
    [accurateScore, ~] = accuracy(NetClass(:,1), NetClass(:,2));
    amiscore=ami(class,Tclass);
    accurateScore=accurateScore/100;
    
    [nFmeasure,~,~,match]=Fmean(matrix);
    NetFmeasure=sum(nFmeasure)/max(class);
    
    
    [Fmeasure,recall,precision]=Fmean2( [matrix noise],match);
    
    %
    [posc,posr]=find(match==1);
    for i=1:size(posc,1)
        indF(posc(i))=Fmeasure(i);
    end
    
    Fmeasure=sum(indF)/max(class);
    recall=sum(recall)/max(class);
    precision=sum(precision)/max(class);
    
else
    match=0;
    nmiscore=0;
    accurateScore=0;
    NetFmeasure=0;
    Fmeasure=0;
    recall=0;
    precision=0;
end

end

