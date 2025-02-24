function SCORE=F1skore(Target,Result)

confusionMat=zeros(4,4);
stejne=zeros(1,length(Target));

for i=1:4
    stejne(Target==Result & Target==i)=1;
    confusionMat(i,i)=sum(stejne);
    stejne=zeros(1,length(Target));
end

for i=1:length(Target)
    switch Target(i)
        case 1
            if Result(i)==2
                confusionMat(1,2)=confusionMat(1,2)+1;
                continue
            elseif Result(i)==3
                confusionMat(1,3)=confusionMat(1,3)+1;
                continue
            elseif Result(i)==4
                confusionMat(1,4)=confusionMat(1,4)+1;
                continue
            end
        case 2
            if Result(i)==1
                confusionMat(2,1)=confusionMat(2,1)+1;
                continue
            elseif Result(i)==3
                confusionMat(2,3)=confusionMat(2,3)+1;
                continue
            elseif Result(i)==4
                confusionMat(2,4)=confusionMat(2,4)+1;
                continue
            end
        case 3
            if Result(i)==1
                confusionMat(3,1)=confusionMat(3,1)+1;
                continue
            elseif Result(i)==2
                confusionMat(3,2)=confusionMat(3,2)+1;
                continue
            elseif Result(i)==4
                confusionMat(3,4)=confusionMat(3,4)+1;
                continue
            end
        case 4
            if Result(i)==1
                confusionMat(4,1)=confusionMat(4,1)+1;
                continue
            elseif Result(i)==2
                confusionMat(4,2)=confusionMat(4,2)+1;
                continue
            elseif Result(i)==3
                confusionMat(4,3)=confusionMat(4,3)+1;
                continue
            end
        end
end

confusionMat=confusionMat';

SeRecall=zeros(1,4);
PPPrecision=zeros(1,4);
F1skorePart=zeros(1,4);
for i=1:4
    TP(i)=confusionMat(i,i);
    FP(i)=sum(confusionMat(i,:))-TP(i);
    FN(i)=sum(confusionMat(:,i))-TP(i);

    if TP(i)==0 & FP(i)==0 & FN(i)==0
        SeRecall(i)=NaN;
        PPPrecision(i)=NaN;
        F1skorePart(i)=1;
    elseif TP(i)==0
        SeRecall(i)=0;
        PPPrecision(i)=0;
        F1skorePart(i)=0;
    else
        SeRecall(i)=TP(i)/(TP(i)+FN(i));
        PPPrecision(i)=TP(i)/(TP(i)+FP(i));
        F1skorePart(i)=(2*PPPrecision(i)*SeRecall(i))/(PPPrecision(i)+SeRecall(i));
    end
end

SCORE=mean(F1skorePart);
f = figure('Units','pixels','Position',[0 0 800 600]);
confusionchart(confusionMat','columnsummary','column-normalized','RowSummary','row-normalized')
end
