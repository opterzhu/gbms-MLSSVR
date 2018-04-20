function nrunOpt()

% References  : [1] "An efficient gradient-based model selection algorithm
%                   for multi-output least-squares support vector regression machines",
%                   Pattern Recognition Letters, 2018, doi="10.1016/j.patrec.2018.01.023"
%
% author: Zhu Xinqi (zhuxq3594@gmail.com)

clc
clear

for nb=1:5
    for im=1:3
        allFEs = [];
        allxopt = [];
        allfopt = [];
        allARE = [];
        allMSE = [];
        allSCC = [];
        alltime = [];
        if im==1
            nrun = 1;
        else
            nrun = 30;
        end
        for in=1:nrun
            output = OptMain(nb,im);
            allFEs = [allFEs; output.FE];
            allfopt = [allfopt; output.fopt];
            allxopt = [allxopt; output.xopt'];
            allARE = [allARE; output.ARE];
            allMSE = [allMSE; output.MSE];
            allSCC = [allSCC; output.SCC];
            alltime = [alltime; output.tt];
        end
        meanFEs = mean(allFEs,1);
        meanfopt = mean(allfopt,1);
        meanxopt = mean(allxopt,1);
        meanARE = mean(allARE,1);
        meanMSE = mean(allMSE,1);
        meanSCC = mean(allSCC,1);
        meantime = mean(alltime,1);
        [filename,dataname,M] = getfilename(nb,im,nrun);
        f1 = fopen('mean_result.dat','a');
        fprintf(f1,'%s %s\n',dataname,M);
        printVar(f1,meanFEs);
        printVar(f1,meanfopt);
        printVar(f1,meanxopt);
        printVar(f1,meanARE);
        printVar(f1,meanMSE);
        printVar(f1,meanSCC);
        printVar(f1,meantime);
        fclose(f1);
        allRes = [allFEs allfopt allxopt allARE allMSE allSCC alltime];
        save(filename,'allRes','-ascii')
    end
end
end

function [] = printVar(f,data)
n = length(data);
for i=1:n
    fprintf(f,'%f ',data(i));
end
fprintf(f,'\n');
end

function [filename,dataname,M] = getfilename(nb,im,nrun)
        switch(nb)
            case 1
                dataname = 'broomcorn';
            case 2
                dataname = 'Synthetic';
            case 3
                dataname = 'CFD100';
            case 4
                dataname = 'CFD500';
            case 5
                dataname = 'enb';
        end
        switch(im)
            case 1
                M = 'GradientDecent';
            case 2
                M = 'PSO';
            case 3
                M = 'GridSearch';
        end
        filename = strcat(dataname,'_',M,'_',num2str(nrun),'runs.dat');
end