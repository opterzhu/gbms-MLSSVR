function output = OptMain(nb,im)
% References  : [1] "An efficient gradient-based model selection algorithm
%                   for multi-output least-squares support vector regression machines",
%                   Pattern Recognition Letters, 2018, doi="10.1016/j.patrec.2018.01.023"
%
% author: Zhu Xinqi (zhuxq3594@gmail.com)
%% Parameter Setting
    switch(nb)
        case 1
            dataname = 'broomcorn';
            load broomcorn;
        case 2
            dataname = 'Synthetic';
            load Synthetic;
        case 3
            dataname = 'CFD100';
            load CFD100;
        case 4
            dataname = 'CFD500';
            load CFD500;
        case 5
            dataname = 'enb';
            load enb;
    end

%     Dim = size(trnX,2) + 2;  
    Dim = 3;
    xmin = -15*ones(Dim,1);
    xmax =  15*ones(Dim,1);
    FEs = 1000;
    func = @(x)(fitness(x));
    %%
%     for im = 1:3
        switch(im)
            case 1
                M = 'GradientDecent';
            case 2
                M = 'PSO';
            case 3
                M = 'GridSearch';
        end
        [output] = hyperpara_opt(M,func,Dim,xmin,xmax,FEs,trnX, trnY, tstX, tstY);
%     end

end

%%
function [output] = hyperpara_opt(M,func,Dim,xmin,xmax,FEs,trnX, trnY, tstX, tstY)
switch(M)
    case 'PSO'
        popsize = 3*Dim;
        tic
        [xopt,fopt,output] = PSO(func,0,xmin,xmax,round(FEs/popsize),popsize,trnX,trnY);
        tt = toc;
        funcCount = output;
    case 'GridSearch'
        tic
        [gamma_best, lambda_best, p_best, fopt,funceval] = GridMLSSVR(trnX, trnY, size(trnX,1));
        tt = toc;
        xopt = [log2(p_best); log2(lambda_best); log2(gamma_best)];
        funcCount = funceval;
    case 'GradientDecent'
        fopt = inf;  xopt = [];
        xmax = 10;    xmin = -10;  x0 = zeros(Dim,1);
        funcCount = 0;
        tic
        for i=1:5
            options = optimoptions(@fminunc,'GradObj','on','Algorithm','trust-region');%,'Diagnostics','on','DerivativeCheck','on');
            [x,fval,exitflag,output,grad] = fminunc(@(x) fitness(x,trnX,trnY),x0,options);
            funcCount = funcCount + output.funcCount;
            if fopt>fval
                fopt = fval;
                xopt = x;
            end
            x0 = (xmax-xmin)*rand(length(x0),1) + repmat(xmin,[length(x0),1]);
        end
        tt = toc;
end
    if length(xopt)==3
        p = 2^xopt(1);
    elseif length(xopt)>3
        p = 2.^xopt(1:size(trnX,2));
    end
    lambda = 2^xopt(end-1);
    gama = 2^xopt(end);
    [alpha, b, ~] = MLSSVRTrain(trnX, trnY, gama, lambda, p);
    [preY, ~, SCC, ARE, MSE] = MLSSVRPredict(tstX, tstY, trnX, alpha, b', lambda, p);
    output = [];
    output.xopt = xopt;  output.fopt = fopt;  output.FE = funcCount;
    output.ARE = ARE;  output.MSE = MSE;  output.SCC = SCC;
    output.tt = tt;
end