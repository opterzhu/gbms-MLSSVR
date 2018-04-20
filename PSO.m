function [xmin,fmin,stopeval,besthistory] = PSO(fname,stopfitness,VarMin,VarMax,maxit,popsize,trnX,trnY)
% Particle Swarm Optimization - PSO
% Haupt & Haupt
% 2003
ff = fname;                     % Objective Function
%% Initializing variables
% popsize = 10;                            % Size of the swarm
npar = length(VarMin);                                % Dimension of the problem
% maxit = 100;                             % Maximum number of iterations
c1 = 1;                                  % cognitive parameter
c2 = 4-c1;                               % social parameter
C=1;                                     % constriction factor
velmax = 5;
%% Initializing swarm and velocities
par = rand(npar,popsize);                  % random population of continuous values
cost = zeros(1,popsize);
for ii = 1:popsize
    par(:,ii) = unifrnd(VarMin,VarMax,[npar 1]);
    cost(ii) = fitness(par(:,ii),trnX,trnY);      % calculates population cost using ff
end
vel = 5*rand(npar,popsize);                % random velocities
minc = zeros(1,maxit);
meanc = zeros(1,maxit);
% minc(1)=min(cost);                       % min cost
% meanc(1)=mean(cost);                     % mean cost
globalmin=min(cost);                       % initialize global minimum
% Initialize local minimum for each particle
localpar = par;                          % location of local minima
localcost = cost;                        % cost of local minima
% Finding best particle in initial population
[globalcost,indx] = min(cost);
globalpar=par(:,indx);
besthistory = [];
%% Start iterations
iter = 0; % counter
while iter < maxit
    iter = iter + 1;
% update velocity = vel
    w=(maxit-iter)/maxit; %inertia weiindxht
    r1 = rand(npar,popsize); % random numbers
    r2 = rand(npar,popsize); % random numbers
    vel = C*(w*vel + c1 *r1.*(localpar-par) + c2*r2.*(globalpar*ones(1,popsize)-par));
    temp1 = vel>velmax;
    vel = vel.*not(temp1) + velmax*temp1;
    temp1 = vel<-velmax;
    vel = vel.*not(temp1) - velmax*temp1;
% update particle positions
    par = par + vel; % updates particle position
    for ii = 1:popsize
        par(:,ii) = min(max(par(:,ii),VarMin),VarMax);
    end
%     overlimit=par<=1;
%     underlimit=par>=0;
%     par=par.*overlimit+not(overlimit);
%     par=par.*underlimit;
% Evaluate the new swarm
    for ii = 1:popsize
        cost(ii) = fitness(par(:,ii),trnX,trnY); % evaluates cost of swarm
    end
% Updating the best local position for each particle
    bettercost = cost < localcost;
    localcost = localcost.*not(bettercost) + cost.*bettercost;
    localpar(:,bettercost==1) = par(:,bettercost==1);
% Updating index g
    [temp, t] = min(localcost);
    if temp<globalcost
        globalpar=localpar(:,t); indx=t; globalcost=temp;
    end
    disp(['function evaluation:  ' num2str(iter*popsize) ' global bestcost: '  num2str(globalcost)]);     % print output each iteration
    minc(iter)=min(cost);         % min for this iteration
    globalmin(iter+1)=globalcost;   % best min so far
    meanc(iter)=mean(cost);       % avg. cost for this iteration
    if(globalcost<stopfitness)
        break;
    end    
    besthistory = [besthistory; iter*popsize globalcost];
end% while
xmin = globalpar;
fmin = globalcost;
stopeval = iter*popsize;
end
% figure(24)
% iters=0:length(minc)-1;
% plot(iters,minc,iters,meanc,'¨C',iters,globalmin,':');
% xlabel('generation');ylabel('cost');
% text(0,minc(1),'best');text(1,minc(2),'population average')