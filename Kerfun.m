function  [K, dK] = Kerfun(kernel, x1, x2, p, p2)
% function [K,dK] = evaluate(ker, x1, x2)

%
% EVALUATE
%
%    Evaluate a radial basis function (RBF) kernel, for example
%
%       K = evaluate(kernel, x1, x2);
%
%    where x1 and x2 are matrices containing input patterns, where each column
%    represents a variable and each row represents an observation.

%
% File        : @rbf/evaluate.m
%
% Date        : Sunday 2nd February 2003
%
% Author      : Dr Gavin C. Cawley
%
% Description : Evaluate a radial basis function (RBF) kernel.  Part of an
%               object-oriented implementation of Vapnik's Support
%               Vector Machine, as described in [1].  
%
% References  : [1] V.N. Vapnik,
%                   "The Nature of Statistical Learning Theory",
%                   Springer-Verlag, New York, ISBN 0-387-94559-8,
%                   1995.
%
% History     : 02/02/2003 - v1.00
%
% Copyright   : (c) Dr Gavin C. Cawley, February 2003.
%

ones1 = ones(size(x1, 1), 1);
ones2 = ones(size(x2, 1), 1);

if length(p) > 1

   if nargout == 1

      eta = sqrt(p);

      x1 = x1.*(ones1*eta');
      x2 = x2.*(ones2*eta');

      D = sum(x1.^2,2)*ones2' + ones1*sum(x2.^2,2)' - 2*x1*x2';

      K = exp(-D);

   else
      
      dK = zeros(length(p),length(ones1),length(ones2));
      
      for i=1:length(p)

         dK(i,:,:) = -(x1(:,i)*ones2' - ones1*x2(:,i)').^2;  %p(i)*

      end
      tmpdK = dK;
      for i=1:length(p)
          tmpdK(i,:,:) = p(i)*dK(i,:,:);
      end
      K = exp(squeeze(sum(tmpdK,1)));

      for i=1:length(p)

         dK(i,:,:) = K.*squeeze(dK(i,:,:));   %log(2)*

      end

   end

else

   D = sum(x1.^2,2)*ones2' + ones1*sum(x2.^2,2)' - 2*x1*x2';
   K = exp(-p*D);

   % optionally compute derivative w.r.t. kernel parameter

   if nargout == 2

      dK(1,:,:) = -K.*D;  %log(2)*p*

   end

end

%%

end
