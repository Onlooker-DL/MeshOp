function out = cd_grf(cmd, varargin)
%CD_GRF Shared localized-GRF machinery (verbatim from the one-sample code).
%   grf = cd_grf('make', cfg, rng)
%   grf = cd_grf('fromcoeffs', cfg, betaAngle, qc, sw, a, b)
%   f   = cd_grf('eval', grf, x, y)
switch lower(cmd)
    case 'make'
        out = grf_make_random(varargin{1}, varargin{2});
    case 'fromcoeffs'
        out = grf_from_coeffs(varargin{1}, varargin{2}, varargin{3}, ...
            varargin{4}, varargin{5}, varargin{6});
    case 'eval'
        out = evalLocalizedForcing(varargin{1}, varargin{2}, varargin{3});
    otherwise
        error('cd_grf:UnknownCmd','Unknown command %s.',cmd);
end
end

function grf = grf_make_random(cfg, seedVal)
    if nargin < 2 || isempty(seedVal), seedVal = 'default'; end
    rng(seedVal);
    grf = makeFourierGRF2D(cfg.grfK,cfg.grfPower,cfg.grfKappa2,cfg.grfAmplitude);
    grf.betaAngle = 2*pi*rand;
    grf.forcingCenter = cfg.forcingCenterMin + ...
        (cfg.forcingCenterMax-cfg.forcingCenterMin)*rand;
    grf.forcingWidth = cfg.forcingWidthMin + ...
        (cfg.forcingWidthMax-cfg.forcingWidthMin)*rand;
end

function grf = grf_from_coeffs(cfg, betaAngle, qc, sw, a, b)
    grf = makeFourierGRF2D(cfg.grfK,cfg.grfPower,cfg.grfKappa2,cfg.grfAmplitude);
    grf.a = double(a(:));
    grf.b = double(b(:));
    grf.betaAngle = double(betaAngle);
    grf.forcingCenter = double(qc);
    grf.forcingWidth = double(sw);
end
function grf=makeFourierGRF2D(K,power,kappa2,amplitude)
%MAKEFOURIERGRF2D
%
% Real Fourier representation of
%
%   g_per^(K) ~ N(0,625 P_K(-Delta+25I)^(-2)P_K)
%
% for the present settings
%
%   K         = 16,
%   power     = 2,
%   kappa2    = 25,
%   amplitude = 25 = sqrt(625).
%
% Since
%
%   (-Delta+25I) e^{i*pi*k.x}
%       = (25+pi^2|k|^2)e^{i*pi*k.x},
%
% the covariance eigenvalue is
%
%   625*(25+pi^2|k|^2)^(-2),
%
% hence the Fourier coefficient standard deviation is
%
%   25*(25+pi^2|k|^2)^(-1).
%
% The real-valued field is represented with one member of each +/-k pair:
%   ky=0, kx=1,...,K,
%   ky=1,...,K, kx=-K,...,K.
%
% No target-standard-deviation rescaling is applied.

kx=[];
ky=[];

% ky = 0, positive kx only
for k1=1:K
    kx(end+1,1)=k1; %#ok<AGROW>
    ky(end+1,1)=0;  %#ok<AGROW>
end

% ky > 0, all kx
for k2=1:K
    for k1=-K:K
        kx(end+1,1)=k1; %#ok<AGROW>
        ky(end+1,1)=k2; %#ok<AGROW>
    end
end

lambda = kappa2 + pi^2*(kx.^2+ky.^2);

% IMPORTANT:
% cfg.grfPower=2 is the covariance-operator exponent.
% A Gaussian coefficient uses the square root of the covariance eigenvalue,
% hence the modal STANDARD DEVIATION has exponent -power/2:
%
%   weight = 25*(25+pi^2|k|^2)^(-1).
%
% Squaring gives
%
%   625*(25+pi^2|k|^2)^(-2),
%
% exactly matching 625(-Delta+25I)^(-2).
weight = amplitude .* lambda.^(-0.5*power);

grf.kx = kx;
grf.ky = ky;
grf.weight = weight;
grf.a = randn(size(weight));
grf.b = randn(size(weight));

end

%% =====================================================================

function value=evalFourierGRF2D(grf,x,y)

originalSize=size(x);

x=x(:);
y=y(:);

if numel(x)~=numel(y)
    error('x and y must have equal numbers of entries.');
end

value=zeros(size(x));

for m=1:numel(grf.weight)

    phase=pi*(grf.kx(m)*x+grf.ky(m)*y);

    value=value+grf.weight(m)*( ...
        grf.a(m)*cos(phase)+grf.b(m)*sin(phase));
end

value=reshape(value,originalSize);

end

%% =====================================================================
%                       Localized random forcing
% ======================================================================


function value=evalLocalizedForcing(grf,x,y)
% f(x,y)=W(q)*g_K(x,y), where
%
%   q = beta_perp dot x
%     = -sin(theta)*x + cos(theta)*y,
%
%   W = exp(-(q-q_c)^2/(2*sigma_q^2)).
%
% g_K is the K=16 two-dimensional Fourier GRF stored in grf.

originalSize=size(x);

xv=x(:);
yv=y(:);

q=-sin(grf.betaAngle).*xv + ...
   cos(grf.betaAngle).*yv;

window=exp( ...
    -0.5*((q-grf.forcingCenter)./grf.forcingWidth).^2);

base=evalFourierGRF2D(grf,xv,yv);

value=window.*base(:);
value=reshape(value,originalSize);

end

%% =====================================================================
%                         FEM geometry helpers
% ======================================================================


