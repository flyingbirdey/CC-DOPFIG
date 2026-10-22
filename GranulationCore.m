classdef GranulationCore
methods (Static)
    function lengths = decodeLengths(position, total, count, minimum)
        validateattributes(total, {'numeric'}, {'scalar','integer','positive'});
        validateattributes(count, {'numeric'}, {'scalar','integer','positive'});
        validateattributes(minimum, {'numeric'}, {'scalar','integer','positive'});
        if total < count * minimum
            error('Granulation:TooShort', ...
                'Selected interval has %d points; %d windows need at least %d points.', ...
                total, count, count * minimum);
        end
        position = double(position(:)');
        if numel(position) ~= count || any(~isfinite(position)) || any(position < 0)
            error('Granulation:Position', 'Window weights must be finite and nonnegative.');
        end
        if sum(position) == 0, position = ones(1,count); end
        remainder = total - count * minimum;
        quota = remainder * (position / sum(position));
        allocation = floor(quota);
        left = remainder - sum(allocation);
        [~, order] = sortrows([-(quota-allocation)', (1:count)'], [1 2]);
        allocation(order(1:left)) = allocation(order(1:left)) + 1;
        lengths = minimum + allocation;
        assert(sum(lengths)==total && all(lengths>=minimum));
    end

    function fit = fitWindow(y, maxOrder)
        y = double(y(:)); n = numel(y);
        if n < maxOrder + 2 || any(~isfinite(y))
            error('Granulation:InvalidWindow', 'A window needs maxOrder+2 finite observations.');
        end
        u = (1:n)' / n;
        coeff = cell(1,maxOrder+1); predictions = cell(1,maxOrder+1);
        rss = zeros(1,maxOrder+1);
        for degree = 0:maxOrder
            coeff{degree+1} = polyfit(u,y,degree);
            predictions{degree+1} = polyval(coeff{degree+1},u);
            rss(degree+1) = sum((y-predictions{degree+1}).^2);
        end
        variance = rss / n;
        if any(~isfinite(variance))
            error('Granulation:NonfiniteFit', 'Polynomial fit produced nonfinite residuals.');
        end
        lo = min(variance); hi = max(variance); center = median(variance);
        if hi > lo
            weight = 1 - (variance-lo)/(hi-lo);
        else
            weight = 0.5 * ones(size(variance));
        end
        memberVariance = weight.*variance + (1-weight)*center;
        membershipSum = zeros(size(variance)); mdl = zeros(size(variance));
        for degree = 0:maxOrder
            residual = y-predictions{degree+1};
            v = memberVariance(degree+1);
            if v == 0
                membership = double(residual == 0); % exact-fit Gaussian limit
            else
                membership = exp(-residual.^2/(2*v));
            end
            membershipSum(degree+1) = sum(membership);
            if membershipSum(degree+1) <= 0
                error('Granulation:ZeroMembership', 'Invalid zero membership sum.');
            end
            mdl(degree+1) = n/2*log(n/membershipSum(degree+1)) ...
                + (degree+1)/2*log(n);
        end
        [bestMdl, index] = min(mdl);
        degree = index-1;
        coeffU = coeff{index};
        coeffT = coeffU ./ (n .^ (degree:-1:0));
        sigmaStd = sqrt(memberVariance(index));
        fit = struct('order',degree,'mdl',bestMdl,'coeff_u',coeffU, ...
            'coeff_t',coeffT,'sigma_std',sigmaStd,'residual_variance',variance(index), ...
            'residual_std',sqrt(variance(index)), ...
            'membership_variance',memberVariance(index),'membership_sum',membershipSum(index), ...
            'fitted',predictions{index},'grain',[coeffT,sigmaStd,n], ...
            'candidate_mdl',mdl,'candidate_variance',variance);
    end

    function fitness = fitness(position, data, cfg)
        lengths = GranulationCore.decodeLengths(position,size(data,2), ...
            cfg.nWindows,cfg.minWindowLength);
        edges = [0,cumsum(lengths)];
        quality = zeros(size(data,1),cfg.nWindows);
        for channel = 1:size(data,1)
            for win = 1:cfg.nWindows
                fit = GranulationCore.fitWindow(data(channel,edges(win)+1:edges(win+1)),cfg.maxOrder);
                quality(channel,win) = fit.membership_sum/lengths(win);
            end
        end
        fitness = mean(quality(:));
        if ~isfinite(fitness), error('Granulation:Fitness','Nonfinite fitness.'); end
    end

    function [position, best, curve] = optimize(data,cfg)
        GranulationCore.decodeLengths(ones(1,cfg.nWindows),size(data,2), ...
            cfg.nWindows,cfg.minWindowLength);
        savedRng = rng; cleanup = onCleanup(@()rng(savedRng));
        rng(cfg.seed,'twister');
        X = rand(cfg.psoPopulation,cfg.nWindows);
        V = (2*rand(size(X))-1)*cfg.psoVmax;
        personal = X; scores = zeros(cfg.psoPopulation,1);
        for i = 1:cfg.psoPopulation
            scores(i) = GranulationCore.fitness(X(i,:),data,cfg);
        end
        [best,index] = max(scores); position = X(index,:);
        curve = zeros(cfg.psoIterations,1);
        for iteration = 1:cfg.psoIterations
            if cfg.psoIterations==1, fraction=0; else, fraction=(iteration-1)/(cfg.psoIterations-1); end
            inertia = cfg.inertiaMax-(cfg.inertiaMax-cfg.inertiaMin)*fraction;
            for i = 1:cfg.psoPopulation
                V(i,:) = inertia*V(i,:) ...
                    + cfg.c1*rand(1,cfg.nWindows).*(personal(i,:)-X(i,:)) ...
                    + cfg.c2*rand(1,cfg.nWindows).*(position-X(i,:));
                V(i,:) = max(-cfg.psoVmax,min(cfg.psoVmax,V(i,:)));
                next = max(0,min(1,X(i,:)+V(i,:)));
                score = GranulationCore.fitness(next,data,cfg);
                X(i,:) = next;
                if score>scores(i), scores(i)=score; personal(i,:)=next; end
                if score>best, best=score; position=next; end
            end
            curve(iteration)=best;
            if mod(iteration,20)==0 || iteration==cfg.psoIterations
                fprintf('PSO %d/%d: mean membership = %.8f\n',iteration,cfg.psoIterations,best);
            end
        end
    end

    function result = granulate(data,lengths,maxOrder)
        [channels,total] = size(data);
        if sum(lengths)~=total || any(lengths<maxOrder+2)
            error('Granulation:Lengths','Invalid common window lengths.');
        end
        edges = [0,cumsum(lengths)]; count = numel(lengths);
        result.grains = cell(1,channels);
        result.orders = zeros(channels,count); result.mdl = zeros(channels,count);
        result.sigma_std = zeros(channels,count);
        result.residual_variance = zeros(channels,count);
        result.residual_std = zeros(channels,count);
        result.membership_variance = zeros(channels,count);
        result.quality = zeros(channels,count);
        result.coefficients_t = cell(channels,count); result.coefficients_u = cell(channels,count);
        result.fitted = cell(channels,count);
        result.fixed_features = zeros(channels,count,maxOrder+3);
        for channel = 1:channels
            result.grains{channel} = cell(1,count);
            for win = 1:count
                fit = GranulationCore.fitWindow(data(channel,edges(win)+1:edges(win+1)),maxOrder);
                result.grains{channel}{win}=fit.grain;
                result.orders(channel,win)=fit.order; result.mdl(channel,win)=fit.mdl;
                result.sigma_std(channel,win)=fit.sigma_std;
                result.residual_variance(channel,win)=fit.residual_variance;
                result.residual_std(channel,win)=fit.residual_std;
                result.membership_variance(channel,win)=fit.membership_variance;
                result.quality(channel,win)=fit.membership_sum/lengths(win);
                result.coefficients_t{channel,win}=fit.coeff_t;
                result.coefficients_u{channel,win}=fit.coeff_u;
                result.fitted{channel,win}=fit.fitted;
                padded=[zeros(1,maxOrder-fit.order),fit.coeff_t];
                result.fixed_features(channel,win,:)=reshape([padded,fit.sigma_std,lengths(win)],1,1,[]);
            end
        end
        result.lengths=lengths; result.edges=edges;
    end

    function distance = equalDistance(a,b)
        GranulationCore.validateGrain(a); GranulationCore.validateGrain(b);
        if a(end)~=b(end), error('Granulation:EqualLength','Durations must agree.'); end
        x=a(1:end-2); y=b(1:end-2); width=max(numel(x),numel(y));
        q=[zeros(1,width-numel(x)),x]-[zeros(1,width-numel(y)),y];
        total=a(end);
        q=q.*(total.^((width-1):-1:0));
        first=find(q~=0,1);
        if isempty(first)
            area=0;
        else
            q=q(first:end); rr=roots(q);
            rr=real(rr(abs(imag(rr))<1e-9 & real(rr)>0 & real(rr)<1));
            cuts=unique([0;sort(rr(:));1]); anti=polyint(q);
            area=total*sum(abs(diff(polyval(anti,cuts))));
        end
        distance=area+sqrt(2*pi)/2*abs(a(end-1)-b(end-1))*total;
        if ~isfinite(distance) || distance<0, error('Granulation:Distance','Invalid distance.'); end
    end

    function translated = shiftPolynomial(coeff,offset)
        translated=coeff(1);
        for j=2:numel(coeff)
            translated=conv(translated,[1,offset]);
            translated(end)=translated(end)+coeff(j);
        end
    end

    function value = derivative(coeff,order,where)
        for j=1:order, coeff=polyder(coeff); end
        value=polyval(coeff,where);
    end

    function row = derivativeRow(degree,order,where)
        row=zeros(1,degree+1);
        for column=1:degree+1
            power=degree-column+1;
            if power>=order
                row(column)=factorial(power)/factorial(power-order)*where^(power-order);
            end
        end
    end

    function coeff = bridge(a,b)
        gap=b(end)-a(end); short=a(1:end-2); long=b(1:end-2);
        if gap<=0, error('Granulation:Bridge','Bridge needs a shorter first granule.'); end
        degree=max(1,numel(short)-1); A=zeros(degree+1); rhs=zeros(degree+1,1);
        A(1,:)=GranulationCore.derivativeRow(degree,0,0);
        rhs(1)=polyval(short,a(end));
        A(2,:)=GranulationCore.derivativeRow(degree,0,1);
        rhs(2)=polyval(long,b(end));
        row=2; order=1;
        while row<degree+1
            row=row+1; A(row,:)=GranulationCore.derivativeRow(degree,order,0);
            rhs(row)=gap^order*GranulationCore.derivative(short,order,a(end));
            if row<degree+1
                row=row+1; A(row,:)=GranulationCore.derivativeRow(degree,order,1);
                rhs(row)=gap^order*GranulationCore.derivative(long,order,b(end));
            end
            order=order+1;
        end
        coeffU=(A\rhs)';
        if norm(A*coeffU'-rhs,inf)>1e-8*max(1,norm(rhs,inf))
            error('Granulation:BridgeResidual','Bridge constraints not satisfied.');
        end
        coeff=coeffU./(gap.^(degree:-1:0));
    end

    function distance = unequalDistance(a,b)
        GranulationCore.validateGrain(a); GranulationCore.validateGrain(b);
        if a(end)==b(end), distance=GranulationCore.equalDistance(a,b); return; end
        if a(end)>b(end), temp=a; a=b; b=temp; end
        gap=b(end)-a(end); head=b; head(end)=a(end);
        tail=[GranulationCore.shiftPolynomial(b(1:end-2),a(end)),b(end-1),gap];
        bridge=[GranulationCore.bridge(a,b),0,gap];
        distance=GranulationCore.equalDistance(a,head)+GranulationCore.equalDistance(bridge,tail);
    end

    function validateGrain(grain)
        if ~isrow(grain) || numel(grain)<3 || any(~isfinite(grain)) || grain(end)<=0 || grain(end-1)<0
            error('Granulation:Grain','Expected [descending coefficients, std>=0, duration>0].');
        end
    end

    function [distance,pathCells] = dtwDistance(first,second,normalization)
        n=numel(first); m=numel(second);
        if n==0 || m==0, error('Granulation:EmptySequence','DTW sequences cannot be empty.'); end
        costs=inf(n+1,m+1); cells=zeros(n+1,m+1); costs(1,1)=0;
        for i=1:n
            for j=1:m
                [previous,index]=min([costs(i,j),costs(i,j+1),costs(i+1,j)]);
                counts=[cells(i,j),cells(i,j+1),cells(i+1,j)];
                costs(i+1,j+1)=previous+GranulationCore.unequalDistance(first{i},second{j});
                cells(i+1,j+1)=counts(index)+1;
            end
        end
        distance=costs(end,end); pathCells=cells(end,end);
        switch normalization
            case 'sum' 
            case 'path_mean', distance=distance/pathCells;
            otherwise, error('Granulation:DTWMode','Choose sum or path_mean.');
        end
    end

    function matrix = distanceMatrix(grains,normalization)
        channels=numel(grains); matrix=zeros(channels);
        for i=1:channels
            for j=i+1:channels
                value=GranulationCore.dtwDistance(grains{i},grains{j},normalization);
                matrix(i,j)=value; matrix(j,i)=value;
            end
        end
    end

    function score = meanSilhouette(matrix,labels)
        groups=unique(labels); n=numel(labels); scores=zeros(n,1);
        if numel(groups)<2 || numel(groups)>=n
            error('Granulation:Silhouette','Silhouette needs 2 through M-1 actual clusters.');
        end
        for i=1:n
            same=find(labels==labels(i)); same(same==i)=[];
            if isempty(same), continue; end 
            a=mean(matrix(i,same)); b=inf;
            for group=groups(:)'
                if group~=labels(i), b=min(b,mean(matrix(i,labels==group))); end
            end
            if max(a,b)>0, scores(i)=(b-a)/max(a,b); end
        end
        score=mean(scores);
    end

    function [labels,tree,selection] = clusterChannels(matrix,cfg)
        channels=size(matrix,1); tree=[];
        selection=struct('method',cfg.clusterSelection,'candidate_requested',[], ...
            'candidate_actual',[],'mean_silhouette',[],'chosen_count',1,'fallback','');
        if size(matrix,2)~=channels || any(~isfinite(matrix(:))) || any(matrix(:)<0) ...
                || norm(matrix-matrix',inf)>1e-10 || any(diag(matrix)~=0)
            error('Granulation:Matrix','Need a finite symmetric nonnegative dissimilarity matrix.');
        end
        if channels<2 || ~any(matrix(:))
            labels=ones(channels,1); selection.fallback='one_channel_or_all_zero_dissimilarity'; return;
        end
        tree=linkage(squareform(matrix),'average');
        if strcmp(cfg.clusterSelection,'fixed')
            target=min(cfg.fixedClusters,channels);
            labels=cluster(tree,'maxclust',target);
        elseif strcmp(cfg.clusterSelection,'silhouette')
            candidates=cfg.clusterCandidates;
            if isempty(candidates), candidates=2:channels-1; end
            candidates=sort(unique(candidates(candidates>=2 & candidates<channels)));
            if isempty(candidates)
                labels=ones(channels,1); selection.fallback='no_valid_candidate_small_M';
            else
                selection.candidate_requested=candidates;
                selection.candidate_actual=zeros(size(candidates));
                selection.mean_silhouette=nan(size(candidates));
                best=-inf; labels=ones(channels,1);
                for j=1:numel(candidates)
                    proposed=cluster(tree,'maxclust',candidates(j)); actual=numel(unique(proposed));
                    selection.candidate_actual(j)=actual;
                    if actual<2 || actual>=channels, continue; end
                    score=GranulationCore.meanSilhouette(matrix,proposed);
                    selection.mean_silhouette(j)=score;
                    if score>best, best=score; labels=proposed; end
                end
                if ~isfinite(best), selection.fallback='no_valid_hierarchical_cut'; end
            end
        else
            error('Granulation:ClusterMode','Choose fixed or silhouette.');
        end
        selection.chosen_count=numel(unique(labels));
    end
end
end
