function result = run_train_only(overrides)

folder=fileparts(mfilename('fullpath'));
workspace=fileparts(fileparts(folder));

cfg.inputFile=fullfile(workspace,'NN5_Complete_101.csv');
cfg.variableIndices=2:6;
cfg.headerLines=0;            
cfg.rowStart=1; cfg.rowStop=[]; 
cfg.trainRatio=0.70; cfg.validationRatio=0.15;
cfg.trainEnd=[];          
cfg.validationEnd=[];  
cfg.nWindows=20;     
cfg.maxOrder=10;
cfg.minWindowLength=12; 
cfg.psoPopulation=80; cfg.psoIterations=200;
cfg.c1=2; cfg.c2=2;
cfg.inertiaMax=0.9; cfg.inertiaMin=0.4; cfg.psoVmax=0.1;
cfg.seed=42;
cfg.dtwNormalization='sum';  
cfg.clusterSelection='silhouette';
cfg.clusterCandidates=[];    
cfg.fixedClusters=3;        
cfg.makePlots=true;
cfg.writeOutputs=true;
cfg.outputRoot=fullfile(folder,'results');

if nargin>=1
    if ~isstruct(overrides) || ~isscalar(overrides)
        error('Granulation:Config','Overrides must be a scalar struct.');
    end
    fields=fieldnames(overrides);
    for i=1:numel(fields)
        if ~isfield(cfg,fields{i}), error('Granulation:Config','Unknown config: %s',fields{i}); end
        cfg.(fields{i})=overrides.(fields{i});
    end
end
validateConfig(cfg);
started=tic;
input=readmatrix(cfg.inputFile,'NumHeaderLines',cfg.headerLines);
if isempty(input) || ~isnumeric(input), error('Granulation:Input','No numeric input.'); end
last=cfg.rowStop; if isempty(last), last=size(input,1); end
if cfg.rowStart>last || last>size(input,1)
    error('Granulation:Rows','Invalid row range.');
end
if any(cfg.variableIndices>size(input,2))
    error('Granulation:Columns','Selected column does not exist.');
end
data=input(cfg.rowStart:last,cfg.variableIndices)'; % M x N
if any(~isfinite(data(:)))
    error('Granulation:Missing','Selected data contain missing/nonfinite values. Handle them causally before running.');
end
total=size(data,2);
first=cfg.trainEnd; if isempty(first), first=floor(total*cfg.trainRatio); end
second=cfg.validationEnd;
if isempty(second), second=floor(total*(cfg.trainRatio+cfg.validationRatio)); end
if ~(first>=1 && first<second && second<total)
    error('Granulation:Split','Need 1 <= trainEnd < validationEnd < N.');
end

trainData=data(:,1:first);
validationData=data(:,first+1:second);
testData=data(:,second+1:end);

lagMatch=zeros(1,max(0,size(data,1)-1));
for channel=1:numel(lagMatch)
    left=trainData(channel,2:end); right=trainData(channel+1,1:end-1);
    lagMatch(channel)=mean(abs(left-right)<=1e-10*max(1,abs(left)+abs(right)));
end
if ~isempty(lagMatch) && all(lagMatch>.999)
    warning('Granulation:LagEmbedding', ...
        ['Selected columns exhibit exact one-step shifts on training rows. ', ...
         'Check lag-embedding semantics and evaluate only unobserved physical-time targets.']);
end
GranulationCore.decodeLengths(ones(1,cfg.nWindows),first,cfg.nWindows,cfg.minWindowLength);
fprintf('Raw points: train=%d, validation=%d, test=%d; channels=%d\n', ...
    first,second-first,total-second,size(data,1));
fprintf('PSO, MDL and clustering receive TRAINING VALUES ONLY.\n');
[bestPosition,bestFitness,curve]=GranulationCore.optimize(trainData,cfg);
lengths=GranulationCore.decodeLengths(bestPosition,first,cfg.nWindows,cfg.minWindowLength);
granulation=GranulationCore.granulate(trainData,lengths,cfg.maxOrder);

exportedFitness=mean(granulation.quality(:));
if abs(exportedFitness-bestFitness)>1e-10*max(1,abs(bestFitness))
    error('Granulation:DecodeMismatch','Exported partition differs from the optimized one.');
end
distances=GranulationCore.distanceMatrix(granulation.grains,cfg.dtwNormalization);
[labels,tree,selection]=GranulationCore.clusterChannels(distances,cfg);
result.config=cfg;
result.input=struct('file',cfg.inputFile,'numeric_data_rows',[cfg.rowStart,last], ...
    'selected_columns',cfg.variableIndices,'channels',size(data,1),'points',total, ...
    'training_adjacent_column_shift_match_fraction',lagMatch);
result.split=struct('index_convention','one_based_inclusive_relative_to_selected_series', ...
    'train',[1,first],'validation',[first+1,second],'test',[second+1,total]);
result.train=struct('raw',trainData,'indices',1:first,'granulation',granulation);

result.validation=struct('raw',validationData,'indices',first+1:second);
result.test=struct('raw',testData,'indices',second+1:total);
result.clustering=struct('distance_matrix',distances,'labels',labels, ...
    'linkage',tree,'selection',selection);
result.pso=struct('best_position',bestPosition,'best_fitness',bestFitness, ...
    'exported_partition_fitness',exportedFitness,'curve',curve);
result.protocol=struct('segmentation_fit_scope','train_only','mdl_fit_scope','train_only', ...
    'clustering_fit_scope','train_only','normalization','none_in_this_MATLAB_preprocessor', ...
    'validation_test_target_granules_fitted',false, ...
    'coefficient_order','descending','coefficient_coordinate','physical_local_t_1_to_T', ...
    'sigma_convention','sqrt_of_weighted_membership_variance; effective_Gaussian_std', ...
    'membership_variance','sigma_std_squared; weighted_candidate_residual_variances', ...
    'residual_variance','RSS_over_T; exported_separately_from_effective_Gaussian_spread', ...
    'forecasting_performed',false,'dtw_normalization',cfg.dtwNormalization, ...
    'distance_property','dissimilarity; triangle inequality not guaranteed');
result.elapsed_seconds=toc(started);
result.output_directory='';
if cfg.writeOutputs
    stamp=char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
    output=fullfile(cfg.outputRoot,['run_',stamp]);
    if exist(output,'dir'), error('Granulation:Output','Output directory already exists.'); end
    mkdir(output); result.output_directory=output;

    training_granularData=granulation.grains;
    training_time_series_matrix=trainData;
    training_cluster_labels=labels;
    save(fullfile(output,'training_artifacts.mat'),'result','training_granularData', ...
        'training_time_series_matrix','training_cluster_labels','-v7');
    role={'train';'validation';'test'};
    start_index=[1;first+1;second+1]; end_index=[first;second;total];
    points=end_index-start_index+1;
    writetable(table(role,start_index,end_index,points),fullfile(output,'time_splits.csv'));
    window=(1:cfg.nWindows)'; start_index=granulation.edges(1:end-1)'+1;
    end_index=granulation.edges(2:end)'; window_length=lengths';
    mean_membership=mean(granulation.quality,1)';
    writetable(table(window,start_index,end_index,window_length,mean_membership), ...
        fullfile(output,'training_windows.csv'));
    [channel,window]=ndgrid(1:size(data,1),1:cfg.nWindows);
    channel=channel(:); window=window(:); order=granulation.orders(:); mdl=granulation.mdl(:);
    sigma_std=granulation.sigma_std(:); residual_variance=granulation.residual_variance(:);
    residual_std=granulation.residual_std(:);
    membership_variance=granulation.membership_variance(:); mean_membership=granulation.quality(:);
    coefficients_t=cellfun(@(x)strjoin(compose('%.17g',x),','), ...
        granulation.coefficients_t(:),'UniformOutput',false);
    coefficients_u=cellfun(@(x)strjoin(compose('%.17g',x),','), ...
        granulation.coefficients_u(:),'UniformOutput',false);
    writetable(table(channel,window,order,mdl,sigma_std,residual_variance,residual_std, ...
        membership_variance,mean_membership,coefficients_t,coefficients_u), ...
        fullfile(output,'training_granules.csv'));
    writematrix(distances,fullfile(output,'training_dtw_matrix.csv'));
    channel=(1:size(data,1))'; cluster_label=labels;
    writetable(table(channel,cluster_label),fullfile(output,'training_clusters.csv'));
    iteration=(1:numel(curve))'; best_mean_membership=curve;
    writetable(table(iteration,best_mean_membership),fullfile(output,'pso_curve.csv'));
    manifest=struct('config',cfg,'input',result.input,'split',result.split, ...
        'protocol',result.protocol,'cluster_selection',selection, ...
        'training_window_lengths',lengths,'best_fitness',bestFitness, ...
        'exported_fitness',exportedFitness,'elapsed_seconds',result.elapsed_seconds);
    writeText(fullfile(output,'run_manifest.json'),jsonencode(manifest,'PrettyPrint',true));
    if cfg.makePlots, makePlots(result,output); end
    fprintf('Outputs: %s\n',output);
end
fprintf('Training window lengths:\n'); disp(lengths);
fprintf('Selected clusters=%d; membership=%.8f; elapsed=%.3fs\n', ...
    numel(unique(labels)),bestFitness,result.elapsed_seconds);
end

function validateConfig(cfg)
for name={'nWindows','psoPopulation','psoIterations','minWindowLength','fixedClusters'}
    validateattributes(cfg.(name{1}),{'numeric'},{'scalar','integer','positive'});
end
validateattributes(cfg.maxOrder,{'numeric'},{'scalar','integer','>=',0,'<=',10});
validateattributes(cfg.seed,{'numeric'},{'scalar','integer','>=',0,'<=',2^32-1});
validateattributes(cfg.headerLines,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(cfg.rowStart,{'numeric'},{'scalar','integer','positive'});
for name={'rowStop','trainEnd','validationEnd'}
    if ~isempty(cfg.(name{1})), validateattributes(cfg.(name{1}),{'numeric'},{'scalar','integer','positive'}); end
end
validateattributes(cfg.variableIndices,{'numeric'},{'vector','integer','positive','nonempty'});
if numel(unique(cfg.variableIndices))~=numel(cfg.variableIndices)
    error('Granulation:Columns','Duplicate selected columns.');
end
if cfg.trainRatio<=0 || cfg.validationRatio<=0 || cfg.trainRatio+cfg.validationRatio>=1 ...
        || any(~isfinite([cfg.trainRatio,cfg.validationRatio]))
    error('Granulation:Ratios','Positive train/validation ratios must sum to less than 1.');
end
if cfg.minWindowLength<cfg.maxOrder+2
    error('Granulation:Minimum','minWindowLength must be at least maxOrder+2.');
end
validateattributes(cfg.c1,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(cfg.c2,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(cfg.inertiaMin,{'numeric'},{'scalar','finite','nonnegative'});
validateattributes(cfg.inertiaMax,{'numeric'},{'scalar','finite','>=',cfg.inertiaMin});
validateattributes(cfg.psoVmax,{'numeric'},{'scalar','finite','positive'});
if ~ismember(cfg.dtwNormalization,{'sum','path_mean'})
    error('Granulation:DTWMode','Choose sum or path_mean.');
end
if ~ismember(cfg.clusterSelection,{'fixed','silhouette'})
    error('Granulation:ClusterMode','Choose fixed or silhouette.');
end
if ~isempty(cfg.clusterCandidates)
    validateattributes(cfg.clusterCandidates,{'numeric'},{'vector','integer','>=',2});
end
if ~islogical(cfg.makePlots) || ~isscalar(cfg.makePlots) || ~islogical(cfg.writeOutputs) || ~isscalar(cfg.writeOutputs)
    error('Granulation:Config','makePlots and writeOutputs must be logical scalars.');
end
end

function writeText(filename,content)
[fid,message]=fopen(filename,'w','n','UTF-8');
if fid<0, error('Granulation:File','Cannot create output: %s',message); end
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s',content);
end

function makePlots(result,output)
f=figure('Visible','off'); cleanup=onCleanup(@()close(f));
plot(result.pso.curve,'LineWidth',1.5); grid on;
xlabel('Iteration'); ylabel('Best mean membership'); title('Training-only PSO');
exportgraphics(f,fullfile(output,'pso_training.png'),'Resolution',160);
clf(f); data=result.train.raw; edges=result.train.granulation.edges;
for channel=1:size(data,1)
    subplot(size(data,1),1,channel); plot(data(channel,:)); hold on;
    for index=2:numel(edges)-1, xline(edges(index)+0.5,'r--'); end
    ylabel(sprintf('Channel %d',channel)); grid on;
end
xlabel('Training time index');
exportgraphics(f,fullfile(output,'training_partition.png'),'Resolution',160);
if ~isempty(result.clustering.linkage)
    clf(f); dendrogram(result.clustering.linkage); title('Training-channel average linkage');
    exportgraphics(f,fullfile(output,'training_dendrogram.png'),'Resolution',160);
end
end
