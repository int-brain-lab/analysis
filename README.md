# tmp_analysis_matlab

Temporary repository for MATLAB based analysis

You will need the ability to read and write Python NumPy files. The current recommended solution is:
https://github.com/kwikteam/npy-matlab

Please put Alf related code in the 'alf' directory.

Please create a site specific 'startup.m' to add the analysis directories and dependencies rather than using hard coded paths in the repository. If you have a global startup.m file, you can add the paths and site specific variables there, but I recommend using local startup.m files for each catagory of project you work on and starting matlab there (details are platform dependent). This allows you to specify appropriate settings seperately for IBL and other projects. An example startup.m might be:

% matlab startup file for IBL
addpath(genpath('~/Workspaces/IBL/tmp_analysis_matlab'))
addpath(genpath('~/Workspaces/IBL/npy-matlab'))         

Please use individual named vectors for storing data for now, for example 'timeOnTask', 'RT', 'reward' and if you want to write code to operate over multiple animals or sessions create catagorical vectors, for example 'rat_id', 'sess_id' or 'contrast' that span the length of the data if it's bigger than one contrast/animal/session. This will allow for easy re-factoring and different workflows. Ideally, we would move to the new MATLAB 'table' arrays but I know not everyone has a fully current MATLAB, so the above is a workaround (and can be converted easily).

Open to alternative suggestions