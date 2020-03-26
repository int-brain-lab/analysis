
% import 

%{
  # Luigi Acerbi's changepoint model
  -> behavior.TrialSet
  -> ModelName
  ---
  tau: float    # mean activity
  bla: float   # standard deviation of activity
  bla: float     # maximum activity
%}

classdef ChangePoint < dj.Computed

    methods(Access=protected)
        function makeTuples(self,key)

            activity = fetch1(tutorial.Neuron & key,'activity');    % fetch activity as Matlab array

            % compute various statistics on activity
            key.mean = mean(activity); % compute mean
            key.stdev = std(activity); % compute standard deviation
            key.max = max(activity);    % compute max
            self.insert(key);
            sprintf('Computed statistics for for %d experiment on %s',key.mouse_id,key.session_date)

        end
    end
end