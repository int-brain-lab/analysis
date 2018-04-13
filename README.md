# tmp_analysis_matlab
Temporary repository for MATLAB based analysis

Please put Alf related code in the 'alf' directory.

Please use individual named vectors for storing data for now, for example 'timeOnTask', 'RT', 'reward' and if you want to write code to operate over multiple animals or sessions create catagorical vectors, for example 'rat_id', 'sess_id' or 'contrast' that span the length of the data if it's bigger than one contrast/animal/session. This will allow for easy re-factoring and different workflows. Ideally, we would move to the new MATLAB 'table' arrays but I know not everyone has a fully current MATLAB, so the above is a workaround (and can be converted easily).

Open to alternative suggestions