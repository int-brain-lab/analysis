## How to run PsyTrack on IBL data

* ` pip install psytrack` in your IBL environment
* see https://github.com/nicholas-roy/psytrack for documentation
* use a tanh() transformation of the contrasts x using a free parameter p which we set as p=5 throughout the paper: new_x = tanh(px)/tanh(p). Specifically, this maps the contrast values from [0, 0.0625, 0.125, 0.25, 0.5, 1] to [0, 0.302, 0.555, 0.848, 0.987, 1].
