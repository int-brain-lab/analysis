## INSTRUCTIONS FOR CITRIC ACID PLOTS ##
Anne Urai, CSHL, 2019

### 1. clone GitHub repo
Clone (first time) or pull (after that) this `analysis` repository to a convenient place on your laptop

### 2. install datajoint environment
Create an Anaconda environment (e.g. `djenv`) that will have all your DataJoint things in it. Activate that environment, then install DataJoint following these steps: https://tutorials.datajoint.io/setting-up/datajoint-python.html

Make sure you can run Jupyter notebook within conda: `conda install nb_conda_kernels` and `conda install nb_conda`.

Then go to the Snapshot viewer, User Guide, and follow the steps under 'Instructions for Users' to setup the IBL-specific packages you'll need. Specifically, it's useful to add your personal login configuration so you're not prompted for your username and password each time you run the code.

### 3. open Jupyter notebook
Notebooks are a nice way to run your code piece by piece and see the results. `source activate djenv`, `jupyter notebook`, then open the citricAcid_datajoint.ipynb file.

### 4. Check that you can load data
Run the first few cells, and check that you get DataJoint tables as output.

### 5. Analyze your data
See tutorials and documentation on pandas, seaborn (nice for visualization) and matplotlib. See the old weightmonitoring_citricacid notebook for some examples.

When you have something you like, save the notebook, close the Jupyter kernel and then commit your new analysis:
```
git add *
git status 
git commit -m "describe your changes"
git push
```
