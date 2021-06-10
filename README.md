# natural-computing
Repository for the course Natural Computing. 

## Preparations
First, clone the repository:

```
git clone https://github.com/JVerbeek/natural-computing/
cd natural-computing/Project
```
### Option 1: Create a conda environment to run the project
Then, build the conda environment from the .yml file in the repo:

```
conda env create -f environment.yml
conda activate naco
```
### Option 2: Make sure that the following packages are installed: 
These packages are required when running from the console or from Spyder IDE:
```
numpy       : 1.19.2
pytorch     : 1.8.1
pandas      : 1.2.3
matplotlib  : 3.3.2
tqdm        : 4.56.0
spyder      : 4.1.5
```
If you want to run the Jupyter Notebook, make sure the following packages are installed (should come with the default Jupyter installation).
```
jupyter core     : 4.7.1
jupyter-notebook : 6.2.0
qtconsole        : 4.7.7
ipython          : 7.19.0
ipykernel        : 5.4.3
jupyter client   : 6.1.11
jupyter lab      : 3.0.6
nbconvert        : 6.0.7
ipywidgets       : not installed
nbformat         : 5.1.2
traitlets        : 5.0.5

```
## Running the project
Run the project through:
```
python3 ACO_clean.py
```
or open `/natural-computing/Project/ACO-clean.py` in an IDE and run it there.
Another option is running Jupyter notebook and running the .ipynb.

## Problems?
Mail j.verbeek@student.ru.nl. 

