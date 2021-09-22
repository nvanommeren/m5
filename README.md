# M5

Created by: Nikki van Ommeren

## Project structure

The project contains the following files:
* data: folder including the input data
* models: directory in which the models are saved after running the pipeline.
* src: source code of the model
* main.py: entry-point of the model
* requirements.txt: libraries used in the project
* example.ipynb: notebook that creates the plots and runs the modules
in separate cells.

## Running the project

It is recommended to first create a virtual environment using:

```
	conda create -n m5_env python=3.7 --file requirements.txt
```

Please activate the newly created environment an run the project
from the root folder, usng the command:

```
	python main.py --days-ahead 1
```
This will run the entire pipeline on all data, creating a one day
ahead forecast which it will saves in the models folder. There are
several optional commandline arguments that help you to run the
project, please run:

```
	python main.py --help
```

for an overview of the commandline arguments. One important one for
testing the project is the --sample 100 argument, which takes a sample
of the project and speeds the process up significantly.

## Cross validation

This project includes a custom cross validation model, which can be run from the
terminal by running or from a notebook:
```
	python main.py --cross-validation
```
Note that the param grid in the function just serves as an example. It is easy
to extend this framework with more parameters or even select the set of features
using this framework.