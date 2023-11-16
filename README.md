# Continuous Management of ML-Based Application Non-Functional Behavior - A Multi-Model Approach

[![CC BY 4.0][cc-by-shield]][cc-by]

[Marco Anisetti](https://homes.di.unimi.it/anisetti), [Claudio A. Ardagna](https://homes.di.unimi.it/ardagna), [Nicola Bena](https://homes.di.unimi.it/bena), [Ernesto Damiani](https://sesar.di.unimi.it/staff/ernesto-damiani/), [Paolo G. Panero](https://www.linkedin.com/in/paolopanero/)

> Modern applications are increasingly driven by Machine Learning (ML) models and their non-deterministic behavior is affecting the entire application life cycle from design to operation. The pervasive adoption of ML is urgently calling for approaches that guarantee a stable non-functional behavior of ML-based applications over time and across model changes. To this aim, non-functional properties of ML models, such as privacy, confidentiality, fairness, and explainability, must be monitored, verified, and maintained. This need is even more pressing when modern applications operate in the cloud-edge continuum, increasing their complexity and dynamicity. Existing approaches mostly focus on *i)* implementing classifier selection solutions according to the functional behavior of ML models, *ii)* finding new algorithmic solutions to this need, such as continuous re-training. In this paper, we propose a multi-model approach built on dynamic classifier selection, where multiple ML models showing similar non-functional properties are available to the application and one model is selected at time according to (dynamic and unpredictable) contextual changes. Our solution goes beyond the state of the art by providing an architectural and methodological approach that continuously guarantees a stable non-functional behavior of ML-based applications, is applicable to any ML models, and is driven by non-functional properties assessed on the models themselves. It consists of a two-step process working during application operation, where *model assessment* verifies non-functional properties of ML models trained and selected at development time, and *model substitution* guarantees a continuous and stable support of non-functional properties. We experimentally evaluate our solution in a real-world scenario focusing on non-functional property fairness.

This repository contains the source code, input dataset, intermediate results and detailed results of our experimental evaluation. **Note**: the dataset description refers to the uncompressed dataset, some directories are compressed for storage reasons.

<!-- vscode-markdown-toc -->
* 1. [Overview](#Overview)
* 2. [Organization](#Organization)
	* 2.1. [Details: Code Organization](#Details:CodeOrganization)
	* 2.2. [Details: Input](#Details:Input)
	* 2.3. [Details: Output](#Details:Output)
* 3. [Reproducibility](#Reproducibility)
* 4. [Citation](#Citation)
* 5. [Acknowledgements](#Acknowledgements)
* 6. [License](#License)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## 1. <a name='Overview'></a>Overview

The code is written in Python 3 and tested on a laptop with Intel Core i7 processor at 2.60 GHz, 16.0 GB of RAM memory, OS Microsoft Windows 10.

The aim of the experimental evaluation is to show the validity of our multi-model MAB-based approach by applying it to a challenging scenario considering non-functional property fairness. We used the well-known dataset of the Connecticut State Department of Correction available at [https://data.ct.gov/Public-Safety/Accused-Pre-Trial-Inmates-in-Correctional-Faciliti/b674-jy6w](https://data.ct.gov/Public-Safety/Accused-Pre-Trial-Inmates-in-Correctional-Faciliti/b674-jy6w). It provides a daily updated list of people detained in the Department facilities awaiting a trial. This dataset anonymously discloses data of individual people detained in the correctional facilities every day starting from July 1st, 2016. It contains attributes such as last admission date, race, gender, age, type of offense and facility description, in more than 4 millions data points (at the download date). We divided this dataset into training and test sets. The training set includes more than 3 million points.

**Note**: all CSV files use `;` as separator.

## 2. <a name='Organization'></a>Organization

This repository is organized in the following directories and files:

* [`Data`](data): contains the CSV files to be fed to the ML models. Files are compressed for storage reasons.
* [`Models`](Models): contains the models used by the MAB (saved in the [*joblib format*](https://scikit-learn.org/stable/model_persistence.html)).
* [`Csv_MAB-OutputFiles.zip`](Csv_MAB-OutputFiles.zip): contains the results of the experiments executions. For each execution with a different memory size, different files are created with different statistics, including raw data used for experiment evaluation, such as:
  * beta distributions (files starting with `beta_detail_noreset`)
  * estimate of each model wins (files starting with `est_models_detail_noreset`)
  * windows information about winner, services performances and so on (files starting with `shift_detail_noreset`)
  * Thompson sampling (files starting with `thommy_detail_noreset`)
  * values remaining in experiment (files starting with `values_remaining`). **For storage reasons, this file is omitted from the repository**.
* [`le_modelv_v2.pkl`](le_modelv_v2.pkl): contains a dictionary for categorical fields used by MAB experiment in pickle format. The same dictionary is provided in JSON format ([`le_modelv_v2.json`](le_modelv_v2.json))
* [`MAB.py`](MAB.py), [`PostProcessData_Fig10.py`](PostProcessData_Fig10.py), and [`PostProcessData_Fig11-12.py`](PostProcessData_Fig11-12.py) contain the experiments code.

### 2.1. <a name='Details:CodeOrganization'></a>Details: Code Organization

The code consists of the following files:

* [`MAB.py`](MAB.py): contains the main algorithm implementing the MAB experiment.
* [`PostProcessData_Fig10.py`](PostProcessData_Fig10.py): contains the postprocessing instructions used to implement fig. 10.
* [`PostProcessData_Fig11-12.py`](PostProcessData_Fig11-12.py): contains the postprocessing instructions used to implement fig. 11 and 12.

### 2.2. <a name='Details:Input'></a>Details: Input

The inputs vary according to the script:. For the sake of simplicity, some settings are hard-coded directly into the scripts:

* [`MAB.py`](MAB.py): uses as input the different datasets in directory [`Data`](data), the serialized models contained in directory [`Models`](Models) and the dictionary for categorical fields in [`le_modelv_v2.pkl`](le_modelv_v2.pkl). Memory percentage is set in the variable `memory` and variance threshold is specified in the constant `VARIANCE`.
* [`PostProcessData_Fig10.py`](PostProcessData_Fig10.py): uses as input some of the files produced by [`MAB.py`](MAB.py), in particular `shift_detail_noreset` and `values_remaining` (converted in CSV format). Files names are hard-coded in variables `master_file*` and `input_file*` (one series for each memory size used) and the values of penalties are set in the `residualError` variable.
* [`PostProcessData_Fig11-12.py`](PostProcessData_Fig11-12.py): uses as input some of the files produced by [`MAB.py`](MAB.py), in particular `shift_detail_noreset` and `values_remaining` (converted in CSV format). It also uses a temporary file from [`PostProcessData_Fig10.py`](PostProcessData_Fig10.py) since the baseline is the same in executions with no memory. Files names are hard-coded in `master_file*` and `input_file*` variables (one series for each memory size used) and the values of penalties are set in the `residualError` variable.

### 2.3. <a name='Details:Output'></a>Details: Output

The code produces aggregated and detailed results in a (custom) txt format which contains serialized data. Such data should usually be converted to CSV format for ease of reading and to be fed to postprocessing scripts, as we did here.

During execution, the code writes output in the execution directory. With the hope that they may be useful, we include **all our data**: our results as well as intermediate data, converted in CSV.

The outputs vary according to the script:

* [`MAB.py`](MAB.py): as mentioned, for each execution with a different memory size, different files are created with different statistics and raw data used for experiment evaluation such as beta distributions (files starting with `beta_detail_noreset`), estimate of each model's wins (files starting with `est_models_detail_noreset`), windows information about winner, services performances and so on (files starting with `shift_detail_noreset`), Thompson sampling (files starting with `thommy_detail_noreset`), values remaining in experiment (files starting with `values_remaining`).
* [`PostProcessData_Fig10.py`](PostProcessData_Fig10.py): produces as well as temporary files, output result in files starting with `Fig10_data` (one for each memory size). The output files provided here are already converted in CSV format.
* [`PostProcessData_Fig11-12.py`](PostProcessData_Fig11-12.py): produces as well as temporary files, output result in files starting with `Fig11-12_data` (one for each memory size). Output files are directly in CSV format.

Individual files have self-explanatory names. Those containing `expAll` contains data evaluating all the experiments together, while the digit `0, 5, 10, 25` indicates the memory used.

**Note**: MAB implies a random execution due to its exploratory behavior. This means that each execution may not produce identical results.

## 3. <a name='Reproducibility'></a>Reproducibility

To reproduce our results, you need to:

1. install the necessary dependencies in [`requirements.txt`](requirements.txt)
2. modify the scripts according to the path where you placed data files
3. run the scripts; scripts do not require parameters.

Experiments on [`MAB.py`](MAB.py) can take hours, while postprocessing scripts usually take minutes.

## 4. <a name='Citation'></a>Citation

It is mandatory to cite our publication, according to our license.

The paper can be cited as follows:

## 5. <a name='Acknowledgements'></a>Acknowledgements

The work was partially supported by the projects:

* MUSA - Multilayered Urban Sustainability Action - project, funded by the European Union - NextGenerationEU, under the National Recovery and Resilience Plan (NRRP) Mission 4 Component 2 Investment Line 1.5: Strengthening of research structures and creation of R&D "innovation ecosystems", set up of "territorial leaders in R&D" (CUP G43C22001370007, Code ECS00000037)
* SERICS (PE00000014) under the NRRP MUR program funded by the EU - NextGenerationEU.

## 6. <a name='License'></a>License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
