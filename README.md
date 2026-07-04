# Rank- and Threat-Dependent Social Modulation of Innate Defensive Behaviors


This repository contains code to reproduce the main and supplementary figures from:

**Gao, X. et al. (2026). Rank- and Threat-Dependent Social Modulation of Innate Defensive Behaviors. eLife.**

## Repository Structure

* `analysis_data.ipynb`: the main code for analyzing data and generating figures. Running it generates all figures.
* `analysis_data_utils.py`: functions that are used in `analysis_data.ipynb`.
* `plot_figures.ipynb`: the code for generating figures from precomputed data (`.pkl` and `.csv` files). Running it generates all figures.
* `data/`: contains all `.csv` and some `.pkl` files required for plotting.
* `lst/`: contains the dataset for looming responses, including mouse tracking recordings (`dlc_h5_data`), behavioral annotations (`bento_annot_data` and `deg_csv_data`), and trial-level summary tables (`.csv`).
* `ret/`: contains the dataset for rat exposure test, including mouse tracking recordings (`ex_xlsx_data`), behavioral annotations (`bento_annot_data` and `deg_csv_data`), and correction tables for analysis (`.csv`).
* `pkl_data`: you need d[`.pkl` files for quickly importing experimental data](https://drive.google.com/file/d/1ww9lFwkD03H-wMi0u8Y5y1aAf-8IegL-/view?usp=sharing).
  
## How to find the code for a specific figure panel
* Run the cells in `analysis_data.ipynb` to generate all figures.
* Search for "FignX" in `analysis_data.ipynb`, where "n" is 1,2,3..., "X" is A,B,C,...

## License
This project is under the Apache 2.0 License.
