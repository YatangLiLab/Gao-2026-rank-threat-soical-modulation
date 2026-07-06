# Rank- and Threat-Dependent Social Modulation of Innate Defensive Behaviors


This repository contains code to reproduce the main and supplementary figures from:

**Gao, X. et al. (2026). Rank- and Threat-Dependent Social Modulation of Innate Defensive Behaviors. eLife.**

## Repository Structure

* `plot_figures.ipynb`: the code for generating figures from precomputed data (`.pkl` and `.csv` files). Running it generates all figures.
* `analysis_data.ipynb`: the main code for analyzing data and generating figures. Running it generates all figures.
* `analysis_data_utils.py`: functions that are used in `analysis_data.ipynb`.
* `data/`: contains all `.csv` and some `.pkl` files required for plotting.
* `lst/`: contains the dataset for the looming exposure paradigm, including mouse tracking recordings (`dlc_h5_data`), behavioral annotations (`bento_annot_data` and `deg_csv_data`), and trial-level summary tables (`.csv`).
* `ret/`: contains the dataset for the rat exposure paradigm, including mouse tracking recordings (`ex_xlsx_data`), behavioral annotations (`bento_annot_data` and `deg_csv_data`), and correction tables for analysis (`.csv`).
> [!NOTE]
> You need to download the [.pkl files](https://drive.google.com/file/d/1ww9lFwkD03H-wMi0u8Y5y1aAf-8IegL-/view?usp=sharing) first to run the `analysis_data.ipynb`.
  
## How to find the code for a specific figure panel
* Run the cells in `analysis_data.ipynb` to generate all figures.
* Search for "FignX" in `analysis_data.ipynb`, where "n" is 1,2,3..., "X" is A,B,C,...

## Environment & Dependencies
This project was developed and tested with the following environment:

### Python version
- **Python 3.8.18**

### Core dependencies
| Package | Version |
|---------|---------|
| `numpy` | 1.22.4 |
| `pandas` | 1.3.5 |
| `matplotlib` | 3.7.4 |
| `seaborn` | 0.11.2 |
| `scipy` | 1.10.1 |
| `h5py` | 3.10.0 |
| `openpyxl` | 3.1.2 |

### Jupyter environment
| Package | Version |
|---------|---------|
| `ipykernel` | 6.29.5 |
| `ipython` | 8.12.3 |
| `jupyter_client` | 8.6.0 |
| `jupyter_core` | 5.7.2 |


## License
This project is under the Apache 2.0 License.
