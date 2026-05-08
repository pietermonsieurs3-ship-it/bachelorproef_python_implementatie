## This repository includes the following code

## Data Extraction (data)

### Script
`JNBcodeWithError.txt`

### How to use
- Open the notebook  
- Copy the contents of `JNBcodeWithError.txt`  
- Paste it into the third cell of the notebook  

### Customize your query
- **Target coordinates**  
  Set your target using decimal coordinates:  
  `target = ...`

- **Pipeline selection**  
  Filter by pipeline name:  
  `if "..." not in author`

- **Sectors**  
  Provide the relevant sectors in order:  
  `tess_spoc_sectors = [...]`

---

## Individual Star Analysis (Results)

### Scripts
- `main_thesis.py`
- `models_thesis.py`

### Input lightcurve files
- `333.85683 +6.82261_sector82_['TESS-SPOC'] (1).txt` — ASAS J071842-5947.7  
- `221.89696 +16.84542_sector51_['TESS-SPOC'] (1).txt` — DH Peg  
- `109.67500 -59.79542_sector64_['TESS-SPOC'] (1).txt` — AE Boo  

### How to run (on Linux)
* python3 -m astro_env venv
* source astro_env/bin/activate
* pip install numpy astropy matplotlib scipy lightkurve
* python3 main_thesis.py

---

## Result Analysis (discussion)

### Scripts
- `fourierparametersplot.py`
-  `petersenplot.py`

### How to run (on Linux)
* python3 -m astro_env venv
* source astro_env/bin/activate
* pip install numpy matplotlib scikit-learn
* python3 fourierparametersplot.py
* python3 petersenplot.py

---
