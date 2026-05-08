## This repository includes the following code

## Data Extraction (data)
**Corresponding script:** `JNBcodeWithError.txt`

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

## Individual Star Analysis (results)
**Corresponding scripts:** `main_thesis.py` en `models_thesis.py`

### How to run (on Linux)
* python3 -m astro_env venv
* source astro_env/bin/activate
* pip install numpy astropy matplotlib scipy lightkurve
* python3 main_thesis.py

---

## Result Analysis (discussion)
**Corresponding scripts:** `fourierparametersplot.py` en `petersenplot.py`

### How to run (on Linux)
* python3 -m astro_env venv
* source astro_env/bin/activate
* pip install numpy matplotlib scikit-learn scipy (for petersen script)
* python3 fourierparametersplot.py
* python3 petersenplot.py

---
