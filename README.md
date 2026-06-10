# 2026 CdC gen project

## Outline 📝

This repository contains the **source code** of the load curve generation project.

---

### Getting Started 🚀

To install the dependencies, you can use the following commands.

```bash
pip install uv
git clone https://gitlab.pleiade.edf.fr/P11MH/ia_generative/2026_generation_cdc
cd 2026_generation_cdc
uv sync
```

### Code Structure 📁

```
.
├── assets                 # assets for the README file 
├── configs                # configs folder (i.e., '.yaml' files)
├── data                   # data folder (smach, cer, etc.)
├── results                # detailed experiment results folder
├── scripts                # scripts to launch experiments
│   ├── run_one_expe.py    #   python script to launch one experiment
│   └── run_all_expe.sh    #   bash script to launch all experiments
├── src                    # source package
│   ├── helpers            #   helper functions (datset, optim)
│   ├── evaluation         #   evaluation function
│   └── loadit             #   loadit model
├── pyproject.toml         # project setup file
└── uv.lock                # lock to resolve dependencies
```

### Launch an Experiment ⚙️

```
sbatch scripts/bash/{machine}/submit_training_no_conditioning.sh
```

## Data

### **SMACH** data 

~20k clients particuliers simulés sur 2 ans.

---

#### 1. Courbes de charge

File : `load_curve.parquet`

---

#### 2. Métadonnées clients

File : `metadata.parquet`
Colonnes principales : `ID_PDL`, `N_OCCUPANTS`, `N_ACTIFS`, `N_ETUDIANTS`, `N_AUTRES`, `N_RETRAITES`, `N_INACTIFS`, `DATE_CONSTRUCTION`, `OPTION_TARIFAIRE`, `TYPE_LOGT`, `CATEG_TAILLE_LOGT`, `SUPERFICIE_M2`, `PS`, `PLAGE_HC`, `VILLE_METEO`, `EFFORT_SOBRIETE`, `PLAQUE_CUISSON`, `CHAUFF_ELEC`, `ECS_ELEC`, `ECS_ASSERVI`, `CLIM`, `LAVE_LINGE`, `SECHE_LINGE`, `VE`, `VE_TYPE_RECHARGE`, `VE_ASSERVI`.

| ID_PDL     | N_OCCUPANTS | N_ACTIFS | N_ETUDIANTS | DATE_CONSTRUCTION | OPTION_TARIFAIRE | TYPE_LOGT | SUPERFICIE_M2 | VILLE_METEO | CHAUFF_ELEC | VE |
| ---------- | ----------- | -------- | ----------- | ----------------- | ---------------- | --------- | ------------- | ----------- | ----------- | -- |
| 0001234567 | 3           | 2        | 1           | 2005              | BASE             | MAISON    | 95            | Paris       | 1           | 0  |
| 0009876543 | 1           | 1        | 0           | 1998              | HC/HP            | APPART    | 45            | Lyon        | 0           | 1  |


#### 3. Temperature

File : `daily_temperature.csv`