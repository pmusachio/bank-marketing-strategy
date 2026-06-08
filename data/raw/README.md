# Data

Source on Kaggle: [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

Expected file in this folder:

- `cc_general.csv`

The pipeline expects the Kaggle CSV at `data/raw/cc_general.csv`.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle datasets download -d arjunbhasin2013/ccdata --unzip -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Rename the file to match what the project expects:

```bash
mv "data/raw/CC GENERAL.csv" data/raw/cc_general.csv 2>/dev/null || true
```

Keep large files out of Git when possible and re-download them in Colab or in your local environment.
