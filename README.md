# Horror authors prediction

**Tensorflow** model for solving a problem specified at a [Kaggle](https://www.kaggle.com/)
problem [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification).

The pipeline assumes that there exists a data directory pointed by a `RESEARCH_DATA_DIR`
environment variable which maintains following structure:

- `$RESEARCH_DATA_DIR`
  - `stackexchange`
    - `preproc`
        - `train` (preprocessed training data in *TFRecords* format)
        - `test` (preprocessed test data in TFRecords format)
        - `vocab.txt` (created during preprocessing step using the corpus)
        - `wiki.simple.npy` (preprocessed embeddings for faster training)
    - **`raw`** (raw CSV files from Kaggle)
    - **`wiki.simple.bin`** and **`wiki.simple.vec`**
    ([fastText pretrained embeddings](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md))
    
**Entities written in bold must be available before the preprocessing step.**
    
## Usage

In order to run the preprocessing invoke following command from the repository root:
```
> python3 -m horror.data.prepare
```

Train and evaluate the module using the main module directly. You might use multiple models
by specifying names with a `-n` option. If a model with given name exists the pipeline
will further train or evaluate this model. An environment variable `TF_MODELS_DIR` might
be defined in order to store models data in specific directory. Otherwise system-specific
temporary directory will be used.
```
> python3 -m horror train -n xyz
... (training logs) ...

> python3 -m horror test -n xyz
... (evaluation report) ...
```
