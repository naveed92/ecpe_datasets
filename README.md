## ECPE Datasets

Datasets related to the Emotion Cause Pair Extraction (ECPE) task for Natural Language Processing, as well as feature extractors and Pytorch Dataloaders for implementing various models in the literature.

ECPE aims to extract clauses relating to emotions and their corresponding causes in a target document. It was introduced in [this 2019 ACL Paper](https://aclanthology.org/P19-1096.pdf).

### Datasets

There are two datasets in this repository.

- The chinese dataset was introduced in [this paper](https://aclanthology.org/P19-1096.pdf) and exists in the directory `data/cn`.

- The english dataset was introduced in [this paper](https://aaditya-singh.github.io/data/ECPE.pdf) and exists in `data/en`.

Both papers introduced 200-dimensional word vectors as well as 12-fold cross validation splits for evaluation. These are included in the dataset directories and are useful for replicating research results.

### Files:

- `all_data_pair.txt`

    Contains the data in text format. 

    For each document, the first line will contain two integers denoting the document id and the number of sentences respectively, the second line will represent all ground-truth emotion-cause pairs, and the third line onwards will contain the sentences with emotion and cause labels.

- `clause_keywords.txt`

    An auxiliary file used to generate the vocabulary for the word embedding model.

- `w2v_200.txt`

    File containing word vectors.

- `cv_folds.json`

    A json file mapping fold ids to train and test document ids for that fold.

### Installation and Usage

```
git clone ecpe_datasets.git
cd ecpe_datasets
pip install -r requirements.txt
pip install .
cd ..
```

For loading the data only:

```python
import os
from ecpe_data.data import load_data

LANG = 'en' # en or cn
DATA_DIR = 'ecpe_datasets/data/' # relative or absolute path to data folder

path_to_data = os.path.join(DATA_DIR, LANG, 'all_data_pair.txt')
docs = load_data(path_to_data)
```

The docs object is a list of dictionaries where each dictionary represents a document in the dataset, as well as its corresponding labels.

For example, the following document

```
1 3
 (2, 2),
1,null,null,That day Jobs walked into the lobby of the video game manufacturer Atari and told the personnel director
2,surprise,was startled by,who was startled by his unkempt hair and attire
3,null,null,that he would n't leave until they gave him a job .
```

Will be represented as:

```python
{
    "doc_id": 1,
    "pairs": [(2, 2)],
    "sentences": [
        {
            "sent_id": 1,
            "tokens": ["That", "day", "Jobs", "walked", "into", "the", "lobby", "of", "the", "video", "game", "manufacturer", "Atari", "and", "told", "the", "personnel", "director"],
            "emotion": "null",
            "phrase": "null"
        },
        {
            "sent_id": 2,
            "tokens": ["who", "was", "startled", "by", "his", "unkempt", "hair", "and", "attire"],
            "emotion": "surprise",
            "phrase": "was startled by"
        },
        {
            "sent_id": 3,
            "tokens": ["that", "he", "would n't", "leave", "until", "they", "gave", "him", "a", "job", "."],
            "emotion": "null",
            "phrase": "null"
        }
    ]
}
```

### About Feature Extractors and DataLoaders

There are two types of features to be extracted, sentence features and pair features.

Sentence features contain basic features that depend on the sentence or document itself as well as emotion and cause labels. These are useful for building independent models that do not utilize emotion-cause pair information.

Pair features are useful for building models that prediction the emotion-cause relationship between two sentences. This could be a simple logistic regression model or end-to-end neural model.

The `EcpeDataloaderFactory` object can be used to generate various dataloaders that contain features used in the research literature, once initialized.

```python
from ecpe_data.dataloader import EcpeDataloaderFactory

LANG = 'en' # en or cn
FOLD_ID = 1 # Cross validation fold, valid values from 1 to 12 for both datasets
DATA_DIR = 'ecpe_datasets/data/' # relative or absolute path to data folder

dataloader_factory = EcpeDataloaderFactory(LANG, DATA_DIR)
```

### For sentence level features

DataLoaders for sentence level features can be created via:

```python
train_loader, test_loader = dataloader_factory.build_doc_dataloader(FOLD_ID=1, max_sen_len=30, max_doc_len=30, batch_size=32)
```

### For pairwise features:

Currently, pairwise features are implemented as DataLoaders for the following models:

- [Emotion-Cause Pair Extraction: A New Task to Emotion Analysis in Texts](https://aclanthology.org/P19-1096.pdf)

    ```python
    train_loader, test_loader = dataloader_factory.build_pair_dataloader_ecpe(FOLD_ID)
    ```

- [A Dual-Questioning Attention Network for Emotion-Cause Pair Extraction with Context Awareness](https://arxiv.org/pdf/2104.07221.pdf)

    ```python
    train_loader, test_loader = dataloader_factory.build_pair_dataloader_dqan(FOLD_ID)
    ```

- [An End-to-End Network for Emotion-Cause Pair Extraction](https://aaditya-singh.github.io/data/ECPE.pdf)

    ```python
    train_loader, test_loader = dataloader_factory.build_pair_dataloader_e2e(FOLD_ID)
    ```

Within pytorch, the usage is simply:

```python
for batch in train_loader
    <train loop>
```

### Documentation

(Auto generated) API documentation is available [here](https://naveed92.github.io/docs/index.html)