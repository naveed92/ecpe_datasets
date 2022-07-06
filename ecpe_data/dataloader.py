import os
import json

from torch.utils.data import DataLoader

from ecpe_data.data import load_data, load_w2v, build_vocabulary
from ecpe_data.doc_features import get_doc_features_batch
from ecpe_data.pair_features import get_pair_features_ecpe, get_pair_features_dqan, get_pair_features_e2e


def load_json(path):
    with open(path, 'rb') as f:
        data = json.load(f)
    print("Loaded json from path: " + str(path))
    return data

class EcpeDataloaderFactory:
    """Class to generate pytorch dataloaders to train various ECPE models

    Attributes:

    LANG (str):
        Language for the dataset to load, Possible values are 'en' for english or 'cn' for chinese
    
    """

    def __init__(self, LANG='en', data_dir='data/'):

        self.LANG = LANG

        w2v_file_path = os.path.join(data_dir, LANG, 'w2v_200.txt')
        data_path = os.path.join(data_dir, LANG, 'all_data_pair.txt')
        vocab_path = os.path.join(data_dir, LANG, 'clause_keywords.txt')
        folds_path = os.path.join(data_dir, LANG, 'cv_folds.json')

        # Build vocabulary
        self.vocabulary, self.word2id, self.id2word = build_vocabulary(vocab_path)

        # Load embedding matrix
        self.emb_matrix = load_w2v(w2v_file_path, self.vocabulary)

        # Load data
        self.docs = load_data(data_path)

        # Load cross validation fold ids
        self.fold_to_doc_ids = load_json(folds_path)

    def get_docs_by_fold(self, FOLD_ID=1):
        """Given index of the cross validation fold, load train and test documents
        
        Arguments:
            FOLD_ID (int):
                id of the cross validation fold, valid values are from 1 to 12

        Returns:
            train_docs (list[dict]):
                list of dictionary objects containing the document and labels for train data

            test_docs (list[dict]):
                same as train_docs but for test data
        
        """
        
        train_ids = self.fold_to_doc_ids[f'fold_{FOLD_ID}_train']
        test_ids = self.fold_to_doc_ids[f'fold_{FOLD_ID}_test']

        train_docs = [doc for doc in self.docs if doc['doc_id'] in train_ids]
        test_docs = [doc for doc in self.docs if doc['doc_id'] in test_ids]  

        return train_docs, test_docs

    def build_doc_dataloader(self, FOLD_ID=1, max_sen_len=30, max_doc_len=30, batch_size=32):
        """Given index of the cross validation fold, load train and test documents, calculate document level
           features and build pytorch dataloaders
        
        Arguments:
            FOLD_ID (int):
                id of the cross validation fold, valid values are from 1 to 12

            max_sen_len (int):
                maximum number of tokens in each sentence, longer sentences are truncated
                pass

            max_doc_len (int):
                maximum number of sentences in each document, longer documents are truncated

            batch_size (int):
                batch size for gradient descent

        Returns:
            train_loader (DataLoader):
                pytorch dataloader for training data

            test_loader (DataLoader):
                pytorch dataloader for test data
        
        """
        
        train_docs, test_docs = self.get_docs_by_fold(FOLD_ID=FOLD_ID)

        tr_x, tr_y_emotion, tr_y_cause, tr_sen_lens, tr_doc_lens, _ = get_doc_features_batch(train_docs, self.word2id, max_doc_len, max_sen_len)
        te_x, te_y_emotion, te_y_cause, te_sen_lens, te_doc_lens, _ = get_doc_features_batch(test_docs, self.word2id, max_doc_len, max_sen_len)

        train_loader = DataLoader(list(zip(tr_x, tr_y_emotion, tr_y_cause, tr_sen_lens, tr_doc_lens)), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(te_x, te_y_emotion, te_y_cause, te_sen_lens, te_doc_lens)), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def build_pair_dataloader_ecpe(self, FOLD_ID=1, max_sen_len=30, max_doc_len=30, batch_size=32):
        """Given index of the cross validation fold, load train and test documents, calculate pair
           level features and build pytorch dataloaders for the following model:
           https://aclanthology.org/P19-1096.pdf
        
        Arguments:
            * same as build_doc_dataloader

        Returns:
            train_loader (DataLoader):
                pytorch dataloader for training data

            test_loader (DataLoader):
                pytorch dataloader for test data
        
        """

        train_docs, test_docs = self.get_docs_by_fold(FOLD_ID=FOLD_ID)

        train_features = get_pair_features_ecpe(train_docs, self.word2id, max_doc_len, max_sen_len)
        test_features = get_pair_features_ecpe(test_docs, self.word2id, max_doc_len, max_sen_len)

        train_loader = DataLoader(list(zip(*train_features)), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(*test_features)), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def build_pair_dataloader_dqan(self, FOLD_ID=1, max_sen_len=30, max_doc_len=30, batch_size=32):
        """Given index of the cross validation fold, load train and test documents, calculate pair
           level features and build pytorch dataloaders for the following model:
           https://arxiv.org/pdf/2104.07221.pdf
        
        Arguments:
            * same as build_doc_dataloader

        Returns:
            train_loader (DataLoader):
                pytorch dataloader for training data

            test_loader (DataLoader):
                pytorch dataloader for test data
        
        """

        train_docs, test_docs = self.get_docs_by_fold(FOLD_ID=FOLD_ID)

        train_features = get_pair_features_dqan(train_docs, self.word2id, max_doc_len, max_sen_len)
        test_features = get_pair_features_dqan(test_docs, self.word2id, max_doc_len, max_sen_len)

        train_loader = DataLoader(list(zip(*train_features)), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(*test_features)), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def build_pair_dataloader_e2e(self, FOLD_ID=1, max_doc_len=30, batch_size=32):
        """Given index of the cross validation fold, load train and test documents, calculate pair
           level features and build pytorch dataloaders for the following model:
           https://aaditya-singh.github.io/data/ECPE.pdf
        
        Arguments:
            * same as build_doc_dataloader

        Returns:
            train_loader (DataLoader):
                pytorch dataloader for training data

            test_loader (DataLoader):
                pytorch dataloader for test data
        
        """

        train_docs, test_docs = self.get_docs_by_fold(FOLD_ID=FOLD_ID)

        train_features = get_pair_features_e2e(train_docs, max_doc_len)
        test_features = get_pair_features_e2e(test_docs, max_doc_len)

        train_loader = DataLoader(list(zip(*train_features)), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(list(zip(*test_features)), batch_size=batch_size, shuffle=True)

        return train_loader, test_loader
