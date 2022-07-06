import numpy as np

def get_doc_features(doc, word2id, max_doc_len=30, max_sen_len=45, oov_token='OOV'):

    """ Parse input document and extract features
    
    Arguments:
        doc (dict):
            dictionary object containing the document and labels
            
        word2id (dict):
            dictionary mapping words to indexes

        max_sen_len (int):
            maximum number of tokens in each sentence to consider, longer sentences are truncated

        max_doc_len (int):
            maximum number of sentences in each documents to consider, longer documents are truncated

        oov_token (str):
            string representation of 'out of vocabulary' token for words which do not have pre-trained embedding vector
    
    Returns:
        x (numpy.ndarray):
            Array of shape (max_doc_len, max_sen_len) 
            containing indexes of words in sentences
            
        y_emotion (numpy.ndarray):
            Array of shape (max_doc_len, 2) 
            labelling whether sentences contain emotion tag or not
            
        y_cause (numpy.ndarray):
            Array of shape (max_doc_len, 2) 
            labelling whether sentences contain cause tag or not
            
        sen_lens (numpy.ndarray):
            Array of shape (max_doc_len) 
            denoting length of each sentence, where maximum value is max_sen_len
            
        doc_len (int):
            integer denoting number of sentences in document or max_doc_len, whichever is larger

        pairs (list[tuple]):
            tuples contains indexes of emotion and cause sentences respectively (index starting from 1)
    
    """

    doc_len = min(len(doc['sentences']), max_doc_len)
    
    pairs = doc['pairs']
    emotion_list, cause_list = zip(*pairs)

    # Data containers
    x = np.zeros(shape=(max_doc_len, max_sen_len), dtype=np.int32)
    y_emotion = np.zeros(shape=(max_doc_len, 2), dtype=np.int32)
    y_cause = np.zeros(shape=(max_doc_len, 2), dtype=np.int32)
    sen_lens = np.zeros(shape=max_doc_len, dtype=np.int32)
    
    # Iterate over sentences
    for i, sent in enumerate(doc['sentences']):
        
        # Max document length cutoff
        if i >= max_doc_len:
            break

        sen_lens[i] = min(len(sent['tokens']), max_sen_len)
        
        y_emotion[i][int(i+1 in emotion_list)] = 1
        y_cause[i][int(i+1 in cause_list)] = 1
        
        # Iterate over words
        for j, word in enumerate(sent['tokens']):
            
            # if maximum sentence length is exceeded, truncate remaining
            if j >= max_sen_len:
                break
                
            # add word index to sentence feature vector
            x[i][j] = word2id[word] if word in word2id else word2id[oov_token]
    
    return x, y_emotion, y_cause, sen_lens, doc_len, pairs

def get_doc_features_batch(docs, word2id, max_doc_len=30, max_sen_len=45, oov_token='OOV'):
    """ Parse input documents and extract features for model-building
    
    Arguments:
        docs (list[dict]):
            list of dictionary objects containing the document and labels
            
        * Remaining arguments same as get_doc_features
    
    Returns:
        x (numpy.ndarray):
            Array of shape (len(docs), max_doc_len, max_sen_len) 
            containing indexes of words in sentences
            
        y_emotion (numpy.ndarray):
            Array of shape (len(docs), max_doc_len, 2) 
            labelling whether sentences contain emotion tag or not
            
        y_cause (numpy.ndarray):
            Array of shape (len(docs), max_doc_len, 2) 
            labelling whether sentences contain cause tag or not
            
        sen_lens (numpy.ndarray):
            Array of shape (len(docs), max_doc_len) 
            denoting length of each sentence, where maximum value is max_sen_len
            
        doc_lens (numpy.ndarray):
            Array of shape (len(docs)) 
            denoting length of each document, where maximum value is max_doc_len

        pairs (list[list[tuple]]):
            tuples contains indexes of emotion and cause sentences respectively (index starting from 1)
    
    """
    
    x, y_emotion, y_cause, sen_lens, doc_lens, pairs = [], [], [], [], [], []

    # Iterate over documents
    for doc in docs:

        x_tmp, y_emotion_tmp, y_cause_tmp, sen_lens_tmp, doc_len_tmp, pairs_tmp = get_doc_features(doc, word2id, max_doc_len, max_sen_len, oov_token)

        # Append all elements
        x.append(x_tmp)
        y_emotion.append(y_emotion_tmp)
        y_cause.append(y_cause_tmp)
        sen_lens.append(sen_lens_tmp)
        doc_lens.append(doc_len_tmp)
        pairs.append(pairs_tmp)

    # Map placeholder elements to numpy
    x, y_emotion, y_cause, sen_lens, doc_lens = map(np.array, [x, y_emotion, y_cause, sen_lens, doc_lens])

    # Printing
    for var in ['x', 'y_emotion', 'y_cause', 'sen_lens', 'doc_lens']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('load data done!\n')
    
    return x, y_emotion, y_cause, sen_lens, doc_lens, pairs