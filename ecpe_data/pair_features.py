import numpy as np

def get_pair_features_ecpe(docs, word2id, max_doc_len=30, max_sen_len=45, oov_token='OOV'):
    """
    Get positive and negative samples and features to train pair extraction model for the Paper:
    https://aclanthology.org/P19-1096.pdf

    Arguments:
        docs (list[dict]):
            list of dictionaries where each element of the list corresponds to one document

        word2id (dict):
            dictionary mapping words to indexes

        max_sen_len (int):
            maximum number of tokens in each sentence to consider, longer sentences are truncated

        max_doc_len (int):
            maximum number of sentences in each documents to consider, longer documents are truncated

        oov_token (str):
            string representation of 'out of vocabulary' token for words which do not have pre-trained embedding vector

    Returns:
        pair_ids (list[int]):
            list of automatically generated pair ids
        
        y_pair (numpy.ndarray):
            Array of shape (len(pair_ids), 2) 
            Specifies whether given sentence pair has a emotion-cause relationship

        x_pair (numpy.ndarray):
            Array of shape (len(pair_ids), 2, max_sen_len)
            Stores the feature vector of each pair of sentences

        sen_lens (numpy.ndarray):
            Array of shape (len(pair_ids), 2)
            Stores sentence length of each pair of sentences
            
        distance (numpy.ndarray):
            Array of shape (len(pair_ids)) 
            Stores distance between sentence indexes in the pair plus 100

    """

    # Output containers
    pair_ids, y_pair, x_pair, sen_lens, distance = [], [], [], [], []
    
    # Iterate over documents
    for doc in docs:

        # Get document id and emotion cause pairs
        doc_id = doc['doc_id']
        pairs = doc['pairs']

        # Generate unique id for all pairs
        pair_id_all = [doc_id*10000+p[0]*100+p[1] for p in pairs]
        
        # Get indexes of emotion and cause sentences
        emotion_list, cause_list = zip(*doc['pairs'])
        emotion_list, cause_list = list(set(emotion_list)), list(set(cause_list))
        emotion_list = [emotion for emotion in emotion_list if emotion <= max_doc_len]
        cause_list = [cause for cause in cause_list if cause <= max_doc_len]

        # Data containers
        sen_len_tmp = np.zeros(shape=max_doc_len, dtype=np.int32)
        x_tmp = np.zeros(shape=(max_doc_len, max_sen_len), dtype=np.int32)

        # Iterate over sentences
        for i, sent in enumerate(doc['sentences']):

            # Max document length cutoff
            if i >= max_doc_len:
                break

            # Get sentence length
            sen_len_tmp[i] = min(len(sent['tokens']), max_sen_len)

            # Iterate over words
            for j, word in enumerate(sent['tokens']):
                
                # if maximum sentence length is exceeded, truncate remaining
                if j >= max_sen_len:
                    break
                    
                # add word index to sentence feature vector
                if word in word2id:
                    x_tmp[i][j] = int(word2id[word])
                else:
                    x_tmp[i][j] = int(word2id[oov_token])
        
        # Iterate over all pairs of cause and emotion
        for i in emotion_list:
            for j in cause_list:
                # generate a pair id
                pair_id_cur = doc_id*10000+i*100+j
                
                # Add to list of ids
                pair_ids.append(pair_id_cur)
                
                # Get true label
                y_pair.append([0,1] if pair_id_cur in pair_id_all else [1,0])

                # Get feature pair for sentences
                x_pair.append([x_tmp[i-1], x_tmp[j-1]])
                
                # Get sentence lengths of pair
                sen_lens.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])

                # Get distance
                distance.append(j-i+100)

    # Cast to numpy array
    y_pair, x_pair, sen_lens, distance = map(np.array, [y_pair, x_pair, sen_lens, distance])

    # Print output shapes
    for var in ['y_pair', 'x_pair', 'sen_lens', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('(y-negative, y-positive): {}'.format(y_pair.sum(axis=0)))
    print('load data done!\n')
    
    # Output
    return pair_ids, y_pair, x_pair, sen_lens, distance

def get_pair_features_dqan(docs, word2id, max_doc_len=30, max_sen_len=45, oov_token='OOV'):
    """
    Get positive and negative samples and features to train pair extraction model for the Paper:
    https://arxiv.org/pdf/2104.07221.pdf

    Arguments:
        * Same as get_pair_features_ecpe

    Returns:
        pair_ids (list[int]):
            list of automatically generated pair ids

        y_pair (numpy.ndarray):
            Array of shape (len(pair_ids), 2) 
            Specifies whether given sentence pair has a emotion-cause relationship

        x_pair (numpy.ndarray):
            Array of shape (len(pair_ids), max_doc_len, max_sen_len)
            Stores the feature vector of the document which the pair belongs to

        pair_idx (numpy.ndarray):
            Array of shape (len(pair_ids), 2)
            Stores indices of the sentences in the pair

        sen_lens (numpy.ndarray):
            Array of shape (len(pair_ids), max_doc_len)
            Stores sentence lengths of all sentences in the document which the pair belongs to
        
        doc_lens (numpy.ndarray):
            Array of shape (len(pair_ids))
            Stores length of the document the pair belongs to
        
        related_dis1 (numpy.ndarray):
            Array of shape (len(pair_ids), max_doc_len)
            Stores distance between index of emotion clause, and each sentence in document plus 100

        related_dis2 (numpy.ndarray):
            Array of shape (len(pair_ids), max_doc_len)
            Stores distance between index of cause clause, and each sentence in document plus 100

    """

    # Output containers
    pair_ids, y_pair, x_pair, sen_lens, pair_idx, doc_lens, related_dis1, related_dis2 = [], [], [], [], [], [], [], []
    
    # Iterate over documents
    for doc in docs:

        # Get document id and emotion cause pairs
        doc_id = doc['doc_id']
        pairs = doc['pairs']

        # Get document lengths
        doc_len = min(len(doc['sentences']), max_doc_len)

        # Generate unique id for all pairs
        pair_id_all = [doc_id*10000+p[0]*100+p[1] for p in pairs]
        
        # Get indexes of emotion and cause sentences
        emotion_list, cause_list = zip(*doc['pairs'])
        emotion_list, cause_list = list(set(emotion_list)), list(set(cause_list))
        emotion_list = [emotion for emotion in emotion_list if emotion <= max_doc_len]
        cause_list = [cause for cause in cause_list if cause <= max_doc_len]

        # Data containers
        sen_len_tmp = np.zeros(shape=max_doc_len, dtype=np.int32)
        x_tmp = np.zeros(shape=(max_doc_len, max_sen_len), dtype=np.int32)

        # Iterate over sentences
        for i, sent in enumerate(doc['sentences']):

            # Max document length cutoff
            if i >= max_doc_len:
                break

            # Get sentence length
            sen_len_tmp[i] = min(len(sent['tokens']), max_sen_len)

            # Iterate over words
            for j, word in enumerate(sent['tokens']):
                
                # if maximum sentence length is exceeded, truncate remaining
                if j >= max_sen_len:
                    break
                    
                # add word index to sentence feature vector
                if word in word2id:
                    x_tmp[i][j] = int(word2id[word])
                else:
                    x_tmp[i][j] = int(word2id[oov_token])

        # Iterate over all pairs of cause and emotion
        for i in emotion_list:
            for j in cause_list:
                # generate a pair id

                pair_id_cur = doc_id*10000+i*100+j

                # Add to list of ids
                pair_ids.append(pair_id_cur)

                # Get true label
                y_pair.append([0,1] if pair_id_cur in pair_id_all else [1,0])

                # Get feature for entire doc
                x_pair.append(x_tmp)

                # Get entire doc len
                doc_lens.append(doc_len)

                # Get pair ids
                pair_idx.append([i-1, j-1])

                # get lengths for all sentences
                sen_lens.append(sen_len_tmp)
                
                # Instead of distance between cause and emotion
                # we take the distance from emotion and cause to each sentence in document
                # as a feature, this allows to use entire document information
                emotion_idx, cause_idx = i-1, j-1
                dist_1, dist_2 = [], []
                for a in range(max_doc_len):
                    dist_1.append(a - emotion_idx + 100)
                    dist_2.append(a - cause_idx + 100)
                related_dis1.append(dist_1)
                related_dis2.append(dist_2)

     # Cast to numpy array
    y_pair, x_pair, pair_idx, sen_lens, doc_lens, related_dis1, related_dis2 = map(np.array, [y_pair, x_pair, pair_idx, sen_lens, doc_lens, related_dis1, related_dis2])
    
    # Print output shapes
    for var in ['y_pair', 'x_pair', 'pair_idx', 'sen_lens', 'doc_lens', 'related_dis1', 'related_dis2']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('(y-negative, y-positive): {}'.format(y_pair.sum(axis=0)))
    print('load data done!\n')
    return pair_ids, y_pair, x_pair, pair_idx, sen_lens, doc_lens, related_dis1, related_dis2

def get_pair_features_e2e(docs, max_doc_len=30):
    """
    Get positive and negative samples and features to train pair extraction model for the Paper:
    https://aaditya-singh.github.io/data/ECPE.pdf
    
    Arguments:
        docs (list[dict]):
            list of dictionaries where each element of the list corresponds to one document
            
        max_doc_len (int):
            maximum number of sentences in each documents to consider, longer documents are truncated

    Returns:
        y_pair (numpy.ndarray):
            Array of shape (len(docs), max_doc_len * max_doc_len, 2) 
            Specifies whether given sentence pair has a emotion-cause relationship
            
        distance (numpy.ndarray):
            Array of shape (len(docs), max_doc_len * max_doc_len) 
            Distance between sentence pairs, values range from 90 to 110, and is 0 for non-existent sentence pair

    """
    
    y_pair, distance = [], []
    
    for doc in docs:
        
        # Get document id and emotion cause pairs
        doc_id = doc['doc_id']
        pairs = doc['pairs']

        doc_len = min(len(doc['sentences']), max_doc_len)

        # Generate unique id for all pairs
        pair_id_all = [doc_id*10000+p[0]*100+p[1] for p in pairs]
        
        # Data containers
        y_pair_tmp = np.zeros(shape=(max_doc_len * max_doc_len, 2))
        distance_tmp = np.zeros(shape=(max_doc_len * max_doc_len, ))

        for i in range(doc_len):
            for j in range(doc_len):
                # get pair id to match with true ids
                pair_id_curr = doc_id*10000+(i+1)*100+(j+1)
                
                # Check whether i, j clauses are emotion cause pairs
                y_pair_tmp[i*max_doc_len+j][int(pair_id_curr in pair_id_all)] = 1
                
                # Find the distance between the clauses, and use the same embedding beyond 10 clauses
                distance_tmp[i*max_doc_len+j] = min(max(j-i+100, 90), 110)

        y_pair.append(y_pair_tmp)
        distance.append(distance_tmp)

    y_pair, distance = map(np.array, [y_pair, distance])

    for var in ['y_pair', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
        
    return y_pair, distance

