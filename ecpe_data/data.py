import numpy as np

def load_data(input_path):
    """Read ECPE format training data in list of dictionary format
    
    Arguments:
        input_path (string):
            path of .txt file containing data to load

    Returns:
        docs (list[dict]):
            list of dictionaries where each element of the list corresponds to one document
    """

    # List to store documents
    docs = []

    input_file = open(input_path, 'r', encoding='UTF-8')

    # Read line by line
    while True:
        # Read first line of documnet
        line = input_file.readline()

        # Empty line denotes end of file
        if line == '': break

        # Initialize data containers
        doc = {}
        sents = []

        # split line by whitespace to get document id and length
        doc_id, doc_len = line.strip().split()
        doc_id, doc_len = int(doc_id), int(doc_len)

        # Read line containing pairs and evaluate as list of tuples
        pairs = eval('[' + input_file.readline().strip() + ']')
        
        # Iterate over sentences
        for i in range(doc_len):
            # Read line
            line = input_file.readline().strip()
            splitted = line.split(',')

            # First 3 elements correspond to sentence id, emotion tag, and the emotion's token span
            sent_id, emotion, phrase = splitted[0], splitted[1], splitted[2]

            # Join all remaining parts to get the sentence
            words = ','.join(splitted[3:])

            # Split words by whitespace (words are pre-tokenized)
            tokens = [word.lower() for word in words.split()]

            # Append sentence data to document
            sent = {}
            sent['sent_id'] = sent_id
            sent['tokens'] = tokens
            sent['emotion'] = emotion
            sent['phrase'] = phrase
            sents.append(sent)

        # Append document data to output
        doc['doc_id'] = doc_id
        doc['pairs'] = pairs
        doc['sentences'] = sents
        docs.append(doc)
    
    print('load data done!\n')
    return docs

def build_vocabulary(train_file_path, pad_token='PAD', oov_token='OOV'):
    """ Build a vocabulary from input text file containing training data
    
    Arguments:
        train_file_path (str): path to txt file where each line is comma delimited and last element of each line is 
                               a tokenized sentence where tokens are separated by whitespace
        pad_token (str): Padding token
        oov_token (str): Token for out of vocabulary elements
                               
    Returns:
        words (list): list of unique tokens in dataset
        word_idx (dict): dictionary mapping unqiue tokens to id
        word_idx_rev (dict): dictionary matpping ids to unique tokens
        
    """
    
    # Build vocabulary
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='UTF-8')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        clause = line[-1]
        tokens = [word.lower() for word in clause.split()]
        words.extend(tokens)
    words = set(words) 
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))

    word_idx[pad_token] = 0
    word_idx_rev[0] = pad_token

    word_idx[oov_token] = len(words) + 1
    word_idx_rev[len(words) + 1] = oov_token

    words = [pad_token] + list(words) + [oov_token]
    
    return words, word_idx, word_idx_rev

"""TODO: Set mean and variance of distribution when sampling for words with no pre-trained vector"""
def load_w2v(embedding_path, vocabulary):
    """Load text file holding word2vec embeddings and extract vector for each word in provided vocabulary list
       if vocab word does not exist in word2vec, then generate it randomly
   
   Arguments:
   embeddings_path (str): path to txt file containing pre-trained word embeddings
                           first line contains 2 values, number of words and embedding dimension, separated by whitespace
                           second line onwards contains pairs of unique words and embedding vectors separated by whitespace,
                           where the first element is the word and second element onwards are embedding values
    vocabulary (list): list of unique tokens 
    
    Returns:
    embedding (list): list of embedding vectors which corresponds to input vocabulary
    
    """
    
    print('\nload embedding...')
    
    w2v = {}
    input_file = open(embedding_path, 'r', encoding='UTF-8')
    _, embedding_dim = input_file.readline().split()
    embedding_dim = int(embedding_dim)
    for line in input_file.readlines():
        line = line.strip().split(' ')
        word, embedding = line[0], line[1:]
        w2v[word] = embedding

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in vocabulary:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            # Randomly initialize
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    
    ### add a noisy embedding in the end for out of vocabulary words
    embedding.extend([list(np.random.rand(embedding_dim) / 5. - 0.1)])

    embedding = np.array(embedding)
    
    # Prints
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(vocabulary), hit))
    print("embedding.shape: {}".format(embedding.shape))
    print("load embedding done!\n")
    
    return embedding

def gen_pos_embedding(n_pos=200, dim=50):
    """Generate positional embeddigns given number of positions and desired embedding dimension
       embedding for position 1 is always assigned a zero vector
    
    Arguments:
        n_pos (int): number of positions 
        dim (int): embedding dimenson
    
    Returns:
        embedding_pos (np.Array): numpy array of shape (n_pos+1, dim)
    
    """
    
    embedding_pos = np.random.normal(loc=0.0, scale=0.1, size=(n_pos+1, dim))
    embedding_pos[0] = 0.0
    print("embedding_pos.shape: {}".format(embedding_pos.shape))
    return embedding_pos

