def preprocessed_data():
    '''Return X and Y train test val'''
    # Make the predictable random
    import numpy as np

    CUST_SEED = 50
    np.random.seed(CUST_SEED)

    # Data set from nltk: (term, tag)
    from nltk.corpus import treebank
    sentences = treebank.tagged_sents(tagset='universal')

    # Extracting a set of tags: 12 tags
    tags = set([tag for sent in sentences for _, tag in sent])
    # Extracting a set of words: 12408 words
    words = set([term for sent in sentences for term,_ in sent])

    ### Separating into train, validation, test data sets
    train_test_cutoff = int(.80 * len(sentences)) 
    training_sentences = sentences[:train_test_cutoff]
    testing_sentences = sentences[train_test_cutoff:]
    train_val_cutoff = int(.25 * len(training_sentences))
    validation_sentences = training_sentences[:train_val_cutoff]
    training_sentences = training_sentences[train_val_cutoff:]

    def add_basic_features(sentence_terms, index):
        """ Compute some very basic word features.
            :param sentence_terms: [w1, w2, ...] 
            :type sentence_terms: list
            :param index: the index of the word 
            :type index: int
            :return: dict containing features
            :rtype: dict
        """
        term = sentence_terms[index]
        return {
            'nb_terms': len(sentence_terms),
            'term': term,
            'is_first': index == 0,
            'is_last': index == len(sentence_terms) - 1,
            'is_capitalized': term[0].upper() == term[0],
            'is_all_caps': term.upper() == term,
            'is_all_lower': term.lower() == term,
            'prefix-1': term[0],
            'prefix-2': term[:2],
            'prefix-3': term[:3],
            'suffix-1': term[-1],
            'suffix-2': term[-2:],
            'suffix-3': term[-3:],
            'prev_word': '' if index == 0 else sentence_terms[index - 1],
            'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
        }

    def untag(tagged_sentence):
        """ 
        Remove the tag for each tagged term.
    :param tagged_sentence: a POS tagged sentence
        :type tagged_sentence: list
        :return: a list of tags
        :rtype: list of strings
        """
        return [w for w, _ in tagged_sentence]
    def transform_to_dataset(tagged_sentences):
        """
        Split tagged sentences to X and y datasets and append some basic features.
    :param tagged_sentences: a list of POS tagged sentences
        :param tagged_sentences: list of list of tuples (term_i, tag_i)
        :return: 
        """
        X, y = [], []
        for pos_tags in tagged_sentences:
            for index, (term, class_) in enumerate(pos_tags):
                # Add basic NLP features for each sentence term
                X.append(add_basic_features(untag(pos_tags), index))
                y.append(class_)
        return X, y

    X_train, y_train = transform_to_dataset(training_sentences)
    X_test, y_test = transform_to_dataset(testing_sentences)
    X_val, y_val = transform_to_dataset(validation_sentences)


    # mapping dictionary to vector (one hot)
    from sklearn.feature_extraction import DictVectorizer
    # Fit our DictVectorizer with our set of features
    dict_vectorizer = DictVectorizer(sparse=False)
    dict_vectorizer.fit(X_train + X_test + X_val)
    # Convert dict features to vectors
    X_train = dict_vectorizer.transform(X_train)
    X_test = dict_vectorizer.transform(X_test)
    X_val = dict_vectorizer.transform(X_val)

    # Encode POS to int
    from sklearn.preprocessing import LabelEncoder
    # Fit LabelEncoder with our list of classes
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train + y_test + y_val)
    # Encode class values as integers
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_val = label_encoder.transform(y_val)

    # Convert integers to dummy variables (one hot encoded)
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val