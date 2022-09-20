# -*- coding: utf-8 -*-
import csv
import time
import numpy as np


def dataset_cleanup(data_path: str="", min_sent_len: int=10, max_sent_len: int=28):
    """ Perform final cleanup on the preprocessed data

    Arguments:
        data_path (str): Path to dataset
        min_sent_length (int): Minimum sentence length
        max_sent_length (int): Maximum sentence length

    Returns:
        all_sents_batched (list): Containing the equally sized sequences of text
        max_length (int): Maximum length after adding sos and eos tokens
    """

    # Load the dataset:
    with open(f"drive/MyDrive/BA_2.0/Dataset/news_data_preprocessed_voc_6826_sent_750000.csv", encoding='utf-8', newline="") as file:
        reader = csv.reader(file)
        news_tokenized = list(reader)


    # After-Preprocessing Cleanup:
    # 1. Replace <, NUM, > with <NUM>
    for idefix, sent in enumerate(news_tokenized):
        for obelix, token in enumerate(sent):
            if token=="<":
                del sent[obelix:obelix+3]
                news_tokenized[idefix].insert(obelix, "<NUM>")


    # 2. Delete sentences that are shorter than 10 and longer than 28 tokens
    avg = 0.0
    count = 0
    news_cache= []
    for sent in news_tokenized:
        if len(sent)>(min_sent_len-1) and len(sent)<(max_sent_len+1) and not ("(" in sent and "hr" in sent):
            avg += len(sent)
            news_cache.append(sent)
        else:
            count+=1

    news_tokenized = news_cache


    # 3. Print information
    print(f"Average sentence length: {avg/len(news_tokenized)}")
    print(f"Number of deleted sentences: {count}")
    
    
    # 4. Add start and end of sequence tokens to every sentence and create word2vec data
    word2vec_data = []

    for sent in news_tokenized:
        sent.insert(len(sent), "<End>")
        sent.insert(0, "<Start>")
        word2vec_data.append(sent)
        

    # 5. Finding max length out of all sentences in our dataset 
    # in order to set the max sequence length that our models can generate in inference mode
    max_length = 0
    idx = 0
    for sent in word2vec_data:
        if len(sent) > max_length:
            max_length = len(sent)

    print(f"Longest Sentence has {max_length} tokens.")   


    # Create equally-sized (length=30) sequences of text
    all_sents = []
    for sent in word2vec_data:
        all_sents += sent


    all_sents_batched = []
    counter = 0
    append = False

    for idx, word in enumerate(all_sents):

        counter += 1

        if word == "<Start>" and append == False:
            append = True
            all_sents_batched.append(all_sents[idx:idx+max_length])
            counter = 1

        elif counter == max_length:
            append = False

    all_sents_batched = all_sents_batched[:-1]


    return all_sents_batched, max_length



def word2vec_to_matrix(word2vec_model, embedding_size: int, create_unk: bool=False):
    """Function that converts the word2vec model word vectors into a numpy matrix that is suitable for insertion into our TensorFlow/Keras embedding layer.

    Arguments:
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        embedding_size (int): Size of the word embeddings
        create_unk (bool): Whether to create a mean vector of all token embeddings used for word dropout

    Returns:
        embedding_matrix (ndarray): Matrix containing the converted word embeddings
        vocab_size (int): Size of the vocabulary
    """

    embedding_matrix = np.zeros((len(word2vec_model.wv.vocab), embedding_size))

    for i in range(len(word2vec_model.wv.vocab)):
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    if create_unk:
        mean_vec = np.mean(embedding_matrix, axis = 0)
        word2vec_model.wv["<UNK>"] = mean_vec
        embedding_matrix = np.concatenate((embedding_matrix,[mean_vec]), axis=0)

    print(f"Shape of embedding matrix: {embedding_matrix.shape}")

    vocab_size = len(word2vec_model.wv.vocab)
    print(f"Vocab size of our word2vec model: {vocab_size}")

    return embedding_matrix, vocab_size
    
    
    
def timing(start):
    """Function to time the duration of each epoch

    Arguments:
        start (time): Start time needed for computation 
    
    Returns:
        time_per_training_step (time): Rounded time in seconds 
    """
    now = time.time()
    time_per_training_step = now - start
    return round(time_per_training_step, 4)
    