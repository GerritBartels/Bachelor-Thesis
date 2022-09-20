# -*- coding: utf-8 -*-
import copy
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use('ggplot') 

from tqdm.auto import tqdm
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

# Imports for latent space analysis
from bokeh.models import Title
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Colorblind3                  
from bokeh.models import ColumnDataSource               
from bokeh.transform import factor_cmap                 
from bokeh.io import output_notebook




def generate_sentences(model, index_decoder, print_sentences: bool, tokenizer=None, model_name=None, latent_sample_gen=None, num_sent: int=5, start_token: int=0, end_token: int=2, max_seq_length: int=30):
    """Function that generates a given amount of sentences and calculates the average length
  
    Arguments:
        model (Model): Model to generate sentences
        index_decoder (word2vec_model.wv.index2word/tokenizer.decode): List / method to decode generated indices back to a sentence
        print_sentences (bool): Whether to also print the genrated sentences 
        tokenizer (transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast): GPT2 tokenizer
        model_name (str): Name of the model 
        latent_sample_gen (Generator/Prior): Generator or prior distribution to generate samples from the latent space used for decoding
        num_sent (int): Number of sentences that should be generated
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence
    """
    
    if print_sentences:
        print("Generated Sentences:")
        print()

    average_length = 0

    if model_name == "News_cVAELM":
        latent_sample = [tf.expand_dims(sample, axis=0) for sample in latent_sample_gen.sample(num_sent)]

    elif model_name == "News_LaTextGAN":
        latent_sample = [latent_sample_gen(tf.random.normal([1, 1024])) for _ in range(num_sent)]

    else:
        latent_sample = [None]*num_sent


    if model_name == "News_GPT2":
        from transformers import pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        sentences = generator("<Start> ", max_length=max_seq_length, num_return_sequences=num_sent, return_tensors=True, pad_token_id=4)['generated_token_ids']
        
        generated_sentences = []
        
        for sentence in sentences:
            temp = []
            for token_id in sentence[1:]:
                if token_id == 4:
                    break
                temp.append(token_id)
                
            average_length += len(temp)
            generated_sentences.append(temp)
        
        if print_sentences: 
            for sentence in generated_sentences:
                print(index_decoder(sentence))

    else:
        for sample in tqdm(latent_sample):
            tokens = [index_decoder[i.numpy()[0]] for i in model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length-1, states=sample)]
            average_length += len(tokens)
            
            if print_sentences:
                print(f"{' '.join(tokens)}")
                print()

    print()
    print(f"Average length of generated sentences: {average_length/num_sent} tokens")
    print()




def bleu_score(model, index_decoder, reference_data, tokenizer=None, model_name=None, latent_sample_gen=None, num_sent: int=500, n_grams: int=4, start_token: int=0, end_token: int=2, max_seq_length: int=30):
    """Function that calculates a bleu score for a given amount of generated sentences.
    
    Arguments:
        model (Model): Model to generate sentences
        index_decoder (word2vec_model.wv.index2word/tokenizer.decode): List / method to decode generated indices back to a sentence
        reference_data (list): List containing the reference data for the bleu score computation (unbatched)
        tokenizer (transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast): GPT2 tokenizer
        model_name (str): Name of the model 
        latent_sample_gen (Generator/tfp.distributions): Generator or prior distribution to generate samples from the latent space used for decoding
        num_sent (int): Number of sentences that should be generated
        n_grams (int): Order of the to-be-caluclated bleu score
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence 
        
    Returns:
        score_bleu (float): Bleu score for the given n-gram level
    """

    generated_sentence = []

    if model_name == "News_cVAELM":
        latent_sample = [tf.expand_dims(sample, axis=0) for sample in latent_sample_gen.sample(num_sent)]

    elif model_name == "News_LaTextGAN":
        latent_sample = [latent_sample_gen(tf.random.normal([1, 1024])) for _ in range(num_sent)]

    else:
        latent_sample = [None]*num_sent

    if model_name == "News_GPT2":
        from transformers import pipeline
        generator_truncated = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        sentences = generator_truncated("<Start> ", max_length=max_seq_length, pad_token_id=4, num_return_sequences=num_sent, return_tensors=True)['generated_token_ids']
        for sentence in tqdm(sentences):
            temp = []
            for token_id in sentence[1:]:
                if token_id == 4:
                    break
                temp.append(index_decoder(token_id))
            generated_sentence.append(temp)
    else:
        for sample in tqdm(latent_sample):
            generated_sentence.append([index_decoder[i.numpy()[0]] for i in model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length-1, states=sample)])
    
    hyp = generated_sentence
    
    weights = tuple(1./n_grams for _ in range(n_grams))
 
    score_bleu = corpus_bleu([reference_data for i in range(num_sent)], hyp, weights=weights, smoothing_function=SmoothingFunction().method1)
    
    return score_bleu
    
    
    
    
def self_bleu_score(model, index_decoder, tokenizer=None, model_name=None, latent_sample_gen=None, num_sent: int=500, n_grams: int=4, start_token: int=0, end_token: int=2, max_seq_length: int=30):
    """Function that calculates a self-bleu score for a given amount of generated sentences.

    Arguments:
        model (Model): Model to generate sentences
        index_decoder (word2vec_model.wv.index2word/tokenizer.decode): List / method to decode generated indices back to a sentence
        tokenizer (transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast): GPT2 tokenizer
        model_name (str): Name of the model 
        latent_sample_gen (Generator/tfp.distributions): Generator or prior distribution to generate samples from the latent space used for decoding
        num_sent (int): Number of sentences that should be generated
        n_grams (int): Order of the to-be-caluclated bleu score
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence
        
    Returns:
        self_bleu (float): Self-Bleu score for the given n-gram level
    """

    generated_sentence = []

    if model_name == "News_cVAELM":
        latent_sample = [tf.expand_dims(sample, axis=0) for sample in latent_sample_gen.sample(num_sent)]

    elif model_name == "News_LaTextGAN":
        latent_sample = [latent_sample_gen(tf.random.normal([1, 1024])) for _ in range(num_sent)]

    else:
        latent_sample = [None]*num_sent


    if model_name == "News_GPT2":
        from transformers import pipeline
        generator_truncated = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        sentences = generator_truncated("<Start> ", max_length=max_seq_length, pad_token_id=4, num_return_sequences=num_sent, return_tensors=True)['generated_token_ids']
        for sentence in tqdm(sentences):
            temp = []
            for token_id in sentence[1:]:
                if token_id == 4:
                    break
                temp.append(index_decoder(token_id))
            generated_sentence.append(temp)
    else:
        for sample in tqdm(latent_sample):
            generated_sentence.append([index_decoder[i.numpy()[0]] for i in model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length-1, states=sample)])

            
    weights = tuple(1./n_grams for _ in range(n_grams))
    
    references = []
    
    hyps = []
    
    for idx, hyp in enumerate(tqdm(generated_sentence)):
        
        bleu_reference = copy.deepcopy(generated_sentence)

        bleu_reference.pop(idx)
        
        references.append(bleu_reference)
        
        hyps.append(hyp)
        

    self_bleu = corpus_bleu(references, hyps, weights=weights, smoothing_function=SmoothingFunction().method1)

    return self_bleu
    
    
    
    
def word_freq(model, index_decoder, reference_data, tokenizer=None, model_name=None, latent_sample_gen=None, start_token: int=0, end_token: int=2, max_seq_length: int=30):
    """Function that counts word frequencies for generated sentences and test set sentences
  
    Arguments:
        model (Model): Model to generate sentences
        index_decoder (word2vec_model.wv.index2word/tokenizer.decode): List / method to decode generated indices back to a sentence
        reference_data (list): List containing the reference data 
        tokenizer (transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast): GPT2 tokenizer
        model_name (str): Name of the model 
        latent_sample_gen (Generator/tfp.distributions): Generator or prior distribution to generate samples from the latent space used for decoding
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence 
        
    Returns:
        ref_word_freq (dict): Sorted word frequency dict for the reference data 
        gen_word_freq (dict): Sorted word frequency dict for the generated sentences
    """

    ref_word_freq = {}
    gen_word_freq = {}
    
    if model_name == "News_cVAELM":
        latent_sample = [tf.expand_dims(sample, axis=0) for sample in latent_sample_gen.sample(len(reference_data))]

    elif model_name == "News_LaTextGAN":
        latent_sample = [latent_sample_gen(tf.random.normal([1, 1024])) for _ in range(len(reference_data))]

    else:
        latent_sample = [None]*len(reference_data)

    if model_name == "News_GPT2":
        from transformers import pipeline
        generated_sentence = []
        
        generator_truncated = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        sentences = generator_truncated("<Start> ", max_length=max_seq_length, pad_token_id=4, num_return_sequences=len(reference_data), return_tensors=True)['generated_token_ids']
        
        for sentence in tqdm(sentences):
            temp = []
            for token_id in sentence[1:]:
                if token_id == 4:
                    break
                temp.append(index_decoder(token_id))
            generated_sentence.append(temp)

            
    for idx, sample in enumerate(tqdm(latent_sample)):
        
        if model_name == "News_GPT2":
            tokens = generated_sentence[idx]
        
        else:
            tokens = [index_decoder[i.numpy()[0]] for i in model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length-1, states=sample)]


        for token in reference_data[idx]:
            
            if token in ref_word_freq:
                ref_word_freq[token] += 1
            else:
                ref_word_freq[token] = 1

        for token in tokens:
            
            if token in gen_word_freq:
                gen_word_freq[token] += 1
            else:
                gen_word_freq[token] = 1

        
    return dict(sorted(ref_word_freq.items(), key=lambda item: item[1], reverse=True)), dict(sorted(gen_word_freq.items(), key=lambda item: item[1], reverse=True))
    
    
    

def word_freq_plots(reference_freq_dict, generated_freq_dict, top_k: int=12, save_plots: bool=False, save_path: str=""):
    """Function to compare the top_k words from the given dicts and plot them 

    Arguments:
        reference_freq_dict (dict): Dictionary containing the raw frequencies of the reference data
        generated_freq_dict (dict): Dictionary containing the raw frequencies of the generated data
        top_k (int): Number of top tokens to be compared to each other 
        save_plots (bool): Whether to save the plots 
        save_path (str): Path to save the plots to 
    """

    unique_keys = list(set(list(reference_freq_dict.keys())[:top_k] + list(generated_freq_dict.keys())[:top_k]))

    for idx, key in enumerate(unique_keys):
        ref_count = 0
        gen_count = 0

        if key in list(reference_freq_dict.keys())[:top_k]:
            ref_count = reference_freq_dict[key]/1000
        
        if key in list(generated_freq_dict.keys())[:top_k]:
            gen_count = generated_freq_dict[key]/1000
        
        unique_keys[idx] = [key, ref_count, gen_count]

    unique_keys.sort(key = lambda x: x[1]+x[2], reverse=True)

    words = [word for word, _, _ in unique_keys]
    ref_count = [round(num, 1) for _, num, _ in unique_keys]
    gen_count = [round(num, 1) for _, _, num in unique_keys]


    idx = np.arange(len(unique_keys))
    width = 0.37 

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (15, 6))
    bar1 = ax1.bar(x=idx - width/2, height=ref_count, width=width, label='reference', alpha=0.9)
    bar2 = ax1.bar(x=idx + width/2, height=gen_count, width=width, label='generated', alpha=0.9)
    plt.xticks(idx, words, rotation=30)

    for rect in bar1 + bar2:
        height = rect.get_height()
        if height!=0:
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')

    ax1.set(ylabel='Frequency', xlabel='', title=f'Word frequencies in reference and generated data')
    ax1.set_yticklabels(["0", "2k", "4k", "6k", "8k", "10k", "12k"])
    ax1.legend()

    if save_plots:
        plt.savefig(f"{save_path}/LSTMLM_word_freq_top{top_k}_transparent.png", dpi=500.0, format="png", transparent=True)
        plt.savefig(f"{save_path}/LSTMLM_word_freq_top{top_k}.png", dpi=500.0, format="png")
    plt.show()



    
def latent_space_analysis(model, model_name: str, latent_sample_gen, reference_data, num_sent: int, pca: bool=True):
    """Plot 2D TSNE Embedding of Generator against Encoder.

    Arguments:
        model (Encoder): Encoder class instance
        model_name (str): Used for in the title of the plot
        latent_sample_gen (Generator/tfp.distributions): Generator or prior distribution to generate samples from the latent space used for decoding
        reference_data (list): List of sentences to be fed into the Encoder for latent space analysis 
        num_sent (int): number of samples used for plotting   
        pca (bool): whether to apply pca before calculating tsne embeddings
    """

    # Create a list of real tweet encodings from Encoder
    sent_embeddings = [tf.squeeze(model.Encoder(x=tf.expand_dims(sent, axis=0), training=False)) for sent in reference_data[:num_sent]]

    # Create a list of fake tweet encodings from Generator  
    if model_name == "News_cVAELM":
        latent_sample = [sample for sample in latent_sample_gen.sample(num_sent)]
        label = "Prior"

    elif model_name == "News_LaTextGAN":
        latent_sample = [latent_sample_gen(tf.random.normal([1, 1024])) for _ in range(num_sent)]
        label = "Generator"

    num_sent = len(sent_embeddings)
    num_samples = len(latent_sample)

    if pca:
        pca = PCA(n_components=50, svd_solver="randomized", random_state=0)
        sent_embeddings = pca.fit_transform(sent_embeddings)
        print('Cumulative explained variation for encoder embedding: {}'.format(np.sum(pca.explained_variance_ratio_)))
        latent_sample = pca.fit_transform(latent_sample)
        print('Cumulative explained variation for generator embedding: {}'.format(np.sum(pca.explained_variance_ratio_)))
    
    # We apply the TSNE algorithm from scikit to get a 2D embedding of our latent space
    # Once for the Encoder
    tsne = TSNE(n_components=2, perplexity=30., random_state=0)
    tsne_embedding_enc = tsne.fit_transform(sent_embeddings)
    
    # Once for the latent sample generator
    tsne_embedding_gen = tsne.fit_transform(latent_sample)

    # Plotting the TSNE embeddings
    labels =  ["Encoder" for _ in range(num_sent)]
    labels.extend([label for _ in range(num_samples)])

    p = figure(tools="pan,wheel_zoom,reset,save", toolbar_location="above", title=f"2D Encoder and Generator Embeddings.")
    p.title.text_font_size = "25px"
    p.add_layout(Title(text=model_name, text_font_size="15px"), 'above')

    x1=np.concatenate((tsne_embedding_enc[:,0], tsne_embedding_gen[:,0]))
    x2=np.concatenate((tsne_embedding_enc[:,1], tsne_embedding_gen[:,1]))

    # Create column dataset from the tsne embedding and labels
    source = ColumnDataSource(data=dict(x1=x1, x2=x2,names=labels))

    # Create a scatter plot from the column dataset above
    p.scatter(x="x1", y="x2", size=1, source=source, fill_color=factor_cmap('names', palette=Colorblind3, factors=["Encoder", label]), fill_alpha=0.3, line_color=factor_cmap('names', palette=Colorblind3, factors=["Encoder", label]), legend_field='names')  

    show(p)




def js_distance(model, index_decoder, reference_data, tokenizer=None, model_name=None, latent_sample_gen=None, start_token: int=0, end_token: int=2, max_seq_length: int=30):
    """Function that computes the Jensen-Shannon distance for average sentence length and word counts between generated and test set sentences.
    It is implemented as the square root of the Jensen-Shannon divergence.
  
    Arguments:
        model (Model): Model to generate sentences
        index_decoder (word2vec_model.wv.index2word/tokenizer.decode): List / method to decode generated indices back to a sentence
        reference_data (list): List containing the reference data 
        tokenizer (transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast): GPT2 tokenizer
        model_name (str): Name of the model 
        latent_sample_gen (Generator/tfp.distributions): Generator or prior distribution to generate samples from the latent space used for decoding
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence 
        
    Returns:
        jsd_sent_length (float): JS distance for sentence lengths
        jsd_word_count (float): JS distance for word counts
    """

    ref_word_freq = {}
    gen_word_freq = {}

    ref_sent_length = {}
    gen_sent_length = {}

    jsd_sent_length = 0.0
    jsd_word_count = 0.0
    
    if model_name == "News_cVAELM":
        latent_sample = [tf.expand_dims(sample, axis=0) for sample in latent_sample_gen.sample(len(reference_data))]

    elif model_name == "News_LaTextGAN":
        latent_sample = [latent_sample_gen(tf.random.normal([1, 1024])) for _ in range(len(reference_data))]

    else:
        latent_sample = [None]*len(reference_data)

    if model_name == "News_GPT2":
        from transformers import pipeline
        generated_sentence = []
        
        generator_truncated = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        sentences = generator_truncated("<Start> ", max_length=max_seq_length, pad_token_id=4, num_return_sequences=len(reference_data), return_tensors=True)['generated_token_ids']
        
        for sentence in tqdm(sentences):
            temp = []
            for token_id in sentence[1:]:
                if token_id == 4:
                    break
                temp.append(index_decoder(token_id))
            generated_sentence.append(temp)
            

    for idx, sample in enumerate(tqdm(latent_sample)):
        
        if model_name == "News_GPT2":
            tokens = generated_sentence[idx]
        
        else:
            tokens = [index_decoder[i.numpy()[0]] for i in model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length-1, states=sample)]


        # Get the sentence lengths
        ref_length = len(reference_data[idx])
        gen_length = len(tokens)

        # Increment the respective sentence length entry for ref and gen 
        if ref_length in ref_sent_length:
                ref_sent_length[ref_length] += 1
        else:
            ref_sent_length[ref_length] = 1
        
        if gen_length in gen_sent_length:
                gen_sent_length[gen_length] += 1
        else:
            gen_sent_length[gen_length] = 1


        # Loop over the tokens and increment the word count for ref and gen 
        for token in reference_data[idx]:
            
            if token in ref_word_freq:
                ref_word_freq[token] += 1
            else:
                ref_word_freq[token] = 1

        for token in tokens:
            
            if token in gen_word_freq:
                gen_word_freq[token] += 1
            else:
                gen_word_freq[token] = 1

    # Calculate Jensen-Shannon Distance 
    aligned_sent_lengths = align_counts(ref_sent_length, gen_sent_length)
    jsd_sent_length = jensenshannon(aligned_sent_lengths[0], aligned_sent_lengths[1], 2)

    aligned_word_counts = align_counts(ref_word_freq, gen_word_freq)
    jsd_word_count = jensenshannon(aligned_word_counts[0], aligned_word_counts[1], 2)
    
    return jsd_sent_length, jsd_word_count 




def align_counts(ref, gen):

    q_ref = dict.fromkeys(set(list(ref.keys())+list(gen.keys())))
    k_gen = dict.fromkeys(set(list(ref.keys())+list(gen.keys())))

    for key in tqdm(q_ref.keys()):
        try: 
            q_ref[key] = ref[key]
        except:
            q_ref[key] = 0 
        try:
            k_gen[key] = gen[key]
        except:
            k_gen[key] = 0

    return list(q_ref.values()), list(k_gen.values())