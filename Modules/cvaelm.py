# -*- coding: utf-8 -*-
import time
import numpy as np
from tqdm.auto import tqdm 

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
tfkl = tf.keras.layers
tfd = tfp.distributions

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from IPython.display import clear_output

# Import function from custom module 
from helper_functions import timing


# cVAELM Model classes
class Encoder(Model):
    """Encoder architecture used for the cVAELM
    """

    def __init__(self, vocab_size: int, prior, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize the Encoder that creates an embedding of sentences

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the dimensionality of the hidden state of the lstm and output layer
        """ 

        super(Encoder, self).__init__()

        self.embedding = tfkl.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix], trainable=True)
        self.lstm =  tfkl.LSTM(units=hidden_size)

        # We use the IndependentNormal distribution instead of the MultivariateNormalTriL following the advice found here: 
        # https://towardsdatascience.com/6-different-ways-of-implementing-vae-with-tensorflow-2-and-tensorflow-probability-9fe34a8ab981 
        self.dense = tfkl.Dense(tfp.layers.IndependentNormal.params_size(hidden_size), activation=None)
        self.indnorm = tfp.layers.IndependentNormal(hidden_size, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior), convert_to_tensor_fn=tfp.distributions.Distribution.sample)


    def call(self, x, training: bool=True):
        """Activate our Encoder propagating the input through it layer by layer

        Arguments:
            x (tensor): Tensor containing the input to our Encoder
            training (bool): Indicates whether regularization methods should be used or not when calling the Encoder 

        Returns:
            indnorm_out (tensor): Tensor containing the encoder output probabilistic layer
        """

        x = self.embedding(x)
        hs = self.lstm(x, training=training)
        dense_out = self.dense(hs, training=training) 
        indnorm_out = self.indnorm(dense_out, training=training) 
        
        return indnorm_out
        
        

class Decoder(Model):
    """Decoder architecture used for the cVAELM
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize the Decoder that recreates sentences based on the embeddings of the Encoder

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer, as well as the output dimensionality of the readout layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the dimensionality of the hidden state of the lstm
        """  

        super(Decoder, self).__init__()

        self.embedding = tfkl.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix], trainable=True)
        self.lstm = tfkl.LSTM(units=hidden_size, return_sequences=True, return_state=True)

        # Use the dense layer to project back onto vocab size and apply softmax in order to obtain a probability distribution over all tokens
        self.dense = tfkl.Dense(units=vocab_size, activation="softmax")


    def call(self, x, states=None, return_states: bool=False, training: bool=True):
        """Activate our Decoder propagating the input through it by using teacher forcing

        Arguments:
            x (tensor): Tensor containing the teacher input
            states (tensor): Tensor containing the last hidden state of the LSTM from the Encoder 
            return_states (bool): Whether to return the states hidden and cell states of the LSTM
            training (bool): Indicates whether regularization methods should be used or not when calling the Decoder 

        Returns:
            dense_out (tensor): Tensor containing the reconstructed input
            states (list): Combined hidden and cell state of the LSTM
        """

        x = self.embedding(x)
        hidden_states, hs, cs = self.lstm(x, initial_state=states, training=training)
        dense_out = self.dense(hidden_states, training=training)

        if return_states:
            states = [hs, cs]
            return dense_out, states

        else:
            return dense_out


    def inference_mode(self, start_token: int, end_token: int, max_seq_length: int, states=None, generate_to_max: bool=False):
        """Call Decoder in inference mode: Creating a sequence using only start token and embeddings. 
        Each Decoder step gets the previous prediction of the Decoder as additional input.

        Arguments:
            start_token (int): Index of the start token used when generating
            end_token (int): Index of the end token used when generating
            max_seq_length (int): Maximal number of tokens to generate per sentence
            states (list): Tensor containing the initial hidden state for the LSTM 
            generate_to_max (bool): Whether to generate a sentence to max length or to stop when the first eos token is generated

        Returns:
            predictions (List): List containing the generated sequence
        """

        predictions = []

        dense_out, states = self(tf.constant([[start_token]]), states=[states, tf.zeros_like(states)], return_states=True, training=False)
        dense_out = dense_out[:, -1, :]

        pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1)
        predictions.append(pred)


        # Prediction stops either when eos token has been generated or the max sequence length has been reached
        for _ in range(max_seq_length-1):

            dense_out, states = self(tf.expand_dims(pred, axis=0), states=states, return_states=True, training=False)
            dense_out = dense_out[:, -1, :]
            
            pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1) 
           
            if (pred  == end_token) and (generate_to_max==False):
                break
            
            predictions.append(pred)

        return predictions
        


class CVAELM(Model):
    """cVAELM class that combines the previous Encoder and Decoder
    """

    def __init__(self, vocab_size: int, prior, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize a Variational-Autoencoder consisting of an Encoder and a Decoder

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the dimensionality of the hidden state of the lstms
        """  

        super(CVAELM, self).__init__()

        self.Encoder = Encoder(vocab_size=vocab_size, prior=prior, embedding_matrix=embedding_matrix, embedding_size=embedding_size, hidden_size=hidden_size)
        self.Decoder = Decoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size, hidden_size=hidden_size)


    def call(self, input, teacher, training: bool=True):
        """Activate our Variational-Autoencoder propagating the input through the Encoder and Decoder respectively

        Arguments:
            input (tensor): Tensor containing the input to the Encoder
            teacher (tensor): Tensor containing the input to the Decoder
            training (bool): Indicates whether regularization methods should be used or not when calling the Autoencoder 

        Returns:
            predictions (tensor): Tensor containing the reconstructed input
        """

        hs = self.Encoder(input, training=training)
        hs = tf.convert_to_tensor(hs)
        predictions = self.Decoder(teacher, states=[hs, tf.zeros_like(hs)], training=training)

        return predictions 



# cVAELM Training functions
@tf.function(experimental_relax_shapes=True)
def train_step_ae(model, input, target, teacher, loss_function, optimizer, kl_annealing_weight):
    """Perform a training step for the cVAELM
    1. Propagating the input through the network
    2. Calculating the loss between the networks output and the true targets
    3. Adding the KL loss term as regularizing part 
    4. Performing Backpropagation and Updating the trainable variables witht the calculated gradients 

    Arguments:
        model (CVAELM): Given instance of an initialised cVAELM with all its parameters
        input (tensor): Tensor containing the input data for the encoder
        target (tensor): Tensor containing the respective targets 
        teacher (tensor): Tensor containing the input data for the decoder
        loss_function (keras.losses): Function from keras to calculate the loss
        optimizer (keras.optimizers): Function from keras defining the to be applied optimizer during learning 
        kl_annealing_weight (float): Weight for KL cost annealing

    Returns:
        loss (tensor): Tensor containing the loss of the network 
        kl_loss (tensor): Tensor containing the KL loss of the network
    """

    with tf.GradientTape() as tape:
        # 1.
        prediction = model(input, teacher)
        # 2.
        loss = loss_function(target, prediction)
        kl_loss = tf.reduce_sum(model.losses)
        # 3.
        loss_sum = loss + kl_annealing_weight*kl_loss
    # 4.
    gradients = tape.gradient(loss_sum, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, kl_loss



def test_ae(model, test_data, loss_function, kl_annealing_weight):
    """Tests the models loss over the given data set with a given loss_function

    Arguments:
        model (CVAELM): Given instance of an initialised cVAELM with all its parameters
        test_data (Dataset): Test dataset to test the cVAELM on 
        loss_function (keras.losses): Function from keras to calculate the loss 
        kl_annealing_weight (float): Weight for KL cost annealing

    Returns:
        test_loss (float): Average reconstruction loss of the Network over the test set
        kl_loss (float): Average KL loss of the Network over the test set
    """

    test_loss_aggregator = []
    kl_loss_aggregator = []
    
    for input, target, teacher in test_data:
        prediction = model(input, teacher)
        kl_loss_aggregator.append(tf.reduce_sum(model.losses))
        sample_test_loss = loss_function(target, prediction)
        test_loss_aggregator.append(sample_test_loss)
    
    test_loss = tf.reduce_mean(test_loss_aggregator)
    kl_loss = tf.reduce_mean(kl_loss_aggregator)
    
    return test_loss, kl_loss



def visualization(word2vec_model, rec_train_losses, rec_test_losses, kl_train_losses, kl_test_losses, input_sent, predicted_sent, num_epochs, save_path: str, save_plots: bool=False): 
    """Visualize performance and loss for training and test data. 
    
    Arguments:
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        rec_train_losses (list): List containing the training reconstruction losses of the Network
        rec_test_losses (list): List containing the reconstruction losses of the Network over the test data
        kl_train_losses (list): List containing the training kl losses of the Network
        kl_test_losses (list): List containing the kl losses of the Network over the test data
        input_sent (tuple): Tuple containing the to be Autoencoded sentences 
        predicted_sent (tuple): Tuple containing the reconstructed sentences for visualizing the progress of the Network 
        num_epochs (int): Elapsed number of epochs
        save_path (str): Path to save the plots to 
        save_plots (bool): Determines whether to save the plots or just plot them
    """ 

    # We use the inbuilt index2word from the word2vec model to convert the lists of indices back into their respective tokens
    print("Autoencoded Sentence (Training Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[0][1:]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[0], axis=2).numpy()[0]])}")
    print()
    print("Autoencoded Sentence (Training Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[1][1:]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[1], axis=2).numpy()[0]])}")
    print()
    
    print("Autoencoded Sentence (Test Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[2][1:]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[2], axis=2).numpy()[0]])}")
    print()
    print("Autoencoded Sentence (Test Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[3][1:]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[3], axis=2).numpy()[0]])}")
    print()
    print()

    # Plot for visualizing the average loss over the training and test data
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize = (20, 6))
    ax1[0].plot(rec_train_losses, label='training')
    ax1[0].plot(rec_test_losses, label='test')
    ax1[0].set(ylabel='Reconstruction Loss', xlabel='Epochs', title=f'Average reconstruction loss over {num_epochs} epochs')
    ax1[0].legend()
    ax1[1].plot(kl_train_losses, label='training')
    ax1[1].plot(kl_test_losses, label='test')
    ax1[1].set(ylabel='KL Loss', xlabel='Epochs', title=f'Average kl loss over {num_epochs} epochs')
    ax1[1].legend()

    if save_plots:
        plt.savefig(f"{save_path}/cVAELM_loss_plot_epoch{num_epochs}_transparent.png", dpi=500.0, format="png", transparent=True)
        plt.savefig(f"{save_path}/cVAELM_loss_plot_epoch{num_epochs}.png", dpi=500.0, format="png")

    plt.show()
    
    

# Training loop
def trainModel(model, word2vec_model, save_every: int, save_path: str, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, loss_function: tf.keras.losses, num_epochs: int=180, num_cycles=30, learning_rate: float=0.0005, running_average_factor: float=0.95): 
    """Function that implements the training algorithm the cVAELM.
    Prints out useful information and visualizations per epoch.

    Arguments:
        model (CVAELM): Model that the training algorithm should be applied to
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        save_every (int): Determines the amount of epochs before saving weights and plots
        save_path (str): Path to save the weights and images to
        train_dataset (tf.data.Dataset): Dataset to perform training on
        test_dataset (tf.data.Dataset): Dataset to perform testing on
        loss_function (keras.losses): To be applied loss_function during training
        num_epochs (int): Defines the amount of epochs the training is performed
        num_cycles (int): Number of cycles to use for cyclic kl annealing
        learning_rate (float): To be used learning rate
        running_average_factor (float): To be used factor for computing the running average of the trainings loss
    """ 

    tf.keras.backend.clear_session()

    # Extract two fixed sentences from the training and test dataset each for visualization during training.
    for input, target, teacher in train_dataset.take(1):
        train_sent_for_visualisation_1 = (input[0], teacher[0])
        train_sent_for_visualisation_2 = (input[1], teacher[1])
        
    for input, target, teacher in test_dataset.take(1):
        test_sent_for_visualisation_1 = (input[0], teacher[0])
        test_sent_for_visualisation_2 = (input[1], teacher[1])
        
    # Initialize the optimizer: Adam with custom learning rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Initialize lists for later visualization.
    rec_train_losses = []
    rec_test_losses = []
    kl_train_losses = []
    kl_test_losses = []
    
    # Testing once before we begin on the test and train data
    rec_test_loss, kl_test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function, kl_annealing_weight=0)
    rec_test_losses.append(rec_test_loss)
    kl_test_losses.append(kl_test_loss)

    rec_train_loss, kl_train_loss = test_ae(model=model, test_data=train_dataset, loss_function=loss_function, kl_annealing_weight=0)
    rec_train_losses.append(rec_train_loss)
    kl_train_losses.append(kl_train_loss)
    

    # Implements cyclic annealing following: 
    # https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
    kl_annealing_weight = tf.constant(0.0, dtype="float32")
    current_iteration = tf.constant(0, dtype="int32")
    total_iterations = tf.constant(num_epochs*len(train_dataset), dtype="int32")
    restart_cycle = tf.constant(int(total_iterations/num_cycles), dtype="int32")
    step_size = 1/(restart_cycle/2)
    step_size = tf.cast(step_size, dtype="float32")
    kl_weights = []
    kl_losses_it = []

    for epoch in range(num_epochs):
        
        start = time.time()
        
        kl_losses_temp = []
        
        # Training and computing running average
        running_average = 0
        pbar = tqdm(total=len(train_dataset))
        for input, target, teacher in train_dataset:
            
            rec_train_loss, kl_train_loss = train_step_ae(model=model, 
                                                          input=input, target=target, 
                                                          teacher=teacher, 
                                                          loss_function=loss_function, 
                                                          optimizer=optimizer, 
                                                          kl_annealing_weight=kl_annealing_weight)
            kl_losses_temp.append(kl_train_loss)
            running_average = running_average_factor * running_average  + (1 - running_average_factor) * rec_train_loss
            pbar.update(1)
            
            current_iteration += 1
            kl_weights.append(kl_annealing_weight)
            kl_losses_it.append(kl_train_loss)
            
            # Cyclic annealing
            if (current_iteration % restart_cycle == 0):
                kl_annealing_weight = tf.constant(0.0, dtype="float32")
            elif current_iteration % restart_cycle/2 !=0 and kl_annealing_weight+step_size < tf.constant(1.0, dtype="float32"):
                kl_annealing_weight += step_size
            else:
                kl_annealing_weight = tf.constant(1.0, dtype="float32")  
                
        pbar.close()
        
        kl_train_losses.append(np.mean(kl_losses_temp))
        rec_train_losses.append(running_average)

        # Testing
        rec_test_loss, kl_test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function, kl_annealing_weight=kl_annealing_weight)
        rec_test_losses.append(rec_test_loss)
        kl_test_losses.append(kl_test_loss)
        
        # Print useful information
        clear_output()
        print(f"Epoch: {str(epoch+1)}")      
        print()
        print(f"This epoch took {timing(start)} seconds")
        print()
        print(f"Training loss for current epoch: {rec_train_losses[-1]}")
        print()
        print(f"Test loss for current epoch: {rec_test_losses[-1]}")
        print()
        print(f"KL train loss for current epoch: {kl_train_losses[-1]}")
        print()
        print(f"KL test loss for current epoch: {kl_test_losses[-1]}")
        print()

        if (epoch+1) % save_every == 0:
            model.save_weights(f"{save_path}/cVAELM_epoch{epoch+1}")


        #Feed sample sentences through the network for visualization
        train_pred_sent_1 = model(tf.expand_dims(train_sent_for_visualisation_1[0], axis=0), tf.expand_dims(train_sent_for_visualisation_1[1], axis=0))
        train_pred_sent_2 = model(tf.expand_dims(train_sent_for_visualisation_2[0], axis=0), tf.expand_dims(train_sent_for_visualisation_2[1], axis=0))
        test_pred_sent_1 = model(tf.expand_dims(test_sent_for_visualisation_1[0], axis=0), tf.expand_dims(test_sent_for_visualisation_1[1], axis=0), training=False)
        test_pred_sent_2 = model(tf.expand_dims(test_sent_for_visualisation_2[0], axis=0), tf.expand_dims(test_sent_for_visualisation_2[1], axis=0), training=False)

        save_plots = (epoch+1) % save_every == 0

        visualization(word2vec_model,
                      rec_train_losses=rec_train_losses, 
                      rec_test_losses=rec_test_losses, 
                      kl_train_losses=kl_train_losses,
                      kl_test_losses=kl_test_losses,
                      input_sent=(train_sent_for_visualisation_1[0],train_sent_for_visualisation_2[0], test_sent_for_visualisation_1[0], test_sent_for_visualisation_2[0]), 
                      predicted_sent=(train_pred_sent_1, train_pred_sent_2, test_pred_sent_1, test_pred_sent_2), 
                      num_epochs=epoch+1,
                      save_path=save_path,
                      save_plots=save_plots)
        
    print()
    model.summary()