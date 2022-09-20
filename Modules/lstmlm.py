# -*- coding: utf-8 -*-
import time
import numpy as np
from tqdm.auto import tqdm 

import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from IPython.display import clear_output

# Import function from custom module 
from helper_functions import timing


# LSTMLM Model calss
class LSTMLM(Model):
    """LSTM-based Language Model architecture 
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize the LSTMLM that can be used for language modelling and sentence generation

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer, as well as the output dimensionality of the readout layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the size of the LSTM
        """

        super(LSTMLM, self).__init__()

        self.embedding = tfkl.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix], trainable=True)
        self.lstm = tfkl.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)


    def call(self, x, states=None, training=False):
        """Propagates input through the model layer by layer

        Arguments:
            x (tensor): Tensor containing the input
            states (tensor): Tensor containing the initial state for the LSTM
            training (bool): Indicates whether regularization methods should be used or not 
        """
        
        x = self.embedding(x, training=training)
        all_hs, hs, cs = self.lstm(x, initial_state=states, training=training)
        dense_out = self.dense(all_hs, training=training)

        return dense_out, hs, cs


    def inference_mode(self, start_token: int, end_token: int, max_seq_length: int, states=None, temperature=1.0, generate_to_max: bool=False):
        """Call LSTMLM in inference mode: Creating a sequence using only start token and embeddings. 
        Each step gets the previous prediction as additional input.

        Arguments:
            start_token (int): Index of the start token used when generating
            end_token (int): Index of the end token used when generating
            max_seq_length (int): Maximal number of tokens to generate per sentence  
            states(tensor): Tensor containing the initial hidden state for the LSTM 
            temperature (float): Temperature parameter used for temperature sampling of the next token
            generate_to_max (bool): Whether to generate a sentence to max length or to stop when the first eos token is generated

        Returns:
            predictions (List): List containing the generated sequence
        """

        predictions = []

        dense_out, hs, cs = self(x=tf.constant([[start_token]]))
        dense_out = dense_out[:, -1, :]

        # Divide dense_out by temperature for temperature sampling,
        # 0 = Greedy sampling, inf = Uniform sampling
        logits = dense_out/temperature

        # Sample next token from logits
        pred = tf.random.categorical(logits, num_samples=1)
        pred = tf.squeeze(pred, axis=-1)

        states = [hs, cs]

        predictions.append(pred)


        # Prediction stops either when eos token has been generated or the max sequence length has been reached
        for _ in range(max_seq_length-1):

            dense_out, hs, cs = self(x=tf.expand_dims(pred, axis=0), states=states)
            dense_out = dense_out[:, -1, :]
            
            logits = dense_out/temperature

            pred = tf.random.categorical(logits, num_samples=1)
            pred = tf.squeeze(pred, axis=-1)

            states = [hs, cs]

            if (pred  == end_token) and (generate_to_max==False):
                break
                
            predictions.append(pred)

        return predictions
        


# LSTMLM Training functions
@tf.function(experimental_relax_shapes=True)
def train_step(model, input, target, loss_function, optimizer):
    """Perform a training step for the LSTMLM
    1. Propagating the input through the network
    2. Calculating the loss between the networks output and the true targets
    3. Performing Backpropagation and Updating the trainable variables witht the calculated gradients 

    Arguments:
        model (LSTMLM): Given instance of an initialised LSTMLM with all its parameters
        input (tensor): Tensor containing the input data 
        target (tensor): Tensor containing the respective targets 
        loss_function (keras.losses): Function from keras to calculate the loss
        optimizer (keras.optimizers): Function from keras defining the to be applied optimizer during learning 

    Returns:
        loss (tensor): Tensor containing the loss of the network 
    """

    with tf.GradientTape() as tape:
        # 1.
        prediction, _, _ = model(input, training=True)
        # 2. 
        loss = loss_function(target, prediction)
    # 3.
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss



def test_step(model, test_data, loss_function):
    """Tests the models loss over the given dataset with a given loss_function

    Arguments:
        model (LSTMLM): Given instance of an initialised LSTMLM with all its parameters
        test_data (Dataset): Test dataset to test the NN on 
        loss_function (keras.losses): Function from keras to calculate the loss 

    Returns:
        test_loss (float): Average loss of the Network over the test set
    """

    test_loss_aggregator = []

    for input, target in test_data:
        prediction, _, _ = model(input)
        sample_test_loss = loss_function(target, prediction)
        test_loss_aggregator.append(sample_test_loss)

    test_loss = tf.reduce_mean(test_loss_aggregator)

    return test_loss
    


def visualization(word2vec_model, train_losses, test_losses, num_epochs: int, save_path: str, save_plots: bool=False): 
    """Visualize performance and loss for training and test data. 
    
    Arguments:
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        train_losses (list): List containing the training losses of the Network
        test_losses (list): List containing the losses of the Network over the test data
        num_epochs (int): Elapsed number of epochs
        save_path (str): Path to save the plots to 
        save_plots (bool): Determines whether to save the plots or just plot them    
    """ 

    # Plot for visualizing the average loss over the training and test data
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
    ax1.plot(train_losses, label='training')
    ax1.plot(test_losses, label='test')
    ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
    ax1.legend()

    if save_plots:
        plt.savefig(f"{save_path}/LSTMLM_loss_plot_epoch{num_epochs}_transparent.png", dpi=500.0, format="png", transparent=True)
        plt.savefig(f"{save_path}/LSTMLM_loss_plot_epoch{num_epochs}.png", dpi=500.0, format="png")
    plt.show()



# Training loop
def trainModel(model, word2vec_model, start_token: int, end_token: int, max_seq_length: int, save_every: int, save_path: str, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, loss_function: tf.keras.losses, num_epochs: int=20, learning_rate: float=0.001, running_average_factor: float=0.95): 
    """Function that implements the training algorithm for the LSTMLM.
    Prints out useful information and visualizations per epoch.

    Arguments:
        model (LSTMLM): Model that the training algorithm should be applied to
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence
        save_every (int): Determines the amount of epochs before saving weights and plots
        save_path (str): Path to save the weights and images to
        train_dataset (tf.data.Dataset): Dataset to perform training on
        test_dataset (tf.data.Dataset): Dataset to perform testing on
        loss_function (keras.losses): To be applied loss_function during training
        num_epochs (int): Defines the amount of epochs the training is performed
        learning_rate (float): To be used learning rate
        running_average_factor (float): To be used factor for computing the running average of the trainings loss
    """ 

    tf.keras.backend.clear_session()

    # Initialize the optimizer: Adam with custom learning rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    
    # Testing once before we begin on the test and train data
    test_loss = test_step(model=model, test_data=test_dataset, loss_function=loss_function)
    test_losses.append(test_loss)

    train_loss = test_step(model=model, test_data=train_dataset, loss_function=loss_function)
    train_losses.append(train_loss)


    for epoch in range(num_epochs):
        start = time.time()

        # Training and computing running average
        running_average = 0
        pbar = tqdm(total=len(train_dataset))
        for input, target in train_dataset:
            train_loss = train_step(model=model, input=input, target=target, loss_function=loss_function, optimizer=optimizer)
            running_average = running_average_factor * running_average  + (1 - running_average_factor) * train_loss
            pbar.update(1)
        pbar.close()
        
        train_losses.append(running_average)

        # Testing
        test_loss = test_step(model=model, test_data=test_dataset, loss_function=loss_function)
        test_losses.append(test_loss)
        
        # Print useful information
        clear_output()
        print(f"Epoch: {str(epoch+1)}")      
        print()
        print(f"This epoch took {timing(start)} seconds")
        print()
        print(f"Training loss for current epoch: {train_losses[-1]}")
        print()
        print(f"Test loss for current epoch: {test_losses[-1]}")
        print()

        if (epoch+1) % save_every == 0:
            model.save_weights(f"{save_path}/LSTMLM_epoch{epoch+1}")

        print()
        sent = model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        sent = model.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        print()
        
        save_plots = (epoch+1) % save_every == 0
        
        visualization(word2vec_model,
                    train_losses=train_losses, 
                    test_losses=test_losses,
                    num_epochs=epoch+1,
                    save_path=save_path,
                    save_plots=save_plots)

    print()
    model.summary()