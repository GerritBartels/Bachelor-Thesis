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


# Gumbel-Softmax Layer
class GumbelSoftmax(tfkl.Layer):
    """Custom layer that realizes the Gumbel-Softmax trick employed by Kusner and Hernández-Lobato 
    [https://arxiv.org/abs/1611.04051]
    """

    def __init__(self):
        """Initialize the layer
        """

        super(GumbelSoftmax, self).__init__()
        self.gumbel = tfp.distributions.Gumbel(loc=0.0, scale=1.0)


    def call(self, inputs, tau):
        """Adds samples from a Gumbel distribution to the input and applys temperature controlled softmax

        Arguments:
            inputs (tensor): Tensor containing the input
            tau (float): Temperature parameter to control one-hotness of the returned vector
        
        Returns:
            gs_out (tf.tensor): One-hot-like output
            hard_token (tensor): Tensor containing the hard tokens of the one-hot-like output
        """

        # Adds the drawn samples from the gumbel distribution to the input
        inputs += self.gumbel.sample(tf.shape(input=inputs))

        # Create one-hot-like output by applying softmax to the previous result weighted by tau  
        gs_out = tf.nn.softmax(logits=inputs/tau, axis=-1)

        # Get the hard token of the one-hot-like output via argmax
        hard_token = tf.stop_gradient(tf.argmax(input=gs_out, axis=-1, output_type=tf.int32))

        return gs_out, hard_token



# Model classes
class Discriminator(Model):
    """Discriminator architecture used for the GSGAN
    """

    def __init__(self, embedding_size: int=256, hidden_size: int=1024):
        """Initialize a Discriminator that decides whether the input sentence embedding is fake or real 

        Arguments:
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the dimensionality of the hidden state of the lstm
        """ 

        super(Discriminator, self).__init__()

        self.embedding = tfkl.Dense(units=embedding_size)
        self.LSTM = tfkl.LSTM(units=hidden_size, return_sequences=True)
        self.out = tfkl.Dense(units=1, activation="sigmoid")


    def call(self, x, training: bool=True):
        """Activates the Discriminator propagating the input through it layer by layer

        Arguments:
            x (tensor): Tensor containing the input
        
        Returns:
            x (tesnor): Tensor containing the decision of the Discriminator
        """

        x = self.embedding(x, training=training)
        x = self.LSTM(x, training=training)
        x = self.out(x, training=training)

        return x



class Generator(Model):
    """Generator architecture used for the GSGAN
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024, tau: float=5.0):
        """Initialize the Generator

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer, as well as the output dimensionality of the readout layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the hidden size of the LSTM
            tau (float): Temperature hyperparameter
        """

        super(Generator, self).__init__()

        self.tau = tau
        self.hidden_size = hidden_size

        self.embedding = tfkl.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix], trainable=True)
        self.lstm = tfkl.LSTM(units=hidden_size, return_sequences=True, return_state=True)
        self.dense = tfkl.Dense(units=vocab_size)
        self.gumbel = GumbelSoftmax()


    def call(self, x, states=None, return_state: bool=False, return_logits: bool=False):
        """Activate our Generator propagating the input through it layer by layer.

        Arguments:
            x (tensor): Tensor containing the input tokens   
            states (tensor): Tensor containing the initial state for the LSTM
            return_state (bool): Whether to additionally return the last hidden/cell state of the LSTM
            return_logits (bool): Whether to additionally return the logits (output of the dense layer)

        Returns:
            gumbel_out (tensor): One-hot-like output of the Gumbel-Softmax layer
            token (tensor): Tensor containing the hard tokens of the one-hot-like output
            states (tensor): Tensor containing the last hidden/cell state of the LSTM
            x (tensor): Tensor containing the logits 
        """

        x = self.embedding(x)
        x, hs, cs = self.lstm(x, initial_state=states)
        x = self.dense(x)

        gumbel_out, token = self.gumbel(x, self.tau)

        if return_state and return_logits:
            states = [hs, cs]
            return gumbel_out, token, states, x

        elif return_state and (not return_logits):
            states = [hs, cs]
            return gumbel_out, token, states

        elif return_logits:
            return gumbel_out, token, x

        else:
            return gumbel_out, token


    def inference_mode(self, start_token: int, end_token: int, max_seq_length: int=30, states=None, generate_to_max: bool=False):
        """Call Generator in inference mode: Creating a sequence of tokens using only start token and embeddings.
        Each step gets the previous prediction as additional input.

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

        gumbel_out, _, states = self(x = tf.constant([[start_token]]), states=[tf.random.normal(shape=(1,self.hidden_size)), tf.random.normal(shape=(1,self.hidden_size))],  return_state=True)
        gumbel_out = gumbel_out[:, -1, :]

        pred = tf.argmax(gumbel_out, output_type=tf.int32, axis=-1)
        predictions.append(pred)

        # Prediction stops either when eos token has been generated or the max sequence length has been reached
        for _ in range(max_seq_length-1):

            gumbel_out, _, states = self(x = tf.expand_dims(pred, axis=0), states=states, return_state=True)
            gumbel_out = gumbel_out[:, -1, :]

            pred = tf.argmax(gumbel_out, output_type=tf.int32, axis=-1)

            if (pred  == end_token) and (generate_to_max==False):
                break

            predictions.append(pred)    

        return predictions

    
    def generate_batch(self, vocab_size: int, start_idx: int, batch_size: int, max_seq_length: int):
        """Generates a batch of sequences 

        Arguments:
            vocab_size (int): Defines the dimensionality of the one-hot-like gumbel outputs
            start_idx (int): Index of the start token used when generating
            batch_size (int): Size of the to-be-created batch
            max_seq_length (int): Number of tokens to generate per sentence

        Returns:
            generated_batch (tensor): Tensor containing a batch of sequences in token form 
            gumbel_outputs (tensor): Tensor containing a batch of sequences as one-hot-like gumbel outputs
        """

        start_token_batch = tf.zeros(shape=(batch_size,1), dtype="int32")+start_idx
        generated_batch = tf.zeros([batch_size, 0], dtype="int32")
        gumbel_outputs = tf.zeros([batch_size, 0, vocab_size])

        gumbel_out, start_token_batch, states = self(x = start_token_batch, states=[tf.random.normal(shape=(batch_size,self.hidden_size)), tf.random.normal(shape=(batch_size,self.hidden_size))], return_state=True)

        gumbel_outputs = tf.concat(values=[gumbel_outputs, gumbel_out], axis=1)
        generated_batch = tf.concat(values=[generated_batch, start_token_batch], axis=1)


        for _ in range(max_seq_length-1):
 
            gumbel_out, start_token_batch, states = self(x = start_token_batch, states=states, return_state=True)
                                                                                          
            gumbel_outputs = tf.concat(values=[gumbel_outputs, gumbel_out], axis=1)
            generated_batch = tf.concat(values=[generated_batch, start_token_batch], axis=1)


        return generated_batch, gumbel_outputs



# Pre-training functions
@tf.function()
def pre_step_GAN(generator, input, target, optimizer, batch_size: int):
    """Perform a pre-training step for the GSGAN's generator
    1. Propagating the input through the network
    2. Calculating the loss between the networks output and the true targets
    3. Performing Backpropagation and Updating the trainable variables witht the calculated gradients 

    Arguments:
        generator (Generator): Given instance of an initialised GSGAN with all its parameters
        input (tensor): Tensor containing the input data 
        target (tensor): Tensor containing the respective targets 
        optimizer (tf.keras.optimizers): Function from keras defining the to be applied optimizer during training
        batch_size (int): Size of the input batches used for the generators random state
    
    Returns:
        loss (tensor): Tensor containing the loss of the network 
    """
    with tf.GradientTape() as tape:
        # 1.
        _, _, logits = generator(x=input, states=[tf.random.normal(shape=(batch_size,generator.hidden_size)), tf.random.normal(shape=(batch_size,generator.hidden_size))], return_logits=True)
        # 2.
        loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(target, logits))
    # 3.
    gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    return loss



def pre_test_step_GAN(generator, test_data, batch_size: int):
    """Tests the models loss over the given dataset

    Arguments:
        generator (generator): Given instance of an initialised GSGAN with all its parameters
        test_data (Dataset): Test dataset to test the GSGAN on 
        batch_size (int): Size of the input batches used for the generators random state

    Returns:
        test_loss (float): Average loss of the Network over the test set
    """
    test_loss_aggregator = []

    for input, target in test_data:
        _, _, logits = generator(x=input, states=[tf.random.normal(shape=(batch_size,generator.hidden_size)), tf.random.normal(shape=(batch_size,generator.hidden_size))], return_logits=True)
        loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(target, logits))
        test_loss_aggregator.append(loss)
    
    test_loss = tf.reduce_mean(test_loss_aggregator)

    return test_loss
    
    

# Loss functions
def discriminator_loss(real, fake):
    """Calculate the loss for the discriminator.

    Arguments:
        real (tensor): Linear output from discriminator
        fake (tensor): Linear output from discriminator

    Returns:
        x (tensor): loss
    """
    bce = tf.keras.losses.BinaryCrossentropy()
    loss_real = bce(tf.ones_like(real), real)
    loss_fake = bce(tf.zeros_like(fake), fake)
                                              
    return loss_real + loss_fake


def generator_loss(fake):
    """Calculate the loss for the generator.

    Arguments:
        fake (tensor): Linear output from discriminator

    Returns:
        x (tensor): loss
    """
    bce = tf.keras.losses.BinaryCrossentropy()
    loss = bce(tf.ones_like(fake), fake)

    return loss



# Pre-train loop 
def pre_fit(generator, word2vec_model, start_token: int, end_token: int, max_seq_length: int, save_every: int, save_path: str, pre_train_dataset: tf.data.Dataset, pre_test_dataset: tf.data.Dataset, batch_size: int, num_epochs: int=15, running_average_factor: float=0.95, learning_rate: float=0.001):
    """Function for pre-training the GSGAN's generator.
    Prints out useful information and visualizations per epoch.

    Arguments:
        generator (Generator): Generator class instance 
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence
        save_every (int): Determines the amount of epochs before saving weights and plots
        save_path (str): Path to save the weights and images to
        pre_train_dataset (tf.data.Dataset): Dataset to perform training on
        pre_test_dataset (tf.data.Dataset): Dataset to perform testing on
        batch_size (int):  Size of the input batches used for the generators random state
        num_epochs (int): Defines the amount of epochs the training is performed
        running_average_factor (float): To be used factor for computing the running average of the trainings loss
        learning_rate (float): To be used learning rate
    """

    tf.keras.backend.clear_session()

    # Two optimizers one for the generator and of for the discriminator
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []

    train_loss = pre_test_step_GAN(generator=generator, test_data=pre_train_dataset, batch_size=batch_size)
    train_losses.append(float(train_loss))

    test_loss = pre_test_step_GAN(generator=generator, test_data=pre_test_dataset, batch_size=batch_size)
    test_losses.append(float(test_loss))


    for epoch in range(num_epochs):

        # Training and computing running average
        start = time.time()
        running_average = 0

        with tqdm(total=len(pre_train_dataset)) as pbar:
            for input, target in pre_train_dataset:

                loss = pre_step_GAN(generator=generator, input=input, target=target, optimizer=optimizer, batch_size=batch_size)
                running_average = running_average_factor * running_average + (1 - running_average_factor) * loss
                pbar.update(1)

        train_losses.append(float(running_average))

        # Testing
        test_loss = pre_test_step_GAN(generator=generator, test_data=pre_test_dataset, batch_size=batch_size)
        test_losses.append(float(test_loss))

        # Print useful information
        clear_output()
        print(f'Epoch: {epoch+1}')      
        print()
        print(f'This epoch took {timing(start)} seconds')
        print()
        print(f'The current generator train loss: {round(train_losses[-1], 4)}')
        print()
        print(f'The current generator test loss: {round(test_losses[-1], 4)}')
        print()


        if (epoch+1) % save_every == 0:
            generator.save_weights(f"{save_path}/GSGAN_pre_training_epoch{epoch+1}")


        print()
        sent = generator.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        sent = generator.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        print()


        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
        ax1.plot(train_losses, label='training')
        ax1.plot(test_losses, label='test')
        ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {epoch+1} epochs')
        ax1.legend()

        if (epoch+1) % save_every == 0:
            plt.savefig(f"{save_path}/GSGAN_pre_training_loss_plot_epoch{epoch+1}_transparent", dpi=500.0, format="png", transparent=True)
            plt.savefig(f"{save_path}/GSGAN_pre_training_loss_plot_epoch{epoch+1}", dpi=500.0, format="png")

        plt.show()
        


# Post-training function
def train_step_GAN(generator, discriminator, vocab_size, start_idx, train_data, optimizer_generator, optimizer_discriminator):
    """Perform a training step for the GSGAN
    1. Create a batch of fake sentences using the Generator 
    2. One-hot encode the real sentences for the Discriminator
    3. Feeding the fake and real sentences through the Discriminator 
    4. Calculating the loss for the Discriminator and the Generator 
    5. Performing Backpropagation and Updating the trainable variables with the calculated gradients, using the specified optimizers

    Arguments:
        generator (Generator): Generator class instance
        discriminator (Discriminator): Discriminator class instance
        vocab_size (int): Size of the vocab, needed for one-hot encoding the targets
        start_idx (int): Index of the start token used for generating a batch 
        train_data (tf.data.Dataset): Real tweet embedding from Encoder
        optimizer_generator (tf.keras.optimizers): function from keras defining the to be applied optimizer during training
        optimizer_discriminator (tf.keras.optimizers): function from keras defining the to be applied optimizer during training
    
    Returns:
        loss_from_generator, loss_from_discriminator (Tupel): Tupel containing the loss of both the Generator and Discriminator
    """

    # Two Gradient Tapes, one for the Discriminator and one for the Generator 
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        # 1.
        _, generated_sentences = generator.generate_batch(vocab_size=vocab_size, start_idx=start_idx, batch_size=tf.shape(train_data)[0], max_seq_length=tf.shape(train_data)[1])

        # 2. one-hot encode the targets
        train_data = tf.one_hot(train_data, vocab_size)

        # 3.
        real = discriminator(train_data)
        fake = discriminator(generated_sentences)

        # 4.
        loss_from_generator = generator_loss(fake)
        loss_from_discriminator = discriminator_loss(real, fake)

    
    # 5.
    gradients_from_discriminator = discriminator_tape.gradient(loss_from_discriminator, discriminator.trainable_variables)
    optimizer_discriminator.apply_gradients(zip(gradients_from_discriminator, discriminator.trainable_variables))

    gradients_from_generator = generator_tape.gradient(loss_from_generator, generator.trainable_variables)
    optimizer_generator.apply_gradients(zip(gradients_from_generator, generator.trainable_variables))

    return loss_from_generator, loss_from_discriminator



# Post-training loop
def train_GAN(generator, discriminator, word2vec_model, start_token: int, end_token: int, max_seq_length: int, vocab_size: int, save_every: int, save_path: str, train_dataset_GAN: tf.data.Dataset, num_epochs: int=20, running_average_factor: float=0.95, learning_rate: float=0.00001):
    """Function that implements the training algorithm for a GAN.

    Arguments:
        generator (Generator): Generator class instance
        discriminator (Discriminator): Discriminator class instance
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximal number of tokens to generate per sentence
        vocab_size (int): Size of the vocabulary needed for the train_step
        save_every (int): Determines the amount of epochs before saving weights and plots
        save_path (str): Path to save the weights and images to
        train_dataset_GAN (tf.data.Dataset): Dataset to perform training on
        num_epochs (int): Defines the amount of epochs the training is performed
        running_average_factor (float): To be used factor for computing the running average of the trainings loss
        learning_rate (float): To be used learning rate
    """ 

    tf.keras.backend.clear_session()

    # Two optimizers one for the generator and of for the discriminator
    optimizer_generator=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_discriminator=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Initialize lists for later visualization.
    train_losses_generator = []
    train_losses_discriminator = []

    tau_param = num_epochs//4

    for epoch in range(num_epochs):

        start = time.time()
        running_average_gen = 0
        running_average_disc = 0

        with tqdm(total=len(train_dataset_GAN)) as pbar:
            for batch_no, target in enumerate(train_dataset_GAN):

                gen_loss, disc_loss = train_step_GAN(generator=generator, 
                                                     discriminator=discriminator, 
                                                     vocab_size=vocab_size, 
                                                     start_idx=start_token, 
                                                     train_data=target, 
                                                     optimizer_generator=optimizer_generator, 
                                                     optimizer_discriminator=optimizer_discriminator)
                
                running_average_gen = running_average_factor * running_average_gen + (1 - running_average_factor) * gen_loss
                running_average_disc = running_average_factor * running_average_disc + (1 - running_average_factor) * disc_loss
                pbar.update(1)

        train_losses_generator.append(float(running_average_gen))
        train_losses_discriminator.append(float(running_average_disc))

        # Anneal tau down to one
        if epoch<=tau_param:
            generator.tau = 5.0 ** ((tau_param - epoch) / tau_param)
        else:
            generator.tau = 1.0

        if (epoch+1) % save_every == 0:
            generator.save_weights(f"{save_path}/GSGAN_generator_epoch{epoch+1}")
            discriminator.save_weights(f"{save_path}/GSGAN_discriminator_epoch{epoch+1}")

        # Print useful information
        clear_output()
        print(f'Epoch: {epoch+1}')      
        print()
        print(f'This epoch took {timing(start)} seconds')
        print()
        print(f'The current generator loss: {round(train_losses_generator[-1], 4)}')
        print()
        print(f'The current discriminator loss: {round(train_losses_discriminator[-1], 4)}')
        print()
       
        print("Example Sentences")
        sent = generator.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        sent = generator.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        sent = generator.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        sent = generator.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length)
        print(" ".join([word2vec_model.wv.index2word[i.numpy()[0]] for i in sent]))
        print()

        fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
        ax1.plot(train_losses_generator, label='Generator')
        ax1.plot(train_losses_discriminator, label='Discriminator')
        ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {epoch+1} epochs')
        ax1.legend()

        if (epoch+1) % save_every == 0:
            plt.savefig(f"{save_path}/GSGAN_loss_plot_epoch{epoch+1}_transparent", dpi=500.0, format="png", transparent=True)
            plt.savefig(f"{save_path}/GSGAM_loss_plot_epoch{epoch+1}", dpi=500.0, format="png")

        plt.show()