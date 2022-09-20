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


# AutoEncoder classes
class Encoder(Model):
    """Encoder architecture used for the AE
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize the Encoder that creates an embedding of sentences

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the size of the LSTM
        """ 

        super(Encoder, self).__init__()

        self.embedding = tfkl.Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[embedding_matrix], trainable=True)
        self.lstm = tfkl.LSTM(units=hidden_size)


    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training: bool=True):
        """Activate our Encoder propagating the input through it layer by layer

        Arguments:
            x (tensor): Tensor containing the input to our Encoder
            training (bool): Indicates whether regularization methods should be used or not when calling the Encoder 

        Returns:
            hs (tensor): Tensor containing the last hidden state of the lstm projected to the state size of the decoder
        """

        x = self.embedding(x)
        hs = self.lstm(x, training=training)

        return hs



class Decoder(Model):
    """Decoder architecture used for the AE
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize the Decoder that recreates sentences based on the embeddings of the Encoder

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer, as well as the output dimensionality of the readout layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the size of the LSTM
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
        all_hs, hs, cs = self.lstm(x, initial_state=states, training=training)
        dense_out = self.dense(all_hs, training=training)
    	
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
            states (tensor): Tensor containing the initial hidden state for the LSTM 
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



class AutoEncoder(Model):
    """AE class that combines the previous Encoder and Decoder 
    """

    def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=256, hidden_size: int=1024):
        """Initialize an Autoencoder consisting of an Encoder and Decoder

        Arguments:
            vocab_size (int): Defines the input dimensionality of the embedding layer
            embedding_matrix (ndarray): Contains the weights to initialize the embedding layer
            embedding_size (int): Defines the output dimensionality of the embedding layer
            hidden_size (int): Defines the size of the LSTM
        """  

        super(AutoEncoder, self).__init__()

        self.Encoder = Encoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size, hidden_size=hidden_size)
        self.Decoder = Decoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size, hidden_size=hidden_size)

   
    def call(self, input, teacher, training: bool=True):
        """Activate our Autoencoder propagating the input through the Encoder and Decoder respectively

        Arguments:
            input (tensor): Tensor containing the input to the Encoder
            teacher (tensor): Tensor containing the input to the Decoder
            training (bool): Indicates whether regularization methods should be used or not when calling the Autoencoder 

        Returns:
            predictions (tensor): Tensor containing the reconstructed input
        """

        hs = self.Encoder(input, training=training)
        predictions = self.Decoder(teacher, states=[hs, tf.zeros_like(hs)], training=training)
        
        return predictions
        


# Autoencoder Training functions
@tf.function(experimental_relax_shapes=True)
def train_step_ae(model, input, target, teacher, loss_function, optimizer):
    """Perform a training step for the AE
    1. Propagating the input through the network
    2. Calculating the loss between the networks output and the true targets
    3. Performing Backpropagation and Updating the trainable variables witht the calculated gradients 
    
    Arguments:
        model (AutoEncoder): Given instance of an initialised AE with all its parameters
        input (tensor): Tensor containing the input data for the encoder
        target (tensor): Tensor containing the respective targets 
        teacher (tensor): Tensor containing the input data for the decoder
        loss_function (keras.losses): Function from keras to calculate the loss
        optimizer (keras.optimizers): Function from keras defining the to be applied optimizer during learning 
    
    Returns:
        loss (tensor): Tensor containing the loss of the network 
    """

    with tf.GradientTape() as tape:
        # 1.
        prediction = model(input, teacher)
        # 2.
        loss = loss_function(target, prediction)
    # 3.
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss



def test_ae(model, test_data, loss_function, training: bool=False):
    """Tests the models loss over the given data set with a given loss_function
    
    Arguments:
        model (AutoEncoder): Given instance of an initialised AE with all its parameters
        test_data (Dataset): Test dataset to test the AE on 
        loss_function (keras.losses): function from keras to calculate the loss
        training (bool): Indicates whether regularization methods should be used or not when calling the model  
    
    Returns:
        test_loss (float): Average loss of the Network over the test set
    """

    test_loss_aggregator = []
    
    for input, target, teacher in test_data:
        prediction = model(input, teacher, training=training)
        sample_test_loss = loss_function(target, prediction)
        test_loss_aggregator.append(sample_test_loss)
    
    test_loss = tf.reduce_mean(test_loss_aggregator)
    
    return test_loss



def visualize_AE(word2vec_model, train_losses, test_losses, input_sent, predicted_sent, num_epochs, save_path: str, save_plots: bool=False): 
    """Visualize performance and loss for training and test data. 
    
    Arguments:
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        train_losses (list): List containing the training losses of the Network
        test_losses (list): List containing the losses of the Network over the test data
        input_sent (tuple): Tuple containing the to be Autoencoded sentences 
        predicted_sent (tuple): Tuple containing the reconstructed sentences for visualizing the progress of the Network 
        num_epochs (int): Elapsed number of epochs
        save_path (str): Path to save the plots to 
        save_plots (bool): Determines whether to save the plots or just plot them
    """ 

    # We use the inbuilt index2word from the word2vec model to convert the lists of indices back into their respective tokens
    print("Autoencoded Sentence (Training Sample):")
    # Minus 1 since we artificially inserted a 0 column into the embedding matrix (for padding)
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[0]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[0], axis=2).numpy()[0]])}")
    print()
    print("Autoencoded Sentence (Training Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[1]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[1], axis=2).numpy()[0]])}")
    print()
    
    print("Autoencoded Sentence (Test Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[2]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[2], axis=2).numpy()[0]])}")
    print()
    print("Autoencoded Sentence (Test Sample):")
    print(f"Input: {' '.join([word2vec_model.wv.index2word[i] for i in input_sent[3]])}")
    print(f"Output: {' '.join([word2vec_model.wv.index2word[i] for i in tf.argmax(predicted_sent[3], axis=2).numpy()[0]])}")
    print()
    print()


    # Plot for visualizing the average loss over the training and test data
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
    ax1.plot(train_losses, label='training')
    ax1.plot(test_losses, label='test')
    ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
    ax1.legend()

    if save_plots:
        plt.savefig(f"{save_path}/AE_loss_plot_epoch{num_epochs}_transparent", dpi=500.0, format="png", transparent=True)
        plt.savefig(f"{save_path}/AE_loss_plot_epoch{num_epochs}", dpi=500.0, format="png")

    plt.show()



# AE Training loop
def train_AE(model, word2vec_model, save_every: int, save_path: str, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, loss_function: tf.keras.losses, num_epochs: int=20, learning_rate: float=0.001, running_average_factor: float=0.95): 
    """Function that implements the training algorithm for the AE.
    Prints out useful information and visualizations per epoch.

    Arguments:
        model (AutoEncoder): Model that the training algorithm should be applied to
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
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

    # Extract two fixed sentence from the training and test dataset each for visualization during training.
    for input, target, teacher in train_dataset.take(1):
        train_sent_for_visualisation_1 = (input[0], teacher[0])
        train_sent_for_visualisation_2 = (input[1], teacher[1])
        
    for input, target, teacher in test_dataset.take(1):
        test_sent_for_visualisation_1 = (input[0], teacher[0])
        test_sent_for_visualisation_2 = (input[1], teacher[1])
        
    # Initialize the optimizer: Adam with custom learning rate.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    
    # testing once before we begin on the test and train data
    test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function)
    test_losses.append(test_loss)

    train_loss = test_ae(model=model, test_data=train_dataset, loss_function=loss_function, training=True)
    train_losses.append(train_loss)


    for epoch in range(num_epochs):
        start = time.time()

        # Training and computing running average
        running_average = 0
        pbar = tqdm(total=len(train_dataset))
        for input, target, teacher in train_dataset:
            train_loss = train_step_ae(model=model, input=input, target=target, teacher=teacher, loss_function=loss_function, optimizer=optimizer)
            running_average = running_average_factor * running_average  + (1 - running_average_factor) * train_loss
            pbar.update(1)
        pbar.close()
        
        train_losses.append(running_average)

        # Testing
        test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function)
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
            model.save_weights(f"{save_path}/AE_epoch{epoch+1}")

        # Feed sample sentence through the network for visualization
        train_pred_sent_1 = model(tf.expand_dims(train_sent_for_visualisation_1[0], axis=0), tf.expand_dims(train_sent_for_visualisation_1[1], axis=0))
        train_pred_sent_2 = model(tf.expand_dims(train_sent_for_visualisation_2[0], axis=0), tf.expand_dims(train_sent_for_visualisation_2[1], axis=0))
        test_pred_sent_1 = model(tf.expand_dims(test_sent_for_visualisation_1[0], axis=0), tf.expand_dims(test_sent_for_visualisation_1[1], axis=0), training=False)
        test_pred_sent_2 = model(tf.expand_dims(test_sent_for_visualisation_2[0], axis=0), tf.expand_dims(test_sent_for_visualisation_2[1], axis=0), training=False)

        save_plots = (epoch+1) % save_every == 0

        visualize_AE(word2vec_model,
                    train_losses=train_losses, 
                    test_losses=test_losses, 
                    input_sent=(train_sent_for_visualisation_1[0],train_sent_for_visualisation_2[0], test_sent_for_visualisation_1[0], test_sent_for_visualisation_2[0]), 
                    predicted_sent=(train_pred_sent_1, train_pred_sent_2, test_pred_sent_1, test_pred_sent_2), 
                    num_epochs=epoch+1,
                    save_path=save_path,
                    save_plots=save_plots)

    print()
    model.summary()



# LaTextGANs improved WGAN classes
class ResidualBlock(Model):
    """ResidualBlock class to realize the ResNet architecture employed in both Discriminator and Generator of the LaTextGAN
    """

    def __init__(self, hidden_size: int=1024):
        """Initialize a Residual Block where each layer is of the form 
        F(x) = H(x)+x and H(x) are two relu activated dense layers

        Arguments:
            hidden_size (int): Defines the size of the dense layers
        """

        super(ResidualBlock, self).__init__()

        self.ResBlockLayers = [tfkl.Dense(units=hidden_size, activation="relu"), tfkl.Dense(units=hidden_size, activation=None)]
        
 
    def call(self, x):
        """Activate our ResidualBlock, by propagating the input through it layer by layer.

        Arguments:
            x (tensor): Tensor containing the input to our ResBlock

        Returns:
            x (tensor): Tensor containing the activation of our ResBlock
        """

        y = x

        for layer in self.ResBlockLayers:
            y = layer(y)
            
        return x+y
        


class Generator(Model):
    """Generator architecture used for the LaTextGAN
    """

    def __init__(self, hidden_size: int=1024, num_res_blocks: int=40):
        """Initialize a Generator that generates fake sentence embeddings.

        Arguments:
            hidden_size (int): Defines the size of the dense layers
            num_res_blocks (int): Defines the number of Resblocks that are stacked
        """ 

        super(Generator, self).__init__()

        self.hidden_size = hidden_size

        self.generator_layers = [ResidualBlock(hidden_size=hidden_size) for _ in range(num_res_blocks)]


    def call(self, x): 
        """Activate our Generator propagating the input through it layer by layer.

        Arguments:
            x (tensor): Normal distributed noise  

        Returns:
            x (tensor): Generator output
        """

        for layer in self.generator_layers:
            x = layer(x)

        return x



class Discriminator(Model):
    """Discriminator architecture used for the LaTextGAN
    """

    def __init__(self, hidden_size: int=1024, num_res_blocks: int=40):
        """Initialize a Discriminator that decides whether the input sentence embedding is fake or real 

        Arguments:
            hidden_size (int): Defines the size of the dense layers
            num_res_blocks (int): Defines the number of Resblocks that are stacked
        """ 

        super(Discriminator, self).__init__()


        self.discriminator_layers = [ResidualBlock(hidden_size=hidden_size) for _ in range(num_res_blocks)]
        self.discriminator_layers.append(tfkl.Dense(1, activation=None))


    def call(self, x): 
        """Activates the Discriminator propagating the input through it layer by layer

        Arguments:
            x (tensor): Tensor containing the input
        
        Returns:
            x (tesnor): Tensor containing the decision of the Discriminator
        """

        for layer in self.discriminator_layers:
            x = layer(x)

        return x
    


# Improved WGAN loss functions
def discriminator_loss(real_sent, fake_sent):
    """Calculate the Wasserstein loss for the discriminator but swapping the sign in order to apply gradient descent.

    Arguments:
        real_sent (tensor): Linear output from discriminator
        fake_sent (tensor): Linear output from discriminator

    Returns:
        x (tensor): Wasserstein Loss
    """

    loss_real = - tf.math.reduce_mean(real_sent)
    loss_fake = tf.math.reduce_mean(fake_sent)

    return loss_real + loss_fake



def generator_loss(fake_sent):
    """Calculate the Wasserstein loss for the generator.

    Arguments:
        fake_sent (tensor): Linear output from discriminator

    Returns:
        x (tensor): Wasserstein Loss
    """

    loss_fake = - tf.math.reduce_mean(fake_sent)

    return loss_fake
  


def gradient_penalty(discriminator, real_sent, generated_sent):
    """Calculates the gradient penalty for the improved WGAN 
    
    Arguments:
        discriminator (Discriminator): Discriminator class instance
        real_sent (tensor): Real sentence embedding from Encoder
        generated_sent (tensor): Fake sentence embedding from Generator

    Return: 
        penalty (): Gradient penalty that will be added to discriminator loss
    """ 

    alpha = tf.random.uniform(shape=[real_sent.shape[0], 1], minval=0, maxval=1)

    interpolate = alpha*real_sent + (1-alpha)*generated_sent

    output = discriminator(interpolate)

    gradients = tf.gradients(output, interpolate)

    gradient_norm = tf.sqrt(tf.math.reduce_sum(tf.square(gradients), axis=1))

    penalty = 10*tf.math.reduce_mean((gradient_norm-1.)**2)

    return penalty



# LaTextGAN Training functions
@tf.function()  
def train_step_GAN(generator, discriminator, train_data, optimizer_generator, optimizer_discriminator, train_generator):
    """Perform a training step for the LaTextGAN by
    1. Generating random noise for the Generator
    2. Feeding the noise through the Generator to create fake sentence embeddings for the Discriminator 
    3. Feeding the fake and real sentence embeddings through the Discriminator 
    4. Calculating the loss for the Disriminator and the Generator 
    5. Applying the gradient penalty to enforce lipschitz continuity
    6. Performing Backpropagation and Updating the trainable variables with the calculated gradients, using the specified optimizers

    Arguments:
        generator (Generator): Generator class instance
        discriminator (Discriminator): Discriminator class instance
        train_data (tf.data.Dataset): Real sentence embedding from Encoder
        optimizer_generator (tf.keras.optimizers): function from keras defining the to be applied optimizer during training
        optimizer_discriminator (tf.keras.optimizers): function from keras defining the to be applied optimizer during training
        train_generator (bool): Whether to update the generator or not
    
    Returns:
        loss_from_generator (tensor): Tensor containing the loss of the Generator and Discriminator
        loss_from_discriminator (tensor): Tensor containing the loss of the Discriminator
    """

    # 1.
    noise = tf.random.normal([train_data.shape[0], generator.hidden_size]) 

    # Two Gradient Tapes, one for the Discriminator and one for the Generator 
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        # 2.
        generated_sent = generator(noise)

        # 3.
        real = discriminator(train_data)
        fake = discriminator(generated_sent)

        # 4.
        loss_from_generator = generator_loss(fake)
        
        # 5.
        loss_from_discriminator = discriminator_loss(real, fake) + gradient_penalty(discriminator=discriminator, real_sent=train_data, generated_sent=generated_sent)

    # 6.
    gradients_from_discriminator = discriminator_tape.gradient(loss_from_discriminator, discriminator.trainable_variables)
    optimizer_discriminator.apply_gradients(zip(gradients_from_discriminator, discriminator.trainable_variables))

    # We update the generator once for ten updates to the discriminator
    if train_generator:
        gradients_from_generator = generator_tape.gradient(loss_from_generator, generator.trainable_variables)
        optimizer_generator.apply_gradients(zip(gradients_from_generator, generator.trainable_variables))

    return loss_from_generator, loss_from_discriminator



def visualize_GAN(autoencoder, word2vec_model, start_token, end_token, max_seq_length, fixed_input, random_input, train_losses_generator, train_losses_discriminator, num_epochs, save_path: str, save_plots: bool=False):
    """Visualize performance of the Generator by feeding predefined random noise vectors through it.
    
    Arguments:
        autoencoder (AutoEncoder): AutoEncoder class instance
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximum length of the to-be-generated sentences
        fixed_input (tensor): List containing predefined random vectors
        random_input (tensor): List containing predefined random vectors
        train_losses_generator (list): List containing the generator losses
        train_losses_discriminator (list): List containing the discriminator losses 
        num_epochs (int): Current Epoch
        save_path (str): Path to save the plots to 
        save_plots (bool): Determines whether to save the plots or just plot them
    """ 

    print()
    print(f"From Fixed Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0]] for i in autoencoder.Decoder.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length, states=fixed_input[0])])}")
    print(f"From Fixed Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0]] for i in autoencoder.Decoder.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length, states=fixed_input[1])])}")
    print()
    print(f"From Random Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0]] for i in autoencoder.Decoder.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length, states=random_input[0])])}")
    print(f"From Random Vector: {' '.join([word2vec_model.wv.index2word[i.numpy()[0]] for i in autoencoder.Decoder.inference_mode(start_token=start_token, end_token=end_token, max_seq_length=max_seq_length, states=random_input[1])])}")

    plt.style.use('ggplot')
    
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
    ax1.plot(train_losses_generator, label='Generator')
    ax1.plot(train_losses_discriminator, label='Discriminator')
    ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
    ax1.legend()

    if save_plots:
        plt.savefig(f"{save_path}/LaTextGAN_loss_plot_epoch{num_epochs}_transparent.png", dpi=500.0, format="png", transparent=True)
        plt.savefig(f"{save_path}/LaTextGAN_loss_plot_epoch{num_epochs}.png", dpi=500.0, format="png")

    plt.show()



# LaTextGAN Training loop
def train_GAN(generator, discriminator, autoencoder, word2vec_model, start_token: int, end_token: int, max_seq_length: int, save_every: int, save_path: str, train_dataset_GAN: tf.data.Dataset, gen_update: int=10, num_epochs: int=150, learning_rate: float=0.0001, running_average_factor: float=0.95):
    """Function that implements the training algorithm for LaTextGAN.

    Arguments:
        generator (Generator): Generator class instance
        discriminator (Discriminator): Discriminator class instance
        autoencoder (AutoEncoder): AutoEncoder class instance
        word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
        start_token (int): Index of the start token used when generating
        end_token (int): Index of the end token used when generating
        max_seq_length (int): Maximum length of the to-be-generated sentences
        save_every (int): Determines the amount of epochs before saving weights and plots
        save_path (str): Path to save the weights and images to
        train_dataset_GAN (tf.data.Dataset): Dataset to perform training on
        gen_update (int): Number of steps before the generator is updated
        num_epochs (int): Defines the amount of epochs the training is performed
        learning_rate (float): To be used learning rate
        running_average_factor (float): To be used factor for computing the running average of the trainings loss
    """ 

    tf.keras.backend.clear_session()

    # Two optimizers one for the generator and of for the discriminator
    optimizer_generator=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_discriminator=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Fixed, random vectors for visualization
    fixed_generator_input_1 = tf.random.normal([1, generator.hidden_size]) 
    fixed_generator_input_2 = tf.random.normal([1, generator.hidden_size])

    # Initialize lists for later visualization.
    train_losses_generator = []
    train_losses_discriminator = []

    train_generator = False

    for epoch in range(num_epochs):

        start = time.time()
        running_average_gen = 0
        running_average_disc = 0

        with tqdm(total=len(train_dataset_GAN)) as pbar:
            for batch_no, input in enumerate(train_dataset_GAN):

                # Boolean used to train the discriminator x times more often than the generator
                # determined by the gen_update parameter
                train_generator = False
                if batch_no % gen_update == 0:
                    train_generator = True

                gen_loss, disc_loss = train_step_GAN(generator, 
                                                     discriminator, 
                                                     train_data=input, 
                                                     optimizer_generator=optimizer_generator, 
                                                     optimizer_discriminator=optimizer_discriminator, 
                                                     train_generator=train_generator)
                
                running_average_gen = running_average_factor * running_average_gen + (1 - running_average_factor) * gen_loss
                running_average_disc = running_average_factor * running_average_disc + (1 - running_average_factor) * disc_loss
                pbar.update(1)

        train_losses_generator.append(float(running_average_gen))
        train_losses_discriminator.append(float(running_average_disc))

        clear_output()
        print(f'Epoch: {epoch+1}')      
        print()
        print(f'This epoch took {timing(start)} seconds')
        print()
        print(f'The current generator loss: {round(train_losses_generator[-1], 4)}')
        print()
        print(f'The current discriminator loss: {round(train_losses_discriminator[-1], 4)}')
        print()
        
        if (epoch+1) % save_every == 0:
            generator.save_weights(f"{save_path}/LaTextGAN_epoch{epoch+1}")

        # Random vectors for visualization that are sampled each epoch
        random_generator_input_1 = tf.random.normal([1, generator.hidden_size]) 
        random_generator_input_2 = tf.random.normal([1, generator.hidden_size])
        
        save_plots = (epoch+1) % save_every == 0

        visualize_GAN(autoencoder=autoencoder,
                    word2vec_model=word2vec_model,
                    start_token = start_token,
                    end_token = end_token, 
                    max_seq_length = max_seq_length,
                    fixed_input=(generator(fixed_generator_input_1), generator(fixed_generator_input_2)), 
                    random_input=(generator(random_generator_input_1), generator(random_generator_input_2)), 
                    train_losses_generator=train_losses_generator, 
                    train_losses_discriminator=train_losses_discriminator, 
                    num_epochs=epoch+1,
                    save_path=save_path,
                    save_plots=save_plots)
    
    print()
    model.summary()