# Visualizing intermediate activation in Convolutional Neural Networks with Keras
'''
https://github.com/gabrielpierobon/cnnshapes
Visualizing intermediate activations in Convolutional Neural Networks
In this article we're going to train a simple Convolutional Neural Network using Keras in Python for a classification task. For that we are going to use a very small and simple set of images consisting of 100 pictures of circles, 100 pictures of squares and 100 pictures of triangles which I took from Kaggle (https://www.kaggle.com/dawgwelder/keras-cnn-build/data). These will be split into training and testing sets (folders in working directory) and fed to the network.

Most importantly, we are going to replicate some of the work of François Chollet in his book Deep Learning with Python in order to learn how our layer structure processes the data in terms of visualization of each intermediate activation, which consists of displaying the feature maps that are output by the convolution and pooling layers in the network.

We'll go super fast since we are not focusing here on doing a full tutorial of CNNs with Keras but just this simple findings on how CNNs work.

Let's first import all our required libraries:
'''
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio as im
import sys, getopt
from keras import models
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image as kimage
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#path to data sets
train_path = 'Whole/train'
validation_path = 'Whole/validation'
test_path = 'Whole/test'

def display_sample_training_images():
    #Display some of our training images:
    '''
    ## HIGH
    '''
    columns = 5
    rows = 5
    sample_size = 10
    images = []
    for img_path in glob.glob(train_path + '/HIGH/*.jpg'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.title('Sample training images')
    columns = 5
    rows = 5
    for i, image in enumerate(images[:sample_size]):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
    plt.show()
    '''
    ## LOW
    '''
    images = []
    for img_path in glob.glob(train_path + '/LOW/*.jpg'):
        images.append(mpimg.imread(img_path))

    plt.figure(figsize=(20,10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.title('Sample validation images')
    for i, image in enumerate(images[:sample_size]):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image)
    plt.show()

def read_image(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    return tmp
def read_image_1(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    tmp /= 255.
    #tmp = preprocess_input(tmp)
    return tmp

def read_and_stack_image(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    images = np.vstack([tmp])
    return images

def read_image_fortensor(path):
    img = kimage.load_img(path, target_size=(128, 128))
    tmp = kimage.img_to_array(img)
    tmp = np.expand_dims(tmp, axis=0)
    tmp /= 255.
    #tmp = preprocess_input(tmp)
    return tmp

def predict_an_image(file, count, classifier):
    class_labels = {0:'HIGH', 1: 'LOW'}
    image_tensor = read_image_fortensor(file)
    image_stack = read_and_stack_image(file)
    classes = classifier.predict_classes(image_stack, batch_size=10)

    print("Display file:", file)
    print("Predicted class is:", classes, ' which is ', class_labels.get(classes[0]))

    plt.imshow(image_tensor[0])
    plt.title(file + ":" + class_labels.get(classes[0]))
    plt.show()

    if count==0:
        display_activation_layers(classifier, image_tensor)

def predict_images(classifier):
    #predit images in a directory
    count = 0
    for file in glob.glob(test_path + "/*.jpg"):
        predict_an_image(file, count, classifier)
        count =+ 1

##display prediction of\ images from test directory
##batch_size is the max number of images to be displayed
def predict_images_to_check_later(classifier):
    #predit images in a directory
    class_labels = {0:'HIGH', 1: 'LOW'}
    batch_size = 3  #max number of images
    batch_holder = np.zeros((batch_size, 128, 128, 3))
    image_paths = []
    tensor_holder = []

    j = 0
    for file in glob.glob(test_path + "/*.jpg"):
        batch_holder[j, :] = read_image(file)
        tensor_holder.append(read_image_1(file))
        image_paths.append(file)
        j =+ 1

    print("len batch_holder=", len(batch_holder))
    #classes = classifier.predict_classes(batch_holder, batch_size=10)
    classes = classifier.predict_classes(batch_holder)
    print("len predicted classes=", len(classes))
    print("predicted classes=", classes)
    classnum = len(classes)
    for k in range(classnum):
        print("Predicted class is:", k," ",classes[k], ' which is ', class_labels.get(classes[k]))
        print("Display img_tensor[0]")
        plt.imshow(tensor_holder[k][0])
        plt.title(image_paths[k] + ":" + class_labels.get(classes[k]))
        plt.show()

    #Display layers of the first image of the test batch
    #display_activation_layers(classifier, tensor_holder)

def display_activation_layers(classifier, img_tensor):

    '''
    ## Visualizing intermediate activations
    Quoting François Chollet in his book "DEEP LEARNING with Python" (and I'll quote him a lot in this section):

    Intermediate activations are "useful for understanding how successive convnet layers transform their input, and for getting a first idea of the meaning of individual convnet filters."

    "The representations learned by convnets are highly amenable to visualization, in large part because they’re representations of visual concepts. Visualizing intermediate activations consists of displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input (the output of a layer is often called its activation, the output of the activation function). This gives a view into how an input is decomposed into the different filters learned by the network. Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plotting the contents of every channel as a 2D image."

    Next, we’ll get an input image—from test set, not part of the images the network was trained on.

    "In order to extract the feature maps we want to look at, we’ll create a Keras model that takes batches of images as input, and outputs the activations of all convolution and pooling layers. To do this, we’ll use the Keras class Model. A model is instantiated using two arguments: an input tensor (or list of input tensors) and an output tensor (or list of output tensors). The resulting class is a Keras model, just like the Sequential models, mapping the specified inputs to the specified outputs. What sets the Model class apart is that it allows for models with multiple outputs, unlike Sequential."

    Display layers of the first image of the test batch
    ## Instantiating a model from an input tensor and a list of output tensors
    '''
    layer_outputs = [layer.output for layer in classifier.layers[:12]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    '''
    When fed an image input, this model returns the values of the layer activations in the original model

    ## Running the model in predict mode
    '''
    activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
    '''
    For instance, this is the activation of the first convolution layer for the image input:
    '''
    first_layer_activation = activations[0]
    print("first_layer_activation.shape=", first_layer_activation.shape)
    '''
    (1, 28, 28, 32)
    It’s a 28 × 28 feature map with 32 channels.
    Let’s try plotting the fourth channel of the activation of the first layer of the original model
    '''
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    plt.show()

    '''
    Even before we try to interpret this activation, let's instead plot all the activations of this same image across each layer
    ## Visualizing every channel in every intermediate activation
    '''
    layer_names = []
    for layer in classifier.layers[:12]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                if (channel_image.std() != 0):
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


def display_performance_graphs(history):
    ## Displaying curves of loss and accuracy during training
    #Let's now inspect how our model performed over the 30 epochs:

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy\n Epochs: '+str(num_epochs)+'   Steps per epochs:'+str(steps_per_epoch))
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss\n Epochs: '+str(num_epochs)+'   Steps per epochs:'+str(steps_per_epoch))
    plt.legend()

    plt.show()

######################################
##main program starts
##
##
######################################
#Display some of our training images:
def main(argv):

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('python cnnMedJ5 test')
        sys.exit(2)
    # print("opts=",opts)
    # print("argv=",argv)
    #print("argv[0]=",argv[0])

    if argv and argv[0]=='test':
        classifier = Sequential()
        classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (128, 128, 3), activation = 'relu'))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5)) # antes era 0.25

        # Adding a second convolutional layer
        classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5)) # antes era 0.25

        # Adding a third convolutional layer
        classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5)) # antes era 0.25

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(units = 512, activation = 'relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units = 2, activation = 'softmax'))

        ## Load our classifier with the weights of the best model
        #Now we can load those weights as our final model:
        classifier.load_weights('best_weights.hdf5')
        #predit images test directory
        predict_images(classifier)
    else:
        #Display some of our training images:
        '''
        ## HIGH
        '''
        columns = 5
        rows = 5
        sample_size = 10
        images = []
        # display_sample_training_images()

        '''
        (128, 128, 3)
        Images shapes are of 128 pixels by 128 pixels in RGB scale.
        Let's now proceed with our Convolutional Neural Network construction. As usually, we initiate the model with Sequential():
        '''
        # Initialising the CNN
        classifier = Sequential()

        '''
        We specify our convolution layers and add MaxPooling to downsample and Dropout to prevent overfitting. We use Flatten and end with a Dense layer of 2 units, one for each class (HIGH [0], LOW [1]). We specify softmax as our last activation function, which is suggested for multiclass classification.
        '''

        # Step 1 - Convolution
        train_classes = 2 #(high, low)

        classifier.add(Conv2D(32, (3, 3), padding='same', input_shape = (128, 128, 3), activation = 'relu'))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5)) # antes era 0.25

        # Adding a second convolutional layer
        classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5)) # antes era 0.25

        # Adding a third convolutional layer
        classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
        classifier.add(Conv2D(64, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.5)) # antes era 0.25

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(units = 512, activation = 'relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(units = train_classes, activation = 'softmax'))
        '''
        Display classifer's layer summary
        '''
        print("Model summary:")
        classifier.summary()

        # Compiling the CNN
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.02, amsgrad=False)
        classifier.compile(optimizer = adam, #'rmsprop',
                           loss = 'categorical_crossentropy',
                           metrics = ['accuracy'])
        #decay originally 0.01
        '''

        ## Using ImageDataGenerator to read images from directories
        At this point we need to convert our pictures to a shape that the model will accept. For that we use the ImageDataGenerator. We initiate it and feed our images with .flow_from_directory. There are two main folders inside the working directory, called training_set and validation_set. Each of those have 2 subfolders called high and low. I have sent 80% of total images of each shape to the training_set and 20% to the validation_set.

        '''

        train_datagen = ImageDataGenerator(rescale = 1./255)
        test_datagen = ImageDataGenerator(rescale = 1./255)

        training_set = train_datagen.flow_from_directory(train_path,
                                                         target_size = (128, 128),
                                                         batch_size = 16,
                                                         class_mode = 'categorical')

        validation_set = test_datagen.flow_from_directory(validation_path,
                                                          target_size = (128, 128),
                                                          batch_size = 16,
                                                          class_mode = 'categorical')
        '''

        ## Utilize callback to store the weights of the best model
        The model will train for 20 epochs but we will use ModelCheckpoint to store the weights of the best performing epoch. We will specify val_acc as the metric to use to define the best model. This means we will keep the weights of the epoch that scores highest in terms of accuracy on the test set.

        '''
        checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                                       monitor = 'val_acc',
                                       verbose=1,
                                       save_best_only=True)
        '''
        Now it's time to train the model, here we include the callback to our checkpointer
        '''
        number_steps_per_epoch = 20 #100
        number_epoch = 10 #20
        history = classifier.fit_generator(training_set,
                                           steps_per_epoch = number_steps_per_epoch,
                                           epochs = number_epoch,
                                           callbacks=[checkpointer],
                                           validation_data = validation_set,
                                           validation_steps = 50)
        '''
        The model trained for 20 epochs but reached it's best performance at epoch 10. You will notice the message that says:  Epoch 00010: val_acc improved from 0.93333 to 0.95556, saving model to best_weights.hdf5

        That means we have now an hdf5 file which stores the weights of that specific epoch, where the accuracy over the test set was of 95,6%

        ## Load our classifier with the weights of the best model
        Now we can load those weights as our final model:
        '''
        classifier.load_weights('best_weights.hdf5')
        '''
        ## Saving the complete model
        '''
        #classifier.save('shapes_cnn.h5')
        '''

        ## Displaying curves of loss and accuracy during training
        Let's now inspect how our model performed over the 30 epochs:
        '''
        #display_performance_graphs(history)
        '''
        ## Classes
        Let's clarify now the class number assigned to each of our figures set, since that is how the model will produce it's predictions:
        high: 0
        low: 1

        ## Predicting new images
        With our model trained and stored, we can load a simple unseen image from our test set and see how it is classified:
        '''
        #predit images test directory
        predict_images(classifier)

        '''
        So here it is! Let's try to interpret what's going on:

        * The first layer acts is arguably retaining the full shape of the image, although there are several filters that are not activated and are left blank. At that stage, the activations retain almost all of the information present in the initial picture.
        * As we go deeper in the layers, the activations become increasingly abstract and less visually interpretable. They begin to encode higher-level concepts such as single borders, corners and angles. Higher presentations carry increasingly less information about the visual contents of the image, and increasingly more information related to the class of the image.
        * As mentioned above, the model structure is overly complex to the point where we can see our last layers actually not activating at all, there's nothing more to learn at that point.

    '''

if __name__ == "__main__":
   main(sys.argv[1:])
