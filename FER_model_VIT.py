# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:17:26 2023

@author: swava
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:28:46 2023

@author: swava
"""

import os,cv2
import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_data_format=='th'
import tensorflow.keras
#K.set_image_dim_ordering('tf')
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import SGD,RMSprop,adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
#SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import itertools

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.lite.python import lite
from tensorflow.python.client import device_lib
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(device_lib.list_local_devices())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# num_classes = 5
input_shape = (224, 224, 3)



# # set the paths for the train and test folders
# train_path = r'D:\Swavaf\FER_COMPARISON\FER_CNN\FaceExpression\affectnet\train'
# test_path = r'D:\Swavaf\FER_COMPARISON\FER_CNN\FaceExpression\affectnet\test'

# # load the train data
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#         train_path,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='sparse')

# # load the test data
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#         test_path,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='sparse')

# # print the shape of the train and test data
# x_train, y_train = train_generator.next()
# x_test, y_test = test_generator.next()
# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


PATH = os.getcwd()
# Define data path
#data_path =  r'/projects/FaceEmotionComoarison/affectnet/train'
data_path =  r'C:\Users\swava\HeritageProject\train'

data_dir_list = os.listdir(data_path)

img_rows=224
img_cols=224
num_channel=3
num_epoch= 30

# Define the number of classes
num_classes = 2

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(224, 224))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.set_image_data_format=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.set_image_data_format=='tf':
		#img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)

USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
	# using sklearn for preprocessing
	from sklearn import preprocessing
	
	def image_to_feature_vector(image, size=(224, 224)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()
	
	img_data_list=[]
	for dataset in data_dir_list:
		img_list=os.listdir(data_path+'/'+ dataset)
		print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
		for img in img_list:
			input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
			#input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
			input_img_flatten=image_to_feature_vector(input_img,(224, 224))
			img_data_list.append(input_img_flatten)
	
	img_data = np.array(img_data_list)
	img_data = img_data.astype('float32')
	print (img_data.shape)
	img_data_scaled = preprocessing.scale(img_data)
	print (img_data_scaled.shape)
	
	print (np.mean(img_data_scaled))
	print (np.std(img_data_scaled))
	
	print (img_data_scaled.mean(axis=0))
	print (img_data_scaled.std(axis=0))
	
	if K.set_image_data_format=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)
	
	
	if K.set_image_data_format=='th':
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
		print (img_data_scaled.shape)
		
	else:
		img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
		print (img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
	img_data=img_data_scaled

# Assigning Labels

# Define the number of classes
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')


labels[0:4650]=0
labels[4650:9300]=1
	  

names = ['crack','noncrack']


	  
# convert class labels to on-hot encoding
Y = to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = (img_data,Y)
# Split the dataset

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100
image_size = 224  # We'll resize input images to this size
patch_size = 14  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 5
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

"""
## Use data augmentation
"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

"""
## Compute the mean and the variance of the training data for normalization.
"""

data_augmentation.layers[0].adapt(x_train)

"""
## Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation as a layer
"""



class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches



"""
## Implement the patch encoding layer
"""



class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


"""
## Build the ViT model
We put together the Vision Transformer model.
"""


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # logits = layers.Dense(num_classes, activation='softmax')(features)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model


"""
## Compile, train, and evaluate the mode
"""

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    accuracy = keras.metrics.CategoricalAccuracy(name='categorical_accuracy')

    #model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='rmsprop',metrics=[accuracy])

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    	metrics=[accuracy],

        #accuracy=keras.metrics.CategoricalAccuracy(name='categorical_accuracy', dtype=None),
        #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #metrics=[
            #keras.metrics.CategoricalAccuracy()(name='categorical_accuracy', dtype=None),
            # keras.metrics.(5, name="top-5-accuracy"),
            # keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        #],

    )

    #checkpoint_filepath = "./tmp/checkpoint"
    #checkpoint_callback = keras.callbacks.ModelCheckpoint(
        #checkpoint_filepath,
        #monitor="val_accuracy",
        #save_best_only=True,
        #save_weights_only=True,
    #)

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        #callbacks=[checkpoint_callback],
    )

    """
    ##Let's display the final results of the training on CIFAR-100.
    """

    #model.load_weights(checkpoint_filepath)
    #_, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    #print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    """
    ##Let's visualize the training progress of the model.
    """

    # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.figure(1,figsize=(7,5))
    # plt.plot(history.history['categorical_accuracy'])
    # plt.plot(history.history['val_categorical_accuracy'])
    # plt.title('Train and Validation Accuracy Over Epochs (VIT)')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # #plt.show()
    # plt.savefig('accGraph.png', bbox_inches='tight')


    # # summarize history for loss
    # plt.figure(2,figsize=(7,5))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Train and Validation Losses Over Epochs (VIT)')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # #plt.show()
    # plt.savefig('lossGraph.png', bbox_inches='tight')


    # visualizing losses and accuracy
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    train_acc=history.history['categorical_accuracy']
    val_acc=history.history['val_categorical_accuracy']
    xc=range(100)
    import numpy as np, array
    aa = np.asarray(train_loss)
    np.savetxt("train_loss.csv", aa, delimiter=",")
    bb = np.asarray(val_loss)
    np.savetxt("val_loss.csv", bb, delimiter=",")
    cc = np.asarray(train_acc)
    np.savetxt("train_acc.csv", cc, delimiter=",")
    dd = np.asarray(val_acc)
    np.savetxt("val_acc.csv", dd, delimiter=",")
    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    # plt.style.use(['classic'])
    plt.savefig('lossGraph.png', bbox_inches='tight')


    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    # plt.style.use(['classic'])
    plt.savefig('accGraph.png', bbox_inches='tight')

    from sklearn.metrics import classification_report,confusion_matrix

    Y_pred = model.predict(x_test)
    print(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)
    print(y_pred)
    #y_pred = model.predict_classes(x_test)
    #print(y_pred)
    target_names = ['class0(crack)','class1(noncrack)']


					
    print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

    print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
    
    cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

    np.set_printoptions(precision=2)

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
    
    plt.savefig('confGraph.png', bbox_inches='tight')


    return history

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix


# Plot non-normalized confusion matrix

#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
# plt.show()



    # model.save('model_file.h5')

    # Convert the model.
    #converter = lite.TFLiteConverter.from_keras_model(model)
    #tflite_model = converter.convert()

    # Save the model.
    #with open('model.tflite', 'wb') as f:
        #f.write(tflite_model)

    

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


