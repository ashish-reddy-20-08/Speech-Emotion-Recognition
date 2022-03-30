# Speech-Emotion-Recognition
In this project we take the Ravdess, Crema, Tess dataset and perform the Emotion Recognition. 

# Speech Emotion Analyzer

* The idea behind creating this project was to build a machine learning model that could detect emotions from the speech we have with each other all the time. Nowadays personalization is something that is needed in all the things we experience everyday. 

* So why not have a emotion detector that will guage your emotions and in the future recommend you different things based on your mood. 
This can be used by multiple industries to offer different services like marketing company suggesting you to buy products based on your emotions, automotive industry can detect the persons emotions and adjust the speed of autonomous cars as required to avoid any collisions etc.

## Analyzing audio signals

![image](https://user-images.githubusercontent.com/63282184/160746991-87d22c4f-1f9a-480e-8472-3aa6d1eb8e9e.png)



# STEPS FOLLOWED IN THE PROJECT:

## 1. Importing the required libraries

![image](https://user-images.githubusercontent.com/63282184/160747746-17c94d13-a69e-4eef-aea0-caee6adccfa4.png)

## 2. Data Preparation 

- In this step we load the three datasets that Ravdess, Tess and creampie

## 3. Data Exploration

![image](https://user-images.githubusercontent.com/63282184/160747907-4ae4b619-f90e-48cd-9567-95ef59872e63.png)


### Audio files:
Tested out the audio files by plotting out the waveform and a spectrogram to see the sample audio files.<br>
**Waveform**

![image](https://user-images.githubusercontent.com/63282184/160747195-aad8472b-db63-4284-a0c1-565bfdc6150b.png)

**Spectrogram**<br>

![image](https://user-images.githubusercontent.com/63282184/160747247-051a4b6b-6063-49b7-bec3-8e29b2e92c76.png)

## 4. Data Augmentation

Data augmentation is the process by which we create new synthetic data samples by adding small perturbations on our initial training set.

- Noise Injection
It simply add some random value into data by using numpy.

- Shifting Time
The idea of shifting time is very simple. It just shift audio to left/right with a random second. If shifting audio to left (fast forward) with x seconds, first x seconds will mark as 0 (i.e. silence). If shifting audio to right (back forward) with x seconds, last x seconds will mark as 0 (i.e. silence).

- Changing Pitch
This augmentation is a wrapper of librosa function. It change pitch randomly

- Changing Speed
Same as changing pitch, this augmentation is performed by librosa function. It stretches times series by a fixed rate.

![image](https://user-images.githubusercontent.com/63282184/160748073-4c416834-0b91-477e-b32c-174c0ac8ccc0.png)


## 5. Feature Extraction
The next step involves extracting the features from the audio files which will help our model learn between these audio files.
For feature extraction we make use of the [**LibROSA**](https://librosa.github.io/librosa/) library in python which is one of the libraries used for audio analysis. 

![image](https://user-images.githubusercontent.com/63282184/160747342-912fd7e7-cce7-46ee-8530-219f2bd98e69.png)

**The extracted features looks as follows**

<br>

![image](https://user-images.githubusercontent.com/63282184/160747464-1e3de7be-e1de-4d1c-bbcf-154dc63bd4df.png)

<br>

These are array of values with lables appended to them. 



## 6. Splitting the data 
 - One hot encoding 
 - splitting the data 
 - Scaling our model with sklearn's standard scaler 
 - Making the data compatible to the model. 


## 7. Building Models
- 1. **Convo layer + dense**
- Convolutional layers are the major building blocks used in convolutional neural networks.
- Dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer, thus called as dense. Dense Layer is used to classify image based on output from convolutional layers.

- 2. **LSTM + dense**
- An LSTM layer learns long-term dependencies between time steps in time series and sequence data. The state of the layer consists of the hidden state (also known as the output state) and the cell state. The hidden state at time step t contains the output of the LSTM layer for this time step.

Why dense layer is used after LSTM?

Timedistributed dense layer is used on RNN, including LSTM, to keep one-to-one relations on input and output.
## 8. Predictions

### Loss and accuracy 
![image](https://user-images.githubusercontent.com/63282184/160749529-79a36dc0-c802-417f-8716-871e22c0bf2b.png)

### Sample output of the prediction
![image](https://user-images.githubusercontent.com/63282184/160749576-137acece-d44b-4219-a3b4-74587b9dc9c1.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/63282184/160749644-491ce7e1-48cd-452e-a58d-a0c7abf13d04.png)


## Conclusion
We can see our model is more accurate in predicting surprise, angry emotions and it makes sense also because audio files of these emotions differ to other audio files in a lot of ways like pitch, speed etc..
We overall achieved 61% accuracy on our test data and its decent but we can improve it more by applying more augmentation techniques and using other feature extraction methods.
