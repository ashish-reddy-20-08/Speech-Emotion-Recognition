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


## Feature Extraction
The next step involves extracting the features from the audio files which will help our model learn between these audio files.
For feature extraction we make use of the [**LibROSA**](https://librosa.github.io/librosa/) library in python which is one of the libraries used for audio analysis. 

![image](https://user-images.githubusercontent.com/63282184/160747342-912fd7e7-cce7-46ee-8530-219f2bd98e69.png)

**The extracted features looks as follows**

<br>

![image](https://user-images.githubusercontent.com/63282184/160747464-1e3de7be-e1de-4d1c-bbcf-154dc63bd4df.png)

<br>

These are array of values with lables appended to them. 

## Building Models


## Predictions

After tuning the model, tested it out by predicting the emotions for the test data. For a model with the given accuracy these are a sample of the actual vs predicted values.
<br>
<br>
![](images/predict.png?raw=true)
<br>

## Testing out with live voices.
In order to test out our model on voices that were completely different than what we have in our training and test data, we recorded our own voices with dfferent emotions and predicted the outcomes. You can see the results below:
The audio contained a male voice which said **"This coffee sucks"** in a angry tone.
<br>
![](images/livevoice.PNG?raw=true)
<br>
<br>
![](images/livevoice2.PNG?raw=true)
<br>

### As you can see that the model has predicted the male voice and emotion very accurately in the image above.

## NOTE: If you are using the model directly and want to decode the output ranging from 0 to 9 then the following list will help you.

0 - female_angry <br>
1 - female_calm <br>
2 - female_fearful <br>
3 - female_happy <br>
4 - female_sad <br>
5 - male_angry <br>
6 - male_calm <br>
7 - male_fearful <br>
8 - male_happy <br>
9 - male_sad <br>

## Conclusion
Building the model was a challenging task as it involved lot of trail and error methods, tuning etc. The model is very well trained to distinguish between male and female voices and it distinguishes with 100% accuracy. The model was tuned to detect emotions with more than 70% accuracy. Accuracy can be increased by including more audio files for training.
