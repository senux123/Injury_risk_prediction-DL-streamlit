## **Athlete injury risk prediction using Deep Learning (LSTM) **

This project is an advanced version of my previous machine learning model.  
The previous baseline model focused on static , single-day features and in this version I explored trends in athlete training and recovery data using a Long Short-Term Memory (LSTM) network.

The core idea behind this project is that injury risk is created from progressive workload and recovery trends over time rather than from single events.
This model will analyze trends and output the injury risk category when user input of 10 days is given.


## **Problem overview**

Athletes experience changes in training load, fatigue, heart-rate variability (HRV), and sleep quality across several days and sudden changes in normal data could possibly indicate injury risk.

The model classifies each athlete-day into one of three categories below:

* Low risk  
* Medium risk  
* High risk

The prediction is based on a *10-day rolling sequence* of training and recovery data.


## **Dataset Structure**

The pipeline uses these three datasets containing athlete data.

* new_activity_data.csv : training data (training intensity, duration, heart-rate metrics. etc)  
* new_daily_data.csv : recovery metrics (HRV, fatigue, sleep, etc)  
* new_athlete_data.csv : athlete metadata (age, athlete ID, gender)

These three are combined into a single time based dataset *per athlete per day* in preprocessing.


## **Feature Engineering**

Several steps are taken to create signals, which are:

### **Daily Training Load**

Training sessions are created per athlete per day using load (intensity × duration) and heart-rate load (average HR × duration).

### **Rolling Baselines**

To identify deviations rather than single values:

* A 10-day rolling mean is computed for HRV and fatigue  
* Baselines are shifted to avoid data leakage.

   Obtained features include:

* HRV deviation from baseline.
* Fatigue increase relative to baseline.


## Model Input Features

The final LSTM model uses the following features per day:

- Training load metrics: duration, intensity factor, average heart rate  
- Recovery indicators: HRV, sleep hours, fatigue score  
- Athlete metadata: age  
- Features calculated from the baseline:
  - HRV deviation from rolling baseline (hrv_d)
  - Fatigue increase relative to baseline (fatigue_increase)


## **Construction of the target**

A rule-based system is used for risk levels as below:

* High risk: recorded injury.  
* Medium risk: increased fatigue or HRV drop combined with high workload.  
* Low risk: normal training and recovery patterns.


## **Model Architecture**

The model is built using *PyTorch* and *PyTorch Lightning* with the following:

* LSTM network to get sequential data.  
* configurable hidden size, number of layers, and dropout.  
* Fully connected layer for three class classification.

Each model input has the shape: (batch\_size, 10, 9)


## **Training strategy**

* *Athlete level data split* : (train / validation / test) to prevent data leakage.  
* *StandardScaler* used for feature scaling on training data.  
* *Class imbalance* handling using weighted cross-entropy loss.  
* *Hyperparameter optimization* using Optuna.  
* Using early stopping and checkpointing for more stable training.


## **Saved artifacts**

After training, the following files are created for deployment:

* injury\_lstm\_state\_dict.pt – trained model weights.  
* dl\_config.json – model configuration and metadata.  
* dl\_scaler.pkl – used feature scaler.  
* dl\_feature\_columns.pkl – for placing features in order.

## **Deployment**

The model is designed to be deployed via a streamlit web application.

The user enters *10 consecutive days* of training and recovery data. Then the app:

1. Uses the scaler to scale inputs.  
2. Constructs a 10-day sequence.  
3. Runs inference using the trained LSTM.  
4. Outputs the predicted risk level and confidence.


## **Model performance**

| class | precision | recall | f1-score | support |
| ----- | ----- | ----- | ----- | ----- |
| Low risk | 0.95 | 0.69 | 0.80 | 3417 |
| Medium risk | 0.16 | 0.64 | 0.25 | 228 |
| High risk | 0.45 | 0.74 | 0.56 | 395 |


Note: *Handling class imbalance*:

-All three injury-risk classes are not equally frequent, so I used weighted cross-entropy to highlight minority class mistakes more and make them contribute more to the loss.

-This reduces the model’s chances of over-predicting the majority class and improves macro-F1 and recall for less frequent classes. 

-To make sure weighting didn’t just over-predict the minority class, I checked the predicted class distribution and per-class precision/recall. 

-I also prioritized recall of high risk classes with some false positives as avoiding missing high risk days is more valuable.


##  **Learning Outcomes**

- I learned to build a full ML pipeline where I merged multiple datasets 
- Learned how to prevent data leakage using athlete level splitting and rolling baselines.
- Improved deep learning skills in pyTorch and lightning which include training loops, checkpointing, earlystopping.
- Learned how to improve performance of minor classes using imbalance aware training.


## **Limitations**

- Labels are created using a rule-based system.

- Medium risk is harder to separate because it is between low and high.

- Synthetic dataset helps training and testing, but real-world data would be much noisier.


## **NOTES :**

    \- This project is for educational purposes only and is not medically accurate and should not be used for clinical decisions.

     \- I developed this as part of a personal learning project in applying deep learning into real world problems.

     \- I have used a synthetic dataset inspired from a zenodo dataset.


## **Acknowledgements**

Data source : [https://zenodo.org/records/15401061](https://zenodo.org/records/15401061) : inspired

