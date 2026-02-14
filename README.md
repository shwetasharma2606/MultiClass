a) Problem statement :  Problem aims to categorize mobile phones into distinct price ranges based on their technical specifications (features) rather than predicting an exact, continuous price. Classify a mobile phone into one of four distinct price categories: 
0: Low Cost
1: Medium Cost
2: High Cost
3: Very High Cost 

b) Dataset description  : The data set consiss of attributs : battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g,touch_screen,wifi which helps us to classify mobile in price_range

c) Models used and comparision :  
Model	Accuracy	Precision	Recall	F1 Score	AUC	MCC
Logistic Regression	0.9616666666666667	0.9617530196376981	0.9616666666666667	0.9616296446703861	0.9977481481481482	0.9489416093316826
Decision Tree	0.8283333333333334	0.8268471432572724	0.8283333333333334	0.8266221096242187	0.8855555555555557	0.7716972519960191
K-Nearest Neighbor Classifier	0.4766666666666667	0.5008053672886945	0.4766666666666667	0.48185951175134695	0.7322777777777778	0.30451013195356696
Naive Bayes Classifier - Gaussian	0.7983333333333333	0.7984	0.7983333333333333	0.7983328908323815	0.9493925925925926	0.7311463152997468
Ensemble Model - Random Forest	0.8900	0.8891835840365252	0.8900	0.8893533923901263	0.9812203703703705	0.8534882397010957
Ensemble Model - XGBoost	0.9166666666666666	0.9167114192114192	0.9166666666666666	0.9166282526786763	0.9896185185185186	0.8889283976956683

d) Observations on the performance of each model on the chosen
dataset
ML Model Name 	Observation about model performance
Logistic Regression	: Top Performer. It achieved the highest scores across all metrics (Accuracy: ~96.17%, AUC: ~0.997). This indicates an excellent fit and a very low rate of both false positives and false negatives.
Decision Tree	: Moderate Performance. With an accuracy of ~82.8%, it is a decent baseline but significantly lags behind the top models. The gap between its AUC (0.88) and Logistic Regression suggests it may be struggling with some complexity or noise in the data.
kNN	: Weakest Performer. It performed poorly with an accuracy of only ~47.6% and a very low MCC (0.30). This suggests the "neighborhood" approach isn't working well, possibly due to high dimensionality (the "curse of dimensionality") or unscaled features.
Naive Bayes	: Good/Stable. It shows solid performance (Accuracy: ~79.8%, AUC: ~0.949). While not the best, the high AUC indicates it is very good at distinguishing between classes even if its absolute classification threshold is slightly off.
Random Forest (Ensemble)	: Strong Performance. As an ensemble of trees, it significantly improved upon the single Decision Tree, reaching ~89% accuracy. It is a robust model with a high AUC (0.98), showing it handles the data variance well.
XGBoost (Ensemble)	: Runner-Up. This model performed excellently with ~91.6% accuracy and high precision/recall. It is the second-best model here, proving that gradient boosting is highly effective for this specific dataset.
