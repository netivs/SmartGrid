







# Deep Learning-based Electric Load Prediction with Partial Information for Smart Grids

Author

## Table of contents
1. [Introduction](#introduction)
2. [Problem Description](#problem_description)
    1. [Dataset](#dataset_des)
		- [Hourly Load Estimate dataset](#hrl_load_estimate_dataset)
		- [Data Fields](#data_fields)
    2. [Problem Formulation](#problem_formulation)
	    - [Input](#input)
	    - [Output](#output)
3. [Proposed Approach](#proposed_approach)
4. [Evaluation](#evaluation)
		- [Dataset](#dataset_evaluation)
		- [Experiments Setting](#experiment_setting)
		- [Model Training](#model_training)
		- [Evaluation's Metrices](#evaluation_metrices)
		- [Compared Approaches](#compared_approach)

## Introduction <a name="introduction"></a>
Estimating the future electric load of server company in US by using historical data. The predicted values are used for strategies decisions making by smart grids operator in order to maximize the possible return. Then, the actual values will be updated/verified by the companies/users. The _load estimator_ uses the past loads for predicting the future electric demand.
However, only a portion of the data is verified by users which means the historical data used by the _load estimator_ containing two types of data: _verified_ data and _predicted_ data. This situation leads to the decrease in the prediction accuracy. Therefore, in this research, we aim to solve this problem by leveraging the advances of deep learning techniques.

## Problem Description <a name="problem_description"></a>

### Dataset <a name="dataset_des"></a>

***Hourly Load Estimated dataset*** <a name="hrl_load_estimate_dataset"></a>

This data contains estimated integrated hourly loads that are calculated from meter data. Estimated loads reflect revenue-quality meter information but have not yet been verified by the electric distribution companies and are subject to later adjustment. This information is provided for informational purposes only and should not be relied upon by any party for the actual billing values.

***Data Fields*** <a name="data_fields"></a>

| Field name             	| Data Type 	| Description                                                                                	|
|------------------------	|:---------:	|--------------------------------------------------------------------------------------------	|
| Datetime Beginning UTC 	|  Datetime 	| Datetime Beggining according to Coordinated Universal Time                                 	|
| Datetime Beginning EPT 	|  Datetime 	| Datetime Beginning according to Eastern Prevailing Time                                    	|
| NERC Region            	|   String  	| The regions of North American Electric Reliability Corporation. Research on RFC, SERC, RTO 	|
| Market Region          	|   String  	| The market region of the NERC region focus on                                              	|
| Transmission Zone      	|   String  	| Transmission Zone Location !!!!!                                                           	|
| Load Area              	|   String  	| Fully Metered Electric Distribution Company !!!!!                                          	|
| MW                     	|   Number  	| Load in MW                                                                                 	|
| Company Verified       	|  Boolean  	| Indicates whether the metered load has been verified by the Electric Distribution          	|

In this study, we consider to estimate the load for each “load area” hourly using the historical data from 2017 to 2019. Since there are some unverified values in the dataset, for evaluating the performance of the prediction model, we consider all the values are verified.

### Problem Formulation <a name="problem_formulation"></a>

Assume that there are K "load areas" in the grids. Let $x_t^k$ be the load value of area k (k = 1, K) at time-step t which t is considered as the current time-step. The problem is formulated as follows. 

***Input*** <a name="input"></a>

$$x_i^k = 
\begin{cases}
o_i^k,  & \text{if $m_i^k$ = 1} \\[2ex]
x_i^k, & \text{if $m_i^k$ = 0}
\end{cases}
$$
$$\forall k = 1, ..., K; i  = t - l +1, ..., t$$

Where:
- $x_i^k$: the predicted value of area k at time-step i.
- $o_i^k$: the ground-truth (i.e. verified) value of area k at time-step i.
- $m_i^k$: the binary variable indicates that the value will be updated/verified by users.
- l: the number of historical time-steps used for prediction

***Output*** <a name="output"></a>
$$\hat{x}_{t+1}^k, \hat{x}_{t+1}^k,..., \hat{x}_{t+1}^k = \argmax_{{x}_{t+1}^k, {x}_{t+2}^k,..., {x}_{t+h}^k} p({x}_{t+1}^k, {x}_{t+2}^k,..., {x}_{t+h}^k \vee {x}_{t-l+1}^k, {x}_{t-l+2}^k,..., {x}_{t}^k)$$
Where:
- h: the number of time-steps need to be predicted

## Proposed Approach <a name="proposed_approach"></a>
*Diffusion Convolutional Recurrent Neural Network with static graph*

## Evaluation <a name="evaluation"></a>
### Dataset <a name="dataset_evaluation"></a>
The hourly load estimated dataset is divided into three subsets for training, validating and testing. The divided ratio as following: 60% for training set, 20% for validating set and 20% for testing set.
### Experiments Setting <a name="experiment_setting"></a>
| 	Parameter 	|  Type  	 | Description                                                   									|
|	-----------	|	:------:	 |---------------------------------------------------------------				|
|     		l     		| Number | The number of time-steps used for each prediction             		|
|     		h     		| Number | The number of time-steps need to be predicted                 		|
|    		 r     		| String 	 | The percentage of verified data in the period of l time-steps 	|

### Model Training <a name="model_training"></a>
- Prepare training data for LSTM encoder-decoder
dataX(-1, l, 1), dataY(-1,h,1)

Randomly create binary matrix $M_{T*K}$

*For k = 0 -> K:*

&nbsp;&nbsp;&nbsp;&nbsp;;*For i = 0 -> T - l - h:*

1. Data need to be transformed as format: (x, y) where x is the input with shape *(l, 1)*, y is the target with shape *(h,1)*. *x = ($x_i^k,...,x_{i+l}^k$), y = ($x_{i+l+1}^k,...,x_{i+l+h}^k$)*
2. If $m_i^k = 0: x_i^k \to random(x_i^k - x'; x_i^k + x')$ 
3. dataX.append(x); dataY.append(y) 

- Prepare training data for DCRNN
In the training phase, training data needs to be prepared as follows

*dataX(-1, l, K, 1), dataY(-1, h, K, 1)*

Randomly create binary matrix $M_{T*K}$

for i = 0 -> T - l - h: #T is the number of time-steps in the training set

4. Data need to be transformed as format: (x, y) where x is the input with shape *(l, K, 1)*, y is the target with shape *(h, K, 1)*

5. Change the value $x_i^k$ whose $m_i^k = 0 as \to random(x_i^k - x'; x_i^k + x')$, where $x'$ is the stdev of the training set.

6. dataX.append(x); dataY.append(y)
 
### Evaluation's Metrices <a name="evaluation_metric"></a>
1. MAE
2. RMSE
3. MAPE
### Compared Approaches <a name="compared_approach"></a>
1. ARIMA
2. LSTM encoder-decoder
