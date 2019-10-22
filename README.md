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

Assume that there are K "load areas" in the grids. Let <img src="/tex/7b5c69854bf1ba5aecb122c8ddd74fe2.svg?invert_in_darkmode&sanitize=true" align=middle width=16.66101689999999pt height=27.91243950000002pt/> be the load value of area k (k = 1, K) at time-step t which t is considered as the current time-step. The problem is formulated as follows.

***Input*** <a name="input"></a>

<p align="center"><img src="/tex/0bb5c0168087d8c83100a2a70c0f998b.svg?invert_in_darkmode&sanitize=true" align=middle width=162.0530703pt height=59.178683850000006pt/></p>
<p align="center"><img src="/tex/1aca9d30370e93a90236a67f76837718.svg?invert_in_darkmode&sanitize=true" align=middle width=220.49139749999998pt height=14.611878599999999pt/></p>

Where:
- <img src="/tex/47145dd469cc1c3848c30ceccd72bf11.svg?invert_in_darkmode&sanitize=true" align=middle width=16.66101689999999pt height=27.91243950000002pt/>: the predicted value of area k at time-step i.
- <img src="/tex/86c8a2f3dcf85e1aa0acc7d42b3af7d7.svg?invert_in_darkmode&sanitize=true" align=middle width=15.23408039999999pt height=27.91243950000002pt/>: the ground-truth (i.e. verified) value of area k at time-step i.
- <img src="/tex/07cc3366c0ea9a5ed88b72396cedf0f6.svg?invert_in_darkmode&sanitize=true" align=middle width=21.69913019999999pt height=27.91243950000002pt/>: the binary variable indicates that the value will be updated/verified by users.
- l: the number of historical time-steps used for prediction

***Output*** <a name="output"></a>
<p align="center"><img src="/tex/26f40f24ffbc277a20f13c8564743a38.svg?invert_in_darkmode&sanitize=true" align=middle width=606.2777737499999pt height=32.5387656pt/></p>
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

&nbsp;&nbsp;&nbsp;&nbsp;*e_x(-1, l, 1), d_x(-1,h,1), d_y(-1,h,1)*

&emsp;Randomly create binary matrix <img src="/tex/bcd07b807305a9d37467c1be1af88cb4.svg?invert_in_darkmode&sanitize=true" align=middle width=44.068071299999986pt height=22.465723500000017pt/>

&emsp;*For k = 0 -> K:*

&emsp;&emsp;*For i = 0 -> T - l - h:*

1. Data need to be transformed as format: <p align="center"><img src="/tex/d781f6fbc62c738a9382c6f550245dc0.svg?invert_in_darkmode&sanitize=true" align=middle width=76.6173012pt height=17.031940199999998pt/></p> where <img src="/tex/1317edcef4119650134675a9733a3d3d.svg?invert_in_darkmode&sanitize=true" align=middle width=15.10851044999999pt height=14.15524440000002pt/> is the input with shape *(l, 1)*, <img src="/tex/fddb50275f702b5af702ac3f3bc5ed81.svg?invert_in_darkmode&sanitize=true" align=middle width=14.733744299999989pt height=14.15524440000002pt/> is the target with shape *(h,1)*.

	<p align="center"><img src="/tex/63071466b63dd7d45f392bcc9eeafd3e.svg?invert_in_darkmode&sanitize=true" align=middle width=519.1508487pt height=20.2118565pt/></p>

2. If <img src="/tex/96caff1dc9392f9777c86aa50a855b4d.svg?invert_in_darkmode&sanitize=true" align=middle width=290.04922815000003pt height=27.91243950000002pt/>
3. e_x.append(<img src="/tex/1317edcef4119650134675a9733a3d3d.svg?invert_in_darkmode&sanitize=true" align=middle width=15.10851044999999pt height=14.15524440000002pt/>); d_x.append(<img src="/tex/64ddb675e322bf1281650681445f5914.svg?invert_in_darkmode&sanitize=true" align=middle width=16.01033609999999pt height=22.831056599999986pt/>); d_y.append(<img src="/tex/885839bf3fa334609583ddbd937afdfb.svg?invert_in_darkmode&sanitize=true" align=middle width=15.63556994999999pt height=22.831056599999986pt/>)

- Prepare training data for DCRNN

&emsp;In the training phase, training data needs to be prepared as follows

&emsp;*dataX(-1, l, K, 1), dataY(-1, h, K, 1)*

&emsp;Randomly create binary matrix <img src="/tex/bcd07b807305a9d37467c1be1af88cb4.svg?invert_in_darkmode&sanitize=true" align=middle width=44.068071299999986pt height=22.465723500000017pt/>

&emsp;for i = 0 -> T - l - h: #T is the number of time-steps in the training set

1. Data need to be transformed as format: (x, y) where x is the input with shape *(l, K, 1)*, y is the target with shape *(h, K, 1)*

2. Change the value <img src="/tex/47145dd469cc1c3848c30ceccd72bf11.svg?invert_in_darkmode&sanitize=true" align=middle width=16.66101689999999pt height=27.91243950000002pt/> whose <img src="/tex/429ceb369d5d0b5f585aade0bbbbab3c.svg?invert_in_darkmode&sanitize=true" align=middle width=52.65788384999998pt height=27.91243950000002pt/> as <img src="/tex/e49c6dac8af82421dba6bed976a80bd9.svg?invert_in_darkmode&sanitize=true" align=middle width=16.43840384999999pt height=14.15524440000002pt/> random<img src="/tex/2c5576bb382cbfcbf6a548af75b5be33.svg?invert_in_darkmode&sanitize=true" align=middle width=123.25328894999998pt height=27.91243950000002pt/>, where <img src="/tex/aca94dc4280088e4b15ee4be41751fd0.svg?invert_in_darkmode&sanitize=true" align=middle width=13.18495034999999pt height=24.7161288pt/> is the stdev of the training set.

3. dataX.append(x); dataY.append(y)

### Evaluation's Metrices <a name="evaluation_metric"></a>
1. MAE
2. RMSE
3. MAPE
### Compared Approaches <a name="compared_approach"></a>
1. ARIMA
2. LSTM encoder-decoder
