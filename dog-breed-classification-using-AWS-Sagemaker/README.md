# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
i choose resnet18 because it pretrained model and also it trained on the data have a dog image 
and also number of data is fewer so we need to use transfer learning 
paramter i choose is learning rate , batch size range of learning rate are between 0.001, 0.1 , 
and batch size in [16, 32, 64]
and it is the best hyperparamters 
 'batch-size': '"16"',
 'lr': '0.00899693582951801',

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
after make a script that will put it on an estimator then i define inside the estimator what rules that i want to monitor such 
vansishing gradient problem it make model not cabaple of make update weights 
overfitting and it means that model perform on train well and poor on test 
overtraing also
poor weight intialization it make model can't learn well
and alos CPUBottleneck to monitor cpu and see if there are a problems 

### Results
 What are the results/insights did you get by profiling/debugging your model?
 the average step duration on node algo-1-27 was 24.71s. The rule detected 0 outliers, where step duration was larger than 3 times the standard deviation of 2.31s
 according to cpu utilization it work weel no bottleneck occurs 
 there are an overfitting on the data 
 no vanishing gradients occurs 
 number of data points in Data loader is 11 
 and most of result such load blancing and more on the html files 


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
after model has trained then :
i deploy model and identify end point name and also intial instance count , 
instance type and then identify serelizer in case of input it take a numpy array 
and in case of output it return a numpy array
and also i read data and make some of data preprocessing such resize and scale data
