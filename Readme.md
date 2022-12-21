 <html>
    <body>
<h2>Weather prediction</h2>
<h3>domain background </h3>
<p>
in this project it is about computer vision to make image classification to predict type of weather from picture that taken from any place and get the sky in picture .
</p>
<img src="https://imerit.net/wp-content/uploads/2021/07/Weather-recognition-1024x567.jpg">
    </body>
</html>

<p>this it considered as a problem from this image we can predict the type of weather and can cancel any thing that will affect badly on this issua .</p>


```python

```


    
![png](output_2_0.png)
    


<p> as we can see threr are some of sample of dataset that used and here the site of this dataset </p>
<a href ="https://imerit.net/blog/top-13-machine-learning-image-classification-datasets-all-pbm/">site of data</a>

<p>there are a four cast <mark>sunrise</mark> , <mark>cloudy</mark>, <mark>rain</mark> and <mark>shine</mark> and our task is to predict one of them according to image </p>

<h3>problem statement</h3>
<p>in this problem we can solve it by using deep learning models wich pretrained before this technique is called <mark>Transfer Learning</mark> so i use <mark>Resnet</mark> which is trained before on imagenet and make freeze for layer and add output layer that can fit data</p>
<p>the goal of network is to extract features and then use it and pass it through the fully connected layers which has an output layer that can predict result </p>



```python

```

software that used is 
python
aws cloud 
S3 storage  , Lambda 
sagemaker 
numpy , pandas , matplotlib , pytorch , torchvision , smdebug
    
![png](output_6_0.png)
    


<p>and this is a count and frequency for every class that used it consider is low so we need to collect more </p>
<a href ="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4drtyfjtfy-1.zip">Link of Dataset</a>

<h3>solution statement</h3>
<p>my solution is use transfer learning it will the best idea i choose resnet50 and train it on 90% of data and take about 10% to make validation</p>
<p>and also use some of data augumentation techniques to increase data and also make model train on some data that may be see them in the future
    </p>
    <p>i make some of data processing such resize data and remove some of images not contains 3 channels so it considered as a not useful
    </p>

<h3>benchmark model</h3>
<p>there are more models that can handle this well but in our case i haven't much data so we can collect data to increase performane but performance of data on this size it consider a good </p>

<h3>Evaulation Metrics</h3>
<p>our problem is considered as a classification project so in this case i used a <mark>accuracy</mark> to see how many time that model can predict well among all dataset and also see loss function value our loss in this case is Categoricla Cross Entropy Loss it sufficient in multi class classification we can also use some metrics such precision or recall but our case i interset on all classes not speciified one so accuracy it useful</p>

<h3>Poject Design</h3>
<ol>
    <li>upload data from interenet (the link that i have mention)</li>
    <li>extract zip files data in local dir in sagemaker</li>
    <li>make data preprocessing and visulaization </li>
    <li>remove some files that have isuua and not releated to the majority of data </li>
    <li>assert that data is more prepred to get in modeling stage </li>
    <li>upload data from sagemaker to s3</li>
    <li>make test on data in sagemaker studio before make a training script </li>
    <li>write a script that will train network</li>
    <li>upload a pretrained model and freeze features extraction layer </li>
    <li>customize model on our problem and dataset </li>
    <li>build a transformers for trainining and validation data </li>
    <li>make monitor on data to see cpubootelenck and vansishing gradients , overfitting</li>
    <li>Estaplish a Scrpit mode and pass for it out script</li>
    <li>make debugger and Profiler to monior model during training</li>
    <li>train model for 50 epochs</li>
    <li>write a script for interface </li>
    <li>deploy model one for accept PIL image</li>
    <li>make endpoint that acceepts PIL image<li>
    <li>make anthor endpoint to accept numpy image<li>
    <li>create a lambda function for our endpoint<li>
    <li>confiure lambda to can handle and access sagemaker and endpoint </li>
 </ol>


```python

```
