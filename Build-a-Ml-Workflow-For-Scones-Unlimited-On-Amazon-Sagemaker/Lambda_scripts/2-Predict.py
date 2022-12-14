import json
import base64
import boto3
# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2022-09-26-06-59-53-344"
sagemaker = boto3.client('runtime.sagemaker')
def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.invoke_endpoint(EndpointName = ENDPOINT ,ContentType ="image/png" ,
    Body = image)
    # For this model the IdentitySerializer needs to be "image/png"
    
    # Make a prediction:
    inferences =predictor['Body'].read()
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
   
    return {
        'statusCode': 200,
        'body': json.dumps(event) 
    }