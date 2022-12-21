import json
import boto3
def lambda_handler(event, context):
    # TODO implement
    final_resrult = None
    content = event['image']
    type_ = event['Type']
    endpoint_name = "Weather-prediction-v5"
    runtime=boto3.Session().client('sagemaker-runtime')
    if type_ == "image":
        response=runtime.invoke_endpoint(EndpointName=endpoint_name,
                                        ContentType="image/jpeg",
                                    #Accept='application/json',
                                    #Body=bytearray(x)
                                    Body=content)
        res =  response['Body'].read().decode("utf-8")
        res  = json.loads(res)
        out = json.loads(res)
        out =out["type"]
        final_resrult = out
        
                
    return {
        'statusCode': 200,
        'body': json.dumps({"prediction":final_resrult})
    }
