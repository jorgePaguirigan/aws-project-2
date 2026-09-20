import boto3

client = boto3.client("bedrock-agentcore", region_name="us-east-1")

try:
    resp = client.start_browser_session(
        browserIdentifier="aws.browser.v1",
        name="debug-test",
        sessionTimeoutSeconds=60,
    )
    print("SUCCESS:", resp)
except Exception as e:
    print("TYPE:", type(e).__name__)
    print("ERROR:", e)