#!/usr/bin/env python3
import os
import aws_cdk as cdk
from stacks.medmcqa_stack import MedMCQAStack

app = cdk.App()

MedMCQAStack(
    app,
    "MedMCQAStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT") or os.environ.get("AWS_ACCOUNT_ID"),
        region=os.environ.get("CDK_DEFAULT_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-1",
    ),
    description="MedMCQA exam platform — DL@SUTD MSTR-DAIE",
)

app.synth()
