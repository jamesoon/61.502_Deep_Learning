"""
MedMCQA AWS CDK Stack
=====================
Resources:
  - DynamoDB: questions, submissions, user_sessions
  - Cognito User Pool (Admins + Students groups)
  - Lambda: questions CRUD, submit/poll, grade (Step Functions task), session trigger
  - API Gateway HTTP API with Cognito JWT authorizer
  - Step Functions: async explanation pipeline (Map → grade → aggregate)
  - S3 + CloudFront: React SPA hosting
  - Route 53 + ACM: dl.mdaie-sutd.fit with HTTPS
"""

import json
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_dynamodb as dynamodb,
    aws_cognito as cognito,
    aws_lambda as lambda_,
    aws_apigatewayv2 as apigwv2,
    aws_apigatewayv2_authorizers as apigwv2_auth,
    aws_apigatewayv2_integrations as apigwv2_int,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_deployment as s3deploy,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_route53 as route53,
    aws_route53_targets as r53_targets,
    aws_certificatemanager as acm,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_ssm as ssm,
    aws_logs as logs,
)
from constructs import Construct


DOMAIN     = "mdaie-sutd.fit"
SUBDOMAIN  = "dl.mdaie-sutd.fit"
REGION     = "ap-southeast-1"

# HuggingFace model IDs — override via SSM after deploy if needed
HF_EXPLAIN_MODEL = "jamezoon/gemma-3-4b-it-medmcqa-lora"
# Cross-encoder published after training completes:
HF_GRADE_MODEL   = "jamezoon/deberta-v3-large-medmcqa"


class MedMCQAStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ── 1. DynamoDB Tables ─────────────────────────────────────────────────
        questions_table = dynamodb.Table(
            self, "QuestionsTable",
            table_name="medmcqa_questions",
            partition_key=dynamodb.Attribute(name="questionId", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )

        submissions_table = dynamodb.Table(
            self, "SubmissionsTable",
            table_name="medmcqa_submissions",
            partition_key=dynamodb.Attribute(name="submissionId", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )
        # GSI: look up submissions by user
        submissions_table.add_global_secondary_index(
            index_name="userId-index",
            partition_key=dynamodb.Attribute(name="userId", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="submittedAt", type=dynamodb.AttributeType.STRING),
        )

        sessions_table = dynamodb.Table(
            self, "SessionsTable",
            table_name="medmcqa_user_sessions",
            partition_key=dynamodb.Attribute(name="userId", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            time_to_live_attribute="expiresAt",
        )

        # ── 2. Cognito User Pool ───────────────────────────────────────────────
        user_pool = cognito.UserPool(
            self, "UserPool",
            user_pool_name="medmcqa-users",
            self_sign_up_enabled=True,
            sign_in_aliases=cognito.SignInAliases(email=True),
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True),
                fullname=cognito.StandardAttribute(required=False, mutable=True),
            ),
            password_policy=cognito.PasswordPolicy(
                min_length=8,
                require_uppercase=True,
                require_digits=True,
                require_symbols=False,
            ),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Groups
        cognito.CfnUserPoolGroup(self, "AdminsGroup",
            user_pool_id=user_pool.user_pool_id,
            group_name="Admins",
            description="Platform administrators",
            precedence=1,
        )
        cognito.CfnUserPoolGroup(self, "StudentsGroup",
            user_pool_id=user_pool.user_pool_id,
            group_name="Students",
            description="Exam students",
            precedence=10,
        )

        # App Client (SPA — no secret)
        user_pool_client = user_pool.add_client(
            "WebClient",
            user_pool_client_name="medmcqa-web",
            generate_secret=False,
            auth_flows=cognito.AuthFlow(
                user_srp=True,
                user_password=True,
            ),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(implicit_code_grant=True),
                scopes=[cognito.OAuthScope.EMAIL, cognito.OAuthScope.OPENID,
                        cognito.OAuthScope.PROFILE],
            ),
            prevent_user_existence_errors=True,
        )

        # ── 3. Shared Lambda environment ───────────────────────────────────────
        # NOTE: USER_POOL_ID is intentionally omitted here to avoid a circular
        # dependency (UserPool → SessionTriggerFn via trigger config, SessionTriggerFn
        # env → UserPool via ID). It's added per-function below ONLY to Lambdas
        # that are NOT attached as Cognito triggers.
        common_env = {
            "QUESTIONS_TABLE":   questions_table.table_name,
            "SUBMISSIONS_TABLE": submissions_table.table_name,
            "SESSIONS_TABLE":    sessions_table.table_name,
            "HF_EXPLAIN_MODEL":  HF_EXPLAIN_MODEL,
            "HF_GRADE_MODEL":    HF_GRADE_MODEL,
            "REGION":            REGION,
        }

        # Shared role for QuestionsFn, GradeFn, AggregateFn, SessionTriggerFn.
        # SubmitFn gets its own role (submit_role, below) so that
        # state_machine.grant_start_execution doesn't taint the shared role —
        # otherwise LambdaRoleDefaultPolicy would reference StateMachine, which
        # references GradeFn/AggregateFn (via LambdaInvoke tasks), whose auto-
        # dependency on LambdaRoleDefaultPolicy closes a CFN circular reference.
        lambda_role = iam.Role(
            self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )
        questions_table.grant_read_write_data(lambda_role)
        submissions_table.grant_read_write_data(lambda_role)
        sessions_table.grant_read_write_data(lambda_role)
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "cognito-idp:AdminAddUserToGroup",
                "cognito-idp:AdminGetUser",
                "cognito-idp:AdminUserGlobalSignOut",
                "cognito-idp:AdminListGroupsForUser",  # used by session_trigger pre-token-gen
                "ssm:GetParameter",
            ],
            resources=["*"],
        ))

        # Dedicated role for SubmitFn — StartExecution grant lives here only.
        submit_role = iam.Role(
            self, "SubmitRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )
        questions_table.grant_read_data(submit_role)
        submissions_table.grant_read_write_data(submit_role)
        sessions_table.grant_read_data(submit_role)
        submit_role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{REGION}:{self.account}:parameter/medmcqa/*"],
        ))

        py_runtime = lambda_.Runtime.PYTHON_3_12

        # ── 4. Session / Cognito Trigger Lambda ────────────────────────────────
        session_fn = lambda_.Function(
            self, "SessionTriggerFn",
            runtime=py_runtime,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/session_trigger"),
            role=lambda_role,
            environment=common_env,
            timeout=Duration.seconds(10),
        )

        # Attach Cognito triggers
        user_pool.add_trigger(
            cognito.UserPoolOperation.POST_CONFIRMATION,
            session_fn,
        )
        user_pool.add_trigger(
            cognito.UserPoolOperation.PRE_TOKEN_GENERATION,
            session_fn,
        )

        # ── 5. Questions Lambda ────────────────────────────────────────────────
        questions_fn = lambda_.Function(
            self, "QuestionsFn",
            runtime=py_runtime,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/questions"),
            role=lambda_role,
            environment=common_env,
            timeout=Duration.seconds(30),
        )

        # ── 6. Grade Lambda (Step Functions task) ─────────────────────────────
        # Demo mode toggle (SSM-backed, flippable without redeploy).
        # Operator flips this with:
        #   aws ssm put-parameter --name /medmcqa/demo_mode --type String \
        #       --value 1 --overwrite --region ap-southeast-1
        # When "1", the grade Lambda returns canned explanations from
        # demo_explanations.json instead of calling HF (zero cost, sub-second).
        ssm.StringParameter(
            self, "DemoModeParam",
            parameter_name="/medmcqa/demo_mode",
            string_value="0",   # production default — set to "1" before a demo
            description="Grade Lambda demo toggle (1=canned explanations, 0=live HF)",
        )

        # Models evaluated (as test-takers) on each submission. CSV of HF model IDs.
        # Flippable without redeploy:
        #   aws ssm put-parameter --name /medmcqa/eval_models --type String \
        #       --value 'model/a,model/b' --overwrite --region ap-southeast-1
        ssm.StringParameter(
            self, "EvalModelsParam",
            parameter_name="/medmcqa/eval_models",
            string_value="jamezoon/gemma-3-4b-it-medmcqa-lora,jamezoon/deberta-v3-large-medmcqa",
            description="CSV of HuggingFace model IDs to evaluate per submission",
        )

        grade_env = {**common_env, "DEMO_MODE": "0"}  # env-var fallback if SSM lookup fails

        grade_fn = lambda_.Function(
            self, "GradeFn",
            runtime=py_runtime,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/grade"),
            role=lambda_role,
            environment=grade_env,
            timeout=Duration.seconds(120),  # 1 explain + N eval-model HF calls per question
            memory_size=512,
        )
        # Allow grade Lambda to read HF API key + demo flag from SSM
        grade_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{REGION}:{self.account}:parameter/medmcqa/*"],
        ))

        # ── 7. Aggregate Lambda (Step Functions final state) ───────────────────
        aggregate_fn = lambda_.Function(
            self, "AggregateFn",
            runtime=py_runtime,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/aggregate"),
            role=lambda_role,
            environment=common_env,
            timeout=Duration.seconds(30),
        )

        # ── 8. Submit Lambda ──────────────────────────────────────────────────
        # Step Functions ARN injected after SF is created (see below).
        # Uses dedicated submit_role to avoid circular dependency with StateMachine.
        submit_fn = lambda_.Function(
            self, "SubmitFn",
            runtime=py_runtime,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/submit"),
            role=submit_role,
            environment=common_env,
            timeout=Duration.seconds(30),
        )

        # ── 9. Step Functions: Explanation Pipeline ────────────────────────────
        grade_task = tasks.LambdaInvoke(
            self, "GradeQuestionTask",
            lambda_function=grade_fn,
            output_path="$.Payload",
            retry_on_service_exceptions=True,
        )
        grade_task.add_retry(
            max_attempts=3,
            interval=Duration.seconds(5),
            backoff_rate=2,
        )

        aggregate_task = tasks.LambdaInvoke(
            self, "AggregateResultsTask",
            lambda_function=aggregate_fn,
            output_path="$.Payload",
        )

        # Map state: grade all questions in parallel
        grade_map = sfn.Map(
            self, "GradeAllQuestions",
            items_path="$.questions",
            result_path="$.gradedQuestions",
            max_concurrency=10,
        )
        grade_map.item_processor(grade_task)

        pipeline = grade_map.next(aggregate_task)

        sf_log_group = logs.LogGroup(
            self, "SFLogGroup",
            log_group_name="/aws/states/medmcqa-grading",
            removal_policy=RemovalPolicy.DESTROY,
            retention=logs.RetentionDays.ONE_WEEK,
        )

        state_machine = sfn.StateMachine(
            self, "GradingStateMachine",
            state_machine_name="medmcqa-grading",
            definition_body=sfn.DefinitionBody.from_chainable(pipeline),
            state_machine_type=sfn.StateMachineType.STANDARD,
            logs=sfn.LogOptions(
                destination=sf_log_group,
                level=sfn.LogLevel.ERROR,
            ),
            timeout=Duration.minutes(30),
        )

        # Grant submit Lambda permission to start executions
        state_machine.grant_start_execution(submit_fn)
        submit_fn.add_environment("STATE_MACHINE_ARN", state_machine.state_machine_arn)

        # Grant aggregate Lambda permission to update submissions
        submissions_table.grant_read_write_data(aggregate_fn)

        # ── 10. API Gateway HTTP API ────────────────────────────────────────────
        http_api = apigwv2.HttpApi(
            self, "HttpApi",
            api_name="medmcqa-api",
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=[f"https://{SUBDOMAIN}", "http://localhost:5173"],
                allow_methods=[apigwv2.CorsHttpMethod.ANY],
                allow_headers=["Content-Type", "Authorization"],
                max_age=Duration.hours(1),
            ),
        )

        cognito_authorizer = apigwv2_auth.HttpJwtAuthorizer(
            "CognitoAuth",
            jwt_issuer=f"https://cognito-idp.{REGION}.amazonaws.com/{user_pool.user_pool_id}",
            jwt_audience=[user_pool_client.user_pool_client_id],
        )

        questions_integration = apigwv2_int.HttpLambdaIntegration(
            "QuestionsIntegration", questions_fn
        )
        submit_integration = apigwv2_int.HttpLambdaIntegration(
            "SubmitIntegration", submit_fn
        )

        # Questions routes (auth required)
        for method in [apigwv2.HttpMethod.GET, apigwv2.HttpMethod.POST]:
            http_api.add_routes(
                path="/questions",
                methods=[method],
                integration=questions_integration,
                authorizer=cognito_authorizer,
            )
        for method in [apigwv2.HttpMethod.PUT, apigwv2.HttpMethod.DELETE]:
            http_api.add_routes(
                path="/questions/{questionId}",
                methods=[method],
                integration=questions_integration,
                authorizer=cognito_authorizer,
            )

        # Submit + poll routes
        http_api.add_routes(
            path="/submit",
            methods=[apigwv2.HttpMethod.POST],
            integration=submit_integration,
            authorizer=cognito_authorizer,
        )
        http_api.add_routes(
            path="/submissions/{submissionId}",
            methods=[apigwv2.HttpMethod.GET],
            integration=submit_integration,
            authorizer=cognito_authorizer,
        )

        # Student dashboard — synthetic default exam + past submissions.
        # Served by the submit Lambda (shares DDB clients) until the Exam Groups
        # admin UI is fully wired.
        for p in ("/my-exam", "/my-submissions"):
            http_api.add_routes(
                path=p,
                methods=[apigwv2.HttpMethod.GET],
                integration=submit_integration,
                authorizer=cognito_authorizer,
            )

        # ── 11. S3 + CloudFront + Route 53 + ACM ──────────────────────────────
        frontend_bucket = s3.Bucket(
            self, "FrontendBucket",
            bucket_name=f"medmcqa-frontend-{self.account}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Upload Image Lambda — admin-only, puts question figures under uploads/.
        # Served back to viewers via the same CloudFront distribution (OAC).
        upload_image_fn = lambda_.Function(
            self, "UploadImageFn",
            runtime=py_runtime,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/upload_image"),
            environment={
                "FRONTEND_BUCKET": frontend_bucket.bucket_name,
                "PUBLIC_BASE":     f"https://{SUBDOMAIN}",
            },
            timeout=Duration.seconds(30),
            memory_size=512,
        )
        # Scope writes to uploads/* only — never let this Lambda touch the SPA assets.
        upload_image_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:PutObject"],
            resources=[f"{frontend_bucket.bucket_arn}/uploads/*"],
        ))
        upload_image_integration = apigwv2_int.HttpLambdaIntegration(
            "UploadImageIntegration", upload_image_fn
        )
        http_api.add_routes(
            path="/upload-image",
            methods=[apigwv2.HttpMethod.POST],
            integration=upload_image_integration,
            authorizer=cognito_authorizer,
        )

        # Reference existing hosted zone by ID (two zones exist for mdaie-sutd.fit; use primary)
        hosted_zone = route53.HostedZone.from_hosted_zone_attributes(
            self, "HostedZone",
            hosted_zone_id="Z089465439W0RXJJ0LOEW",
            zone_name=DOMAIN,
        )

        # ACM certificate — MUST live in us-east-1 because CloudFront only
        # accepts certs from us-east-1 regardless of where the distribution is.
        # The stack itself is in ap-southeast-1, so we use DnsValidatedCertificate
        # which spawns a custom resource Lambda that creates the cert cross-region.
        # (DnsValidatedCertificate is deprecated in CDK 2.69+ but remains the
        # least-invasive cross-region pattern for a single-stack deploy.)
        certificate = acm.DnsValidatedCertificate(
            self, "Certificate",
            domain_name=SUBDOMAIN,
            hosted_zone=hosted_zone,
            region="us-east-1",
        )

        distribution = cloudfront.Distribution(
            self, "Distribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(frontend_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD,
            ),
            additional_behaviors={
                # API calls proxied through CloudFront → API Gateway
                "/api/*": cloudfront.BehaviorOptions(
                    origin=origins.HttpOrigin(
                        f"{http_api.api_id}.execute-api.{REGION}.amazonaws.com",
                        origin_path="/",
                    ),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    origin_request_policy=cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER,
                ),
            },
            domain_names=[SUBDOMAIN],
            certificate=certificate,
            default_root_object="index.html",
            error_responses=[
                # SPA deep-link routing: S3+OAC returns 403 for missing keys
                # (not 404), so we must rewrite both to index.html.
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
            ],
        )

        # Route 53: dl.mdaie-sutd.fit → CloudFront
        route53.ARecord(
            self, "AliasRecord",
            zone=hosted_zone,
            record_name="dl",
            target=route53.RecordTarget.from_alias(
                r53_targets.CloudFrontTarget(distribution)
            ),
        )

        # ── 12. Outputs ────────────────────────────────────────────────────────
        CfnOutput(self, "UserPoolId",       value=user_pool.user_pool_id)
        CfnOutput(self, "UserPoolClientId", value=user_pool_client.user_pool_client_id)
        CfnOutput(self, "ApiEndpoint",      value=http_api.api_endpoint)
        CfnOutput(self, "CloudFrontUrl",    value=f"https://{SUBDOMAIN}")
        CfnOutput(self, "FrontendBucketName", value=frontend_bucket.bucket_name)
        CfnOutput(self, "StateMachineArn",  value=state_machine.state_machine_arn)
