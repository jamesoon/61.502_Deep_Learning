# MedMCQA Exam Platform — Demo Runbook

**Audience**: SUTD Deep Learning panel / classmates
**Time budget**: 8–10 minutes
**Strategy**: 3 segments (Architecture → Live Hybrid → Tie-back)
**Fallback**: pre-recorded screencast loaded as final slide

---

## 0. Pre-flight (T-30 min)

Run these in order from a laptop with the project repo checked out and AWS creds for `ap-southeast-1` loaded.

### 0.1 Confirm stack is deployed

```bash
cd aws/cdk
cdk list                    # should print: MedMCQAStack
aws cloudformation describe-stacks \
  --stack-name MedMCQAStack \
  --region ap-southeast-1 \
  --query 'Stacks[0].StackStatus'
# → "CREATE_COMPLETE" or "UPDATE_COMPLETE"
```

If the stack isn't deployed, run `cdk deploy` and budget +20 min for the first deploy (CloudFront + ACM cert validation).

### 0.2 Flip demo mode ON

The grade Lambda reads `/medmcqa/demo_mode` from SSM at cold start. Set to `1` so it returns canned explanations from `aws/lambdas/grade/demo_explanations.json` instead of calling Hugging Face:

```bash
aws ssm put-parameter \
  --name /medmcqa/demo_mode \
  --type String \
  --value 1 \
  --overwrite \
  --region ap-southeast-1
```

Force a fresh container so the new value is picked up immediately:

```bash
# Touch the env var to trigger a redeploy of just the function config
aws lambda update-function-configuration \
  --function-name $(aws lambda list-functions --region ap-southeast-1 \
    --query "Functions[?starts_with(FunctionName, 'MedMCQAStack-GradeFn')].FunctionName | [0]" \
    --output text) \
  --environment "Variables={QUESTIONS_TABLE=medmcqa_questions,SUBMISSIONS_TABLE=medmcqa_submissions,SESSIONS_TABLE=medmcqa_user_sessions,USER_POOL_ID=<paste>,HF_EXPLAIN_MODEL=jamezoon/gemma-3-4b-it-medmcqa-lora,HF_GRADE_MODEL=jamezoon/deberta-v3-large-medmcqa,REGION=ap-southeast-1,DEMO_MODE=1}" \
  --region ap-southeast-1
```

(Easier: just hit `/submit` once before the demo so the warm container picks up SSM.)

### 0.3 Seed demo questions

The `demo_explanations.json` bundle keys explanations by `questionId`. Seed five matching questions into DynamoDB:

```bash
python aws/scripts/seed_demo_questions.py     # see below
```

(If `seed_demo_questions.py` doesn't exist yet, the live admin flow in segment 2 covers this — the templated fallback in `_demo_explanation()` handles ad-hoc questions gracefully.)

### 0.4 Pre-create Cognito users

Avoid the email-verification loop on stage. Pre-create two users in advance:

```bash
USER_POOL=$(aws cloudformation describe-stacks --stack-name MedMCQAStack \
  --region ap-southeast-1 \
  --query 'Stacks[0].Outputs[?OutputKey==`UserPoolId`].OutputValue' --output text)

# Admin user
aws cognito-idp admin-create-user \
  --user-pool-id "$USER_POOL" \
  --username demo-admin@mdaie-sutd.fit \
  --user-attributes Name=email,Value=demo-admin@mdaie-sutd.fit Name=email_verified,Value=true \
  --temporary-password 'Demo!Admin2026' \
  --message-action SUPPRESS \
  --region ap-southeast-1

aws cognito-idp admin-add-user-to-group \
  --user-pool-id "$USER_POOL" \
  --username demo-admin@mdaie-sutd.fit \
  --group-name Admins \
  --region ap-southeast-1

# Student user (repeat with demo-student@..., add to Students group)
```

Set the permanent password once via the SPA login → "force change password" flow before demo day.

### 0.5 Warm the Lambdas

Even in demo mode, cold starts add 1–2 s. Warm them:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://dl.mdaie-sutd.fit/api/questions
# Expect: 401 (auth required) — that's a successful warm hit
```

### 0.6 Open these tabs in advance

1. `https://dl.mdaie-sutd.fit` — the SPA (logged out)
2. AWS Console → Step Functions → `medmcqa-grading` → Executions tab
3. AWS Console → CloudFormation → `MedMCQAStack` → Resources tab (for reference during walkthrough)
4. Local editor with `aws/cdk/stacks/medmcqa_stack.py` open
5. Final slide of `Deep_learning_presentation_v4.pptx` (the new System Architecture slide)

---

## 1. Segment A — Architecture walkthrough (2 min)

### Talking track

Open Slide 15 (System Architecture). Walk left-to-right across the data path:

> "The same LoRA adapter our report evaluates — `jamezoon/gemma-3-4b-it-medmcqa-lora` — sits behind the Hugging Face inference endpoint at the right end of this pipeline. Everything between the browser and that model is serverless AWS, defined as code in 430 lines of CDK Python.
>
> Auth is Cognito with two groups, Admins and Students. The student takes the exam, the submit Lambda computes correctness immediately against the answer key — no model needed for that — and kicks off a Step Functions execution that fans out one branch per question, each calling the LoRA adapter for an explanation, then aggregates back into DynamoDB. The student's UI polls until status flips to COMPLETE."

Optional: tab over to the open `medmcqa_stack.py` and scroll through the Step Functions Map definition (lines 244–290). For an infra-savvy audience this lands harder than a diagram alone.

---

## 2. Segment B — Live hybrid demo (5 min)

### 2.1 Admin flow (60 s)

1. Tab to `https://dl.mdaie-sutd.fit`. Click **Sign In**.
2. Login as `demo-admin@mdaie-sutd.fit`.
3. Navigate to **Manage Questions**.
4. Show the existing 5 demo questions. Click **+ Add Question** and create one new question live to demonstrate the CRUD path:
   ```
   Q: A 24-year-old presents with sudden-onset hemiparesis. CT shows hyperdensity
      in the right MCA territory. The most likely cause is:
   A. Hemorrhagic stroke
   B. Ischemic stroke with thromboembolism
   C. Subarachnoid hemorrhage
   D. Brain abscess
   Correct: B
   ```
5. Mention: writes to `medmcqa_questions` DynamoDB table via API Gateway → questions Lambda, JWT-verified by Cognito.

### 2.2 Student flow (90 s)

1. Open a second incognito window. Sign in as `demo-student@mdaie-sutd.fit`.
2. Navigate to **Take Exam**. Show the 5–6 questions loaded from DDB.
3. Answer them — get one wrong on purpose so the audience sees the explanation logic for both correct and incorrect cases.
4. Click **Submit**. The UI immediately shows score (e.g. "4/5 = 80 %") and a "Explanations being generated…" banner.

### 2.3 Step Functions side-show (60 s)

1. Tab to the Step Functions console. The new execution should be visible at the top of the list with status **Running**.
2. Click into it. Show the **Graph view** — the Map state is fanning out 5 branches in parallel, each running `GradeQuestionTask`.
3. After ~2–3 s in demo mode, all branches turn green and `AggregateResultsTask` fires.

> "Each of those green branches just ran the `_demo_explanation()` function with a 250 ms artificial delay. Flip the SSM parameter back to 0 and each branch instead does an HTTPS POST to `api-inference.huggingface.co/models/jamezoon/gemma-3-4b-it-medmcqa-lora` and waits for a real generation — usually 8–15 s per branch, run in parallel."

### 2.4 Result poll-back (30 s)

Tab back to the student window. The polling should already have flipped the page to **Results** with explanations rendered per question.

Read out one of the explanations — the canned MedMCQA-style ones in `demo_explanations.json` are clinically accurate so they hold up to scrutiny.

---

## 3. Segment C — Tie-back (1–2 min)

Return to Slide 15. Point to the right-hand callout panel:

> "The model behind the live demo is the exact artifact our report evaluates. 45.4 % on the 4,183-question MedMCQA dev set, 48.1 % on our 1,609-question MCAT benchmark. We're not showing you a different toy app — we're showing you the LoRA adapter from Section 4 of the report, deployed."

Then advance to the Conclusion slide (Slide 16) and close.

---

## 4. Recovery / fallback procedures

### CloudFront 404 / SPA broken

```bash
# Check the bucket actually has files
aws s3 ls s3://medmcqa-frontend-$(aws sts get-caller-identity --query Account --output text)/ --region ap-southeast-1
# Re-deploy frontend if empty:
cd medmcqa/webapp/frontend && npm run build && aws s3 sync dist/ s3://medmcqa-frontend-XXXXXX/
aws cloudfront create-invalidation --distribution-id <id> --paths '/*'
```

### Step Functions execution fails

Open the failed execution in the console. Most common cause: cold start exceeding the 5 s retry interval. Map state retries 3 times so you usually see green within 30 s. Talking point if it actually fails:

> "Fault tolerance was designed in — the retry policy is 3 attempts, 5 s interval, exponential backoff. In production I'd add an SNS alarm on `ExecutionsFailed`."

### Cognito sign-in 401

Pre-created users sometimes need a force-password-reset before the demo. If a sign-in fails:

```bash
aws cognito-idp admin-set-user-password \
  --user-pool-id "$USER_POOL" \
  --username demo-student@mdaie-sutd.fit \
  --password 'Demo!Student2026' \
  --permanent \
  --region ap-southeast-1
```

### Total network failure

Switch to the pre-recorded screencast slide. Do *not* attempt to debug live — say "we have a recorded run that captures the full path" and play it.

---

## 5. Post-demo cleanup

```bash
# Flip demo mode back off (returns to live HF)
aws ssm put-parameter --name /medmcqa/demo_mode --type String --value 0 \
  --overwrite --region ap-southeast-1

# (Optional) Delete demo submissions so they don't pollute analytics
aws dynamodb scan --table-name medmcqa_submissions --region ap-southeast-1 \
  --filter-expression "studentName = :n" \
  --expression-attribute-values '{":n":{"S":"Student"}}' \
  --query 'Items[].submissionId.S' --output text \
  | xargs -n1 -I{} aws dynamodb delete-item --table-name medmcqa_submissions \
      --key '{"submissionId":{"S":"{}"}}' --region ap-southeast-1
```

---

## 6. Cost (post-demo sanity)

The whole demo with DEMO_MODE=1 costs effectively nothing:

- **Lambda**: 5 invocations × 250 ms each = within free tier
- **DynamoDB**: PAY_PER_REQUEST, < 50 RCUs/WCUs total
- **Step Functions**: 6 state transitions × 1 execution = ~$0.000025
- **CloudFront / S3**: < 100 KB transfer
- **Cognito**: free tier (< 50,000 MAU)

If demo mode is OFF the only added cost is HF Inference API calls — currently free for the published `jamezoon/gemma-3-4b-it-medmcqa-lora` model (community endpoint).

Total: **< $0.01** for a full demo run.

---

## 7. Reference

| Resource | Value |
|---|---|
| URL | https://dl.mdaie-sutd.fit |
| Region | ap-southeast-1 |
| CDK stack | `MedMCQAStack` |
| State machine | `medmcqa-grading` |
| Demo flag | SSM `/medmcqa/demo_mode` |
| HF model (deployed) | `jamezoon/gemma-3-4b-it-medmcqa-lora` |
| HF grader (planned) | `jamezoon/deberta-v3-large-medmcqa` |
| Architecture slide | `Deep_learning_presentation_v4.pptx`, slide 15 |
| Demo bundle | `aws/lambdas/grade/demo_explanations.json` |
