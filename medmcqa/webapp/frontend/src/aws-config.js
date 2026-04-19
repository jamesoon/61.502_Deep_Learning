/**
 * AWS / Amplify configuration.
 * Values are injected at build time via Vite env vars (VITE_*).
 * After CDK deploy, set these in .env.production from the stack outputs:
 *
 *   VITE_USER_POOL_ID=ap-southeast-1_XXXXXXXXX
 *   VITE_USER_POOL_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
 *   VITE_API_ENDPOINT=https://xxxxxxxxxx.execute-api.ap-southeast-1.amazonaws.com
 *   VITE_REGION=ap-southeast-1
 *
 * For local dev pointing at the deployed API, copy .env.production → .env.local
 */

export const awsConfig = {
  Auth: {
    Cognito: {
      userPoolId:       import.meta.env.VITE_USER_POOL_ID       || "ap-southeast-1_PLACEHOLDER",
      userPoolClientId: import.meta.env.VITE_USER_POOL_CLIENT_ID || "PLACEHOLDER",
      loginWith: { email: true },
    },
  },
};

/**
 * API Gateway base URL.
 * - Dev:  leave VITE_API_ENDPOINT unset → Vite proxy forwards /api/* to localhost:8000/api
 * - Prod: set VITE_API_ENDPOINT=https://xxx.execute-api.ap-southeast-1.amazonaws.com/prod
 *         in webapp/frontend/.env.production (from SAM stack outputs)
 */
export const API_BASE = import.meta.env.VITE_API_ENDPOINT || "/api";
