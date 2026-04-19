/**
 * Auth helpers wrapping aws-amplify v6.
 * Usage:
 *   import { signIn, signOut, getSession, getRole } from "./auth.js"
 */

import {
  signIn as amplifySignIn,
  signOut as amplifySignOut,
  signUp as amplifySignUp,
  confirmSignUp as amplifyConfirmSignUp,
  resendSignUpCode as amplifyResendSignUpCode,
  getCurrentUser,
  fetchAuthSession,
} from "aws-amplify/auth";

/**
 * Thrown when a sign-in is attempted while a session already exists.
 * The caller should prompt the user to confirm replacing the session.
 */
export class ActiveSessionError extends Error {
  constructor(email) {
    super("active_session");
    this.name = "ActiveSessionError";
    this.email = email;
  }
}

/**
 * Sign in with email + password.
 * Throws ActiveSessionError if a session is already active — caller
 * should show a warning and call signInForce() if the user confirms.
 */
export async function signIn(email, password) {
  try {
    return await amplifySignIn({ username: email, password });
  } catch (err) {
    if (
      err?.name === "UserAlreadyAuthenticatedException" ||
      err?.message?.includes("already a signed in user") ||
      err?.message?.includes("UserAlreadyAuthenticated")
    ) {
      throw new ActiveSessionError(email);
    }
    throw err;
  }
}

/**
 * Force sign in: signs out the existing session first, then signs in.
 * Call this after the user confirms they want to replace their session.
 */
export async function signInForce(email, password) {
  await amplifySignOut();
  return amplifySignIn({ username: email, password });
}

/** Register a new student account. Sends a verification email. */
export async function signUp(email, password, name) {
  return amplifySignUp({
    username: email,
    password,
    options: { userAttributes: { email, name } },
  });
}

/** Confirm a new account with the 6-digit verification code. */
export async function confirmSignUp(email, code) {
  return amplifyConfirmSignUp({ username: email, confirmationCode: code });
}

/** Resend the verification code email. */
export async function resendSignUpCode(email) {
  return amplifyResendSignUpCode({ username: email });
}

/** Sign out the current user. */
export async function signOut() {
  return amplifySignOut();
}

/**
 * Returns the current session's ID token JWT string,
 * or null if not authenticated.
 */
export async function getIdToken() {
  try {
    const session = await fetchAuthSession();
    return session.tokens?.idToken?.toString() ?? null;
  } catch {
    return null;
  }
}

/**
 * Decodes the ID token payload (base64) and returns the claims object.
 * No signature verification — Amplify already handles that.
 */
export function decodeToken(idToken) {
  if (!idToken) return null;
  try {
    const payload = idToken.split(".")[1];
    const decoded = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(decoded);
  } catch {
    return null;
  }
}

/**
 * Returns "Admin" | "Student" | null based on the `custom:role` claim
 * injected by the Cognito Pre-Token Generation trigger.
 */
export async function getRole() {
  const token  = await getIdToken();
  const claims = decodeToken(token);
  return claims?.["custom:role"] ?? null;
}

/**
 * Returns basic user info from the current session:
 * { userId, email, name, role }
 */
export async function getCurrentUserInfo() {
  try {
    const [user, token] = await Promise.all([getCurrentUser(), getIdToken()]);
    const claims = decodeToken(token);
    return {
      userId: user.userId,
      email:  claims?.email ?? user.username,
      name:   claims?.name  ?? claims?.email ?? user.username,
      role:   claims?.["custom:role"] ?? "Student",
    };
  } catch {
    return null;
  }
}

/** Returns true if a valid session exists. */
export async function isAuthenticated() {
  const token = await getIdToken();
  return !!token;
}
