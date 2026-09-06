/*
 * Browser-to-API configuration.
 *
 * Leave apiBaseUrl empty when FastAPI serves this UI (the standard setup).
 * When hosting frontend/public from a different local server, set it to the
 * FastAPI origin, e.g. "http://127.0.0.1:8000". CORS origins are configured
 * by the backend's CORS_ALLOW_ORIGINS environment variable.
 */
window.RAZORPAY_AI_CONFIG = {
    apiBaseUrl: ""
};
