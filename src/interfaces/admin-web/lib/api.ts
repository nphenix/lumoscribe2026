import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || '/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const data = error.response?.data;
    const err = data?.error || data;
    const message =
      err?.message ||
      error.message ||
      'An unknown error occurred';
    const details = err?.details;
    const firstValidationError =
      details?.errors &&
      Array.isArray(details.errors) &&
      details.errors.length > 0
        ? details.errors[0]
        : null;
    const requestId = err?.request_id;
    if (firstValidationError) {
      console.warn('API Error:', message, firstValidationError, requestId ? { requestId } : {});
    } else {
      console.warn('API Error:', message, requestId ? { requestId } : {});
    }
    return Promise.reject(error);
  }
);

export default api;
