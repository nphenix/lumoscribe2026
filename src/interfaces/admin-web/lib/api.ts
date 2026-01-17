import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Standardized error handling can go here
    const data = error.response?.data;
    const message = data?.message || error.message || 'An unknown error occurred';
    const firstValidationError =
      data?.details?.errors && Array.isArray(data.details.errors) && data.details.errors.length > 0
        ? data.details.errors[0]
        : null;
    if (firstValidationError) {
      console.error('API Error:', message, firstValidationError);
    } else {
      console.error('API Error:', message);
    }
    return Promise.reject(error);
  }
);

export default api;
