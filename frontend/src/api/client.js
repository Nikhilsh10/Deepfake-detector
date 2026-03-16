/**
 * Axios-based API client for the Deepfake Detection backend.
 *
 * Every function is async with try/catch error handling and returns
 * either the response data or throws a user-friendly error.
 */

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 minutes for large video uploads
});

/**
 * Upload an image file for deepfake detection.
 * @param {File} file - Image file (jpg, png, webp)
 * @returns {Promise<Object>} Detection result
 */
export async function detectImage(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post('/detect/image', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.error ||
      'Image analysis failed. Please try again.'
    );
  }
}

/**
 * Upload a video file for deepfake detection.
 * @param {File} file - Video file (mp4, avi, mov, mkv)
 * @returns {Promise<Object>} Detection result
 */
export async function detectVideo(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post('/detect/video', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.error ||
      'Video analysis failed. Please try again.'
    );
  }
}

/**
 * Send a base64-encoded webcam frame for detection.
 * @param {string} base64 - Base64 image string (no data: prefix)
 * @returns {Promise<Object>} Detection result
 */
export async function detectWebcam(base64) {
  try {
    const { data } = await api.post('/detect/webcam', {
      image_base64: base64,
    });
    return data;
  } catch (error) {
    throw new Error(
      error.response?.data?.detail ||
      error.response?.data?.error ||
      'Webcam analysis failed. Please try again.'
    );
  }
}

/**
 * Build the full URL for a heatmap image.
 * @param {string} filename - Heatmap filename
 * @returns {string} Full URL
 */
export function getHeatmapUrl(filename) {
  return `${BASE_URL}/heatmap/${encodeURIComponent(filename)}`;
}

/**
 * Build the full URL for an extracted frame image.
 * @param {string} filename - Frame filename (may include subfolder)
 * @returns {string} Full URL
 */
export function getFrameUrl(filename) {
  return `${BASE_URL}/frame/${encodeURIComponent(filename)}`;
}

export default api;
