import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, StopCircle, ShieldAlert, ShieldCheck, Loader2 } from 'lucide-react';
import { detectWebcam } from '../api/client';

const CAPTURE_INTERVAL_MS = 1500;

/**
 * LiveDetect — real-time webcam deepfake detection.
 * Captures a frame every 1.5 s while active, sends it as base64 to
 * the backend, and overlays the verdict on the video feed.
 */
export default function LiveDetect() {
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);

  const [active, setActive] = useState(false);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  /** Capture a frame, strip the data‑URI prefix, and send base64. */
  const captureAndAnalyze = useCallback(async () => {
    if (!webcamRef.current) return;

    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return;

    // Remove "data:image/jpeg;base64," prefix
    const base64 = screenshot.split(',')[1];
    if (!base64) return;

    try {
      setAnalyzing(true);
      const data = await detectWebcam(base64);
      setResult(data);
      setError('');
    } catch (err) {
      setError(err.message || 'Webcam analysis failed.');
    } finally {
      setAnalyzing(false);
    }
  }, []);

  /** Start / stop detection loop. */
  const toggleDetection = () => {
    if (active) {
      // Stop
      clearInterval(intervalRef.current);
      intervalRef.current = null;
      setActive(false);
      setResult(null);
    } else {
      // Start
      setActive(true);
      setError('');
      captureAndAnalyze(); // fire immediately
      intervalRef.current = setInterval(captureAndAnalyze, CAPTURE_INTERVAL_MS);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const isFake = result?.prediction === 'FAKE';
  const conf = result ? (result.confidence * 100).toFixed(1) : null;

  return (
    <div className="space-y-6">
      {/* Webcam feed */}
      <div className="relative rounded-2xl overflow-hidden border border-slate-700">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{ facingMode: 'user', width: 640, height: 480 }}
          className="w-full"
        />

        {/* Verdict overlay */}
        <AnimatePresence>
          {active && result && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className={`
                absolute inset-0 flex flex-col items-center justify-center
                pointer-events-none
                ${isFake ? 'bg-red-500/20' : 'bg-green-500/20'}
              `}
            >
              <div
                className={`
                  flex items-center gap-2 px-5 py-2.5 rounded-full text-lg font-bold
                  backdrop-blur-sm
                  ${isFake
                    ? 'bg-red-900/60 text-red-300 border border-red-500/40'
                    : 'bg-green-900/60 text-green-300 border border-green-500/40'}
                `}
              >
                {isFake ? (
                  <ShieldAlert className="w-6 h-6" />
                ) : (
                  <ShieldCheck className="w-6 h-6" />
                )}
                {result.prediction}
              </div>
              <span className="mt-2 text-sm font-semibold text-white/80 backdrop-blur-sm px-3 py-1 rounded-lg bg-black/30">
                {conf}% confidence
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Analysing spinner (top‑right) */}
        {analyzing && (
          <div className="absolute top-3 right-3">
            <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <button
          id="live-detect-toggle"
          onClick={toggleDetection}
          className={`
            flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm
            transition-all duration-200 shadow-lg
            ${active
              ? 'bg-red-600 hover:bg-red-700 shadow-red-600/25'
              : 'bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 shadow-indigo-500/25'}
          `}
        >
          {active ? (
            <>
              <StopCircle className="w-5 h-5" />
              Stop Detection
            </>
          ) : (
            <>
              <Camera className="w-5 h-5" />
              Start Detection
            </>
          )}
        </button>
      </div>

      {/* Live stats */}
      {active && result && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-4 grid grid-cols-3 gap-4 text-center"
        >
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Verdict</p>
            <p className={`text-lg font-bold ${isFake ? 'text-red-400' : 'text-green-400'}`}>
              {result.prediction}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Confidence</p>
            <p className="text-lg font-bold text-white">{conf}%</p>
          </div>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider">Fake Prob</p>
            <p className="text-lg font-bold text-red-400">
              {(result.fake_probability * 100).toFixed(1)}%
            </p>
          </div>
        </motion.div>
      )}

      {/* Error */}
      {error && (
        <p className="text-sm text-red-400 text-center">{error}</p>
      )}
    </div>
  );
}
