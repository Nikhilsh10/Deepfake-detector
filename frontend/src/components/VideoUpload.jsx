import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Film, Image, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { detectVideo, detectImage } from '../api/client';
import ResultCard from './ResultCard';
import HeatmapView from './HeatmapView';
import FrameStrip from './FrameStrip';

const PROGRESS_STEPS = [
  { label: 'Uploading file…', icon: Upload },
  { label: 'Extracting frames…', icon: Film },
  { label: 'Analyzing frames…', icon: Image },
  { label: 'Generating heatmaps…', icon: CheckCircle2 },
];

/**
 * VideoUpload — drag-and-drop uploader for videos and images.
 * Shows animated progress steps during analysis and renders
 * ResultCard / HeatmapView / FrameStrip upon completion.
 */
export default function VideoUpload({ mode = 'video' }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const isVideo = mode === 'video';
  const accept = isVideo
    ? { 'video/*': ['.mp4', '.avi', '.mov', '.mkv'] }
    : { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] };

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setResult(null);
      setError('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: 1,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Simulate progress steps for a better UX
      for (let i = 0; i < PROGRESS_STEPS.length; i++) {
        setProgressStep(i);
        if (i < PROGRESS_STEPS.length - 1) {
          await new Promise((r) => setTimeout(r, 600));
        }
      }

      const data = isVideo ? await detectVideo(file) : await detectImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
      setProgressStep(0);
    }
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-6">
      {/* Drop zone */}
      <div
        {...getRootProps()}
        id="upload-dropzone"
        className={`
          border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer
          transition-all duration-300
          ${isDragActive
            ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
            : 'border-slate-600 hover:border-indigo-500 hover:bg-slate-800/40'}
        `}
      >
        <input {...getInputProps()} />
        <Upload className="w-12 h-12 mx-auto mb-4 text-indigo-400" />
        <p className="text-lg font-medium text-slate-200">
          {isDragActive
            ? 'Drop the file here…'
            : `Drag & drop a ${isVideo ? 'video' : 'image'} file, or click to browse`}
        </p>
        <p className="text-sm text-slate-500 mt-2">
          {isVideo ? 'MP4, AVI, MOV, MKV' : 'JPG, JPEG, PNG, WEBP'}
        </p>
      </div>

      {/* Selected file */}
      <AnimatePresence>
        {file && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="glass rounded-xl p-4 flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              {isVideo ? (
                <Film className="w-6 h-6 text-indigo-400" />
              ) : (
                <Image className="w-6 h-6 text-indigo-400" />
              )}
              <div>
                <p className="text-sm font-medium text-white truncate max-w-xs">
                  {file.name}
                </p>
                <p className="text-xs text-slate-400">{formatSize(file.size)}</p>
              </div>
            </div>

            <button
              id="analyze-button"
              onClick={handleAnalyze}
              disabled={loading}
              className="px-5 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200
                bg-gradient-to-r from-indigo-500 to-purple-600
                hover:from-indigo-600 hover:to-purple-700
                disabled:opacity-50 disabled:cursor-not-allowed
                shadow-lg shadow-indigo-500/25"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing…
                </span>
              ) : (
                `Analyze ${isVideo ? 'Video' : 'Image'}`
              )}
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Progress steps */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass rounded-xl p-6"
          >
            <div className="space-y-3">
              {PROGRESS_STEPS.map((step, idx) => {
                const Icon = step.icon;
                const active = idx === progressStep;
                const done = idx < progressStep;

                return (
                  <div
                    key={step.label}
                    className={`flex items-center gap-3 transition-colors duration-300 ${
                      active
                        ? 'text-indigo-400'
                        : done
                        ? 'text-green-400'
                        : 'text-slate-600'
                    }`}
                  >
                    {active ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : done ? (
                      <CheckCircle2 className="w-5 h-5" />
                    ) : (
                      <Icon className="w-5 h-5" />
                    )}
                    <span className="text-sm font-medium">{step.label}</span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30"
        >
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </motion.div>
      )}

      {/* Results */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="space-y-6"
        >
          <ResultCard result={result} />
          {result.frame_results && result.frame_results.length > 0 && (
            <>
              <FrameStrip frames={result.frame_results} />
              <HeatmapView frames={result.frame_results} />
            </>
          )}
        </motion.div>
      )}
    </div>
  );
}
