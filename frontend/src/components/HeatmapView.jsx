import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ZoomIn } from 'lucide-react';
import { getHeatmapUrl } from '../api/client';

/**
 * HeatmapView — grid of frame thumbnails with click-to-zoom
 * Grad-CAM heatmap modal.  Each thumbnail has a colour-coded
 * border (red = FAKE, green = REAL) and a confidence badge.
 */
export default function HeatmapView({ frames = [] }) {
  const [selected, setSelected] = useState(null);

  if (frames.length === 0) return null;

  /** Extract just the filename part from a full/relative path. */
  const basename = (p) => {
    if (!p) return '';
    return p.split(/[\\/]/).pop();
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white flex items-center gap-2">
        <ZoomIn className="w-5 h-5 text-indigo-400" />
        Grad-CAM Heatmaps
      </h3>

      {/* Thumbnail grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {frames.map((frame, idx) => {
          const isFake = frame.prediction === 'FAKE';
          const heatmapFile = basename(frame.heatmap_path);
          const conf = (frame.confidence * 100).toFixed(0);

          return (
            <motion.button
              key={idx}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => setSelected(frame)}
              className={`
                relative rounded-xl overflow-hidden
                border-2 transition-colors
                ${isFake ? 'border-red-500/70' : 'border-green-500/70'}
                focus:outline-none focus:ring-2 focus:ring-indigo-400
              `}
            >
              <img
                src={getHeatmapUrl(heatmapFile)}
                alt={`Frame ${idx + 1} heatmap`}
                className="w-full h-28 object-cover"
                loading="lazy"
              />
              {/* Label */}
              <span
                className={`
                  absolute top-1.5 left-1.5 text-[10px] font-bold px-1.5 py-0.5 rounded-md
                  ${isFake ? 'bg-red-600/80 text-white' : 'bg-green-600/80 text-white'}
                `}
              >
                {frame.prediction}
              </span>
              {/* Confidence badge */}
              <span className="absolute bottom-1.5 right-1.5 text-[10px] font-semibold px-1.5 py-0.5 rounded-md bg-black/60 text-white">
                {conf}%
              </span>
            </motion.button>
          );
        })}
      </div>

      {/* Modal */}
      <AnimatePresence>
        {selected && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
            onClick={() => setSelected(null)}
          >
            <motion.div
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 260, damping: 24 }}
              className="glass rounded-2xl overflow-hidden max-w-2xl w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
                <span
                  className={`text-sm font-bold px-3 py-1 rounded-full ${
                    selected.prediction === 'FAKE'
                      ? 'bg-red-500/20 text-red-400'
                      : 'bg-green-500/20 text-green-400'
                  }`}
                >
                  {selected.prediction} —{' '}
                  {(selected.confidence * 100).toFixed(1)}% confidence
                </span>
                <button
                  onClick={() => setSelected(null)}
                  className="p-1.5 rounded-lg hover:bg-slate-700 transition-colors"
                >
                  <X className="w-5 h-5 text-slate-400" />
                </button>
              </div>
              <img
                src={getHeatmapUrl(basename(selected.heatmap_path))}
                alt="Full heatmap"
                className="w-full max-h-[70vh] object-contain bg-black/40"
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
