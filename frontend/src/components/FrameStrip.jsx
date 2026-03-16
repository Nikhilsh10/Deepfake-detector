import React from 'react';
import { motion } from 'framer-motion';
import { getHeatmapUrl } from '../api/client';

/**
 * FrameStrip — horizontal scrollable strip of extracted frames.
 * Each frame shows a prediction label and confidence percentage,
 * tinted red for FAKE and green for REAL.
 */
export default function FrameStrip({ frames = [] }) {
  if (frames.length === 0) return null;

  /** Extract just the filename part from a full/relative path. */
  const basename = (p) => {
    if (!p) return '';
    return p.split(/[\\/]/).pop();
  };

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-white">Frame Analysis Strip</h3>

      <div className="flex gap-3 overflow-x-auto pb-3 scrollbar-thin">
        {frames.map((frame, idx) => {
          const isFake = frame.prediction === 'FAKE';
          const conf = (frame.confidence * 100).toFixed(0);
          const heatmapFile = basename(frame.heatmap_path);

          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.05 }}
              className={`
                flex-shrink-0 w-36 rounded-xl overflow-hidden
                border transition-colors
                ${isFake ? 'border-red-500/50' : 'border-green-500/50'}
              `}
            >
              {/* Frame image with color tint overlay */}
              <div className="relative">
                <img
                  src={getHeatmapUrl(heatmapFile)}
                  alt={`Frame ${idx + 1}`}
                  className="w-full h-24 object-cover"
                  loading="lazy"
                />
                {/* Colour overlay */}
                <div
                  className={`absolute inset-0 ${
                    isFake ? 'bg-red-500/15' : 'bg-green-500/15'
                  }`}
                />
              </div>

              {/* Footer */}
              <div className="bg-slate-800/80 px-2 py-1.5 flex items-center justify-between">
                <span
                  className={`text-[11px] font-bold ${
                    isFake ? 'text-red-400' : 'text-green-400'
                  }`}
                >
                  {frame.prediction}
                </span>
                <span className="text-[11px] text-slate-400 font-medium">
                  {conf}%
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
