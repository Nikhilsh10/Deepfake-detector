import React from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Clock, Layers } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

const COLORS = {
  fake: '#ef4444',
  real: '#22c55e',
  ring_bg: '#334155',
};

/**
 * Circular SVG progress ring that displays a confidence percentage.
 */
function ConfidenceRing({ value, color, size = 120 }) {
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={COLORS.ring_bg}
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="progress-ring-circle"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-white">
          {Math.round(value)}%
        </span>
        <span className="text-[10px] text-slate-400 uppercase tracking-wider">
          confidence
        </span>
      </div>
    </div>
  );
}

/**
 * ResultCard — displays the detection verdict, confidence ring,
 * probability bars, frame stats, and a pie chart.
 */
export default function ResultCard({ result }) {
  if (!result) return null;

  const isFake = result.prediction === 'FAKE';
  const accentColor = isFake ? COLORS.fake : COLORS.real;
  const confidence = (result.confidence * 100).toFixed(1);
  const fakePercent = (result.fake_probability * 100).toFixed(1);
  const realPercent = (result.real_probability * 100).toFixed(1);

  // Pie chart data
  const pieData = result.fake_frames != null
    ? [
        { name: 'Fake', value: result.fake_frames },
        { name: 'Real', value: result.real_frames },
      ]
    : [
        { name: 'Fake', value: result.fake_probability },
        { name: 'Real', value: result.real_probability },
      ];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 22 }}
      className="glass rounded-2xl p-6 space-y-6"
    >
      {/* Verdict badge */}
      <div className="flex items-center justify-center">
        <div
          className={`
            flex items-center gap-3 px-6 py-3 rounded-full text-lg font-bold
            ${isFake
              ? 'bg-red-500/15 text-red-400 border border-red-500/30'
              : 'bg-green-500/15 text-green-400 border border-green-500/30'}
          `}
        >
          {isFake ? (
            <ShieldAlert className="w-6 h-6" />
          ) : (
            <ShieldCheck className="w-6 h-6" />
          )}
          {isFake ? '🔴 DEEPFAKE DETECTED' : '🟢 AUTHENTIC VIDEO'}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        {/* Confidence ring */}
        <div className="flex justify-center">
          <ConfidenceRing value={parseFloat(confidence)} color={accentColor} />
        </div>

        {/* Probability bars */}
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs mb-1 text-slate-400">
              <span>Fake Probability</span>
              <span className="text-red-400 font-semibold">{fakePercent}%</span>
            </div>
            <div className="h-2.5 rounded-full bg-slate-700 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${fakePercent}%` }}
                transition={{ duration: 0.8 }}
                className="h-full rounded-full bg-gradient-to-r from-red-500 to-red-400"
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1 text-slate-400">
              <span>Real Probability</span>
              <span className="text-green-400 font-semibold">{realPercent}%</span>
            </div>
            <div className="h-2.5 rounded-full bg-slate-700 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${realPercent}%` }}
                transition={{ duration: 0.8 }}
                className="h-full rounded-full bg-gradient-to-r from-green-500 to-green-400"
              />
            </div>
          </div>
        </div>

        {/* Pie chart */}
        <div className="flex justify-center">
          <ResponsiveContainer width={140} height={140}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={60}
                paddingAngle={4}
                dataKey="value"
                animationBegin={200}
                animationDuration={800}
              >
                <Cell fill={COLORS.fake} />
                <Cell fill={COLORS.real} />
              </Pie>
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px',
                  color: '#e2e8f0',
                  fontSize: '12px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Frame stats (video only) */}
      {result.total_frames_analyzed != null && (
        <div className="flex flex-wrap items-center justify-center gap-6 text-sm pt-2 border-t border-slate-700/50">
          <div className="flex items-center gap-2 text-slate-400">
            <Layers className="w-4 h-4" />
            <span>
              <strong className="text-white">{result.total_frames_analyzed}</strong> frames
              analyzed
            </span>
          </div>
          <div className="flex items-center gap-2 text-red-400">
            <span>
              <strong>{result.fake_frames}</strong> fake
            </span>
          </div>
          <div className="flex items-center gap-2 text-green-400">
            <span>
              <strong>{result.real_frames}</strong> real
            </span>
          </div>
          {result.processing_time_seconds != null && (
            <div className="flex items-center gap-2 text-slate-400">
              <Clock className="w-4 h-4" />
              <span>{result.processing_time_seconds}s</span>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
