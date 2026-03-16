import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Video, ImageIcon, Webcam } from 'lucide-react';
import VideoUpload from './components/VideoUpload';
import LiveDetect from './components/LiveDetect';

const TABS = [
  { id: 'video', label: 'Upload Video', icon: Video },
  { id: 'image', label: 'Upload Image', icon: ImageIcon },
  { id: 'webcam', label: 'Live Webcam', icon: Webcam },
];

/**
 * App — root component.
 * Dark‑theme shell with a top navbar, tab navigation, and
 * conditional rendering of the active feature panel.
 */
export default function App() {
  const [activeTab, setActiveTab] = useState('video');

  return (
    <div className="min-h-screen">
      {/* ── Navbar ────────────────────────────────────────────────── */}
      <nav className="glass sticky top-0 z-40 border-b border-slate-700/50">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-5 py-3">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/20">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-bold gradient-text">
              DeepGuard
            </span>
          </div>

          {/* Badge */}
          <span className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 bg-slate-800/60 px-3 py-1.5 rounded-full border border-slate-700/50">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            EfficientNet-B4
          </span>
        </div>
      </nav>

      {/* ── Main content ─────────────────────────────────────────── */}
      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        {/* Hero */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl sm:text-4xl font-extrabold gradient-text">
            Real-Time Deepfake Detection
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto text-sm">
            Powered by EfficientNet-B4 &amp; Grad-CAM — upload a video, image,
            or use your webcam for instant AI analysis
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex justify-center">
          <div className="inline-flex gap-1 p-1 rounded-xl bg-slate-800/60 border border-slate-700/50">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                    transition-all duration-200
                    ${isActive
                      ? 'text-white'
                      : 'text-slate-400 hover:text-slate-200'}
                  `}
                >
                  {isActive && (
                    <motion.div
                      layoutId="tab-bg"
                      className="absolute inset-0 rounded-lg bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border border-indigo-500/30"
                      transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                    />
                  )}
                  <Icon className="w-4 h-4 relative z-10" />
                  <span className="relative z-10 hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Active panel */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.2 }}
          >
            {activeTab === 'video' && <VideoUpload mode="video" />}
            {activeTab === 'image' && <VideoUpload mode="image" />}
            {activeTab === 'webcam' && <LiveDetect />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer className="text-center py-6 text-xs text-slate-600">
        DeepGuard &copy; {new Date().getFullYear()} — Built with PyTorch,
        FastAPI &amp; React
      </footer>
    </div>
  );
}
