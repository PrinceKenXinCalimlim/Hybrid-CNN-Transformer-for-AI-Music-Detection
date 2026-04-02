import { useNavigate, Link } from "react-router";
import { Radio, ArrowLeft } from "lucide-react";

export default function Design2Landing() {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate("/upload");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-600 to-teal-600 p-12 flex flex-col justify-center items-center relative overflow-hidden">

      {/* Animated waveform background */}
      <div className="absolute inset-0 flex items-end justify-around px-8 overflow-hidden pointer-events-none" aria-hidden="true">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className="w-1 bg-white rounded-full"
            style={{
              height: `${20 + Math.random() * 75}%`,
              opacity: 0.15,
              transformOrigin: "bottom",
              animation: `wave ${1 + (i % 3) * 0.4}s ease-in-out infinite`,
              animationDelay: `${(i * 0.05) % 1}s`,
            }}
          />
        ))}
      </div>

      {/* Main content */}
      <div className="relative z-10 max-w-xl w-full text-center">

        {/* Icon */}
        <div className="inline-flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-white/10 backdrop-blur-sm mb-6 sm:mb-8">
          <Radio className="w-8 h-8 sm:w-10 sm:h-10 text-white" />
        </div>

        {/* Title */}
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-4 sm:mb-6 leading-tight">
          AI Music Detection System
        </h1>

        {/* Subtitle */}
        <p className="text-lg sm:text-xl lg:text-2xl text-emerald-50 mb-6 sm:mb-8 leading-relaxed">
          Leverage cutting-edge machine learning to identify artificial intelligence in audio composition
        </p>

        {/* Supported formats badge */}
        <div className="flex flex-wrap gap-4 justify-center mb-6 sm:mb-8">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
            <p className="text-white/80 text-xs sm:text-sm">Supported Formats</p>
            <p className="text-white font-semibold text-sm sm:text-base">MP3 • WAV • FLAC • M4A</p>
          </div>
        </div>

        {/* Get Started button */}
        <button
          onClick={handleGetStarted}
          className="w-full max-w-md bg-white text-emerald-600 hover:bg-emerald-50 active:bg-emerald-100 font-semibold text-base sm:text-lg h-12 sm:h-14 rounded-xl transition-all shadow-lg hover:shadow-xl"
        >
          Get Started
        </button>

      </div>
    </div>
  );
}