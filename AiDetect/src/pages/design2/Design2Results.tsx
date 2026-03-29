import { useLocation, useNavigate } from "react-router";
import { AlertCircle, CheckCircle, ArrowLeft, RotateCcw } from "lucide-react";

export default function Design2Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const fileName = location.state?.fileName || "unknown.mp3";

  const isAIGenerated = true;
  const confidence = 89;
  const patternScore = 90;
  const spectralScore = 88;
  const harmonicScore = 85;

  return (
    <div className="min-h-screen flex">

      {/* LEFT PANEL — fixed */}
      <div className={`fixed top-0 left-0 h-full w-1/2 flex flex-col justify-center p-6 sm:p-8 lg:p-12 overflow-hidden z-10 bg-gradient-to-br ${
        isAIGenerated ? "from-orange-600 to-red-600" : "from-emerald-600 to-teal-600"
      }`}>

        {/* Back button */}
        <button
          onClick={() => navigate("/design2")}
          className="absolute top-6 left-6 inline-flex items-center gap-2 text-white/80 hover:text-white text-sm transition-all px-3 py-2 rounded-lg hover:bg-white/20 active:bg-white/30"
        >
          <ArrowLeft className="w-4 h-4" /> Back
        </button>

        {/* Animated waveform bars */}
        <div className="absolute inset-0 flex items-end justify-around px-4 overflow-hidden pointer-events-none" aria-hidden="true">
          {[...Array(40)].map((_, i) => (
            <div
              key={i}
              className="w-1 bg-white rounded-full"
              style={{
                height: `${20 + Math.random() * 75}%`,
                opacity: 0.2,
                transformOrigin: "bottom",
                animation: `wave ${1 + (i % 3) * 0.4}s ease-in-out infinite`,
                animationDelay: `${(i * 0.05) % 1}s`,
              }}
            />
          ))}
        </div>

        {/* Hero content */}
        <div className="relative z-10">
          {isAIGenerated
            ? <AlertCircle className="w-12 h-12 sm:w-16 sm:h-16 lg:w-20 lg:h-20 text-white mb-4 lg:mb-6" />
            : <CheckCircle className="w-12 h-12 sm:w-16 sm:h-16 lg:w-20 lg:h-20 text-white mb-4 lg:mb-6" />
          }
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-3 lg:mb-4">
            {confidence}<span className="text-2xl sm:text-3xl lg:text-4xl">%</span>
          </h1>
          <p className="text-lg sm:text-xl lg:text-2xl font-semibold text-white mb-2">
            {isAIGenerated ? "AI-Generated" : "Human-Created"}
          </p>
          <p className="text-sm sm:text-base lg:text-xl text-white/80">Detection confidence level</p>
        </div>
      </div>

      {/* RIGHT PANEL — scrollable, responsive */}
      <div className="ml-[50%] w-1/2 min-h-screen bg-white flex items-center justify-center p-6 sm:p-8 lg:p-12">
        <div className="w-full max-w-md">

          {/* File info */}
          <div className="flex items-start gap-3 mb-5 lg:mb-6">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 sm:w-7 sm:h-7 lg:w-8 lg:h-8 text-gray-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="2"/>
              <path d="M8.56 2.9A7 7 0 0 1 19.07 19"/>
              <path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
            </svg>
            <div className="overflow-hidden">
              <p className="text-xs sm:text-sm text-gray-500">Analyzed File</p>
              <p className="font-semibold text-gray-900 leading-snug text-sm sm:text-base truncate">{fileName}</p>
            </div>
          </div>

          <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 mb-5 lg:mb-6">Analysis Results</h2>

          {/* Score bars */}
          <div className="space-y-4 lg:space-y-6 mb-6 lg:mb-8">
            {[
              { label: "Pattern Recognition", value: patternScore },
              { label: "Spectral Analysis",   value: spectralScore },
              { label: "Harmonic Structure",  value: harmonicScore },
            ].map(({ label, value }) => (
              <div key={label}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-xs sm:text-sm font-medium text-gray-700">{label}</span>
                  <span className="text-xs sm:text-sm font-bold text-gray-900">{value}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-1.5 sm:h-2">
                  <div className="bg-gray-900 h-full rounded-full transition-all" style={{ width: `${value}%` }} />
                </div>
              </div>
            ))}
          </div>

          {/* Summary box */}
          <div className={`rounded-xl p-4 sm:p-5 lg:p-6 mb-6 lg:mb-8 border ${
            isAIGenerated ? "bg-orange-50 border-orange-200" : "bg-emerald-50 border-emerald-200"
          }`}>
            <h3 className="font-semibold text-gray-900 mb-2 text-sm sm:text-base">Summary</h3>
            <p className="text-xs sm:text-sm text-gray-700 leading-relaxed">
              {isAIGenerated
                ? "The analysis reveals patterns and characteristics commonly associated with AI-generated music, including algorithmic composition markers and synthetic sound properties."
                : "This audio demonstrates organic qualities typical of human-created music, with natural variations in dynamics, timing, and musical expression."}
            </p>
          </div>

          {/* CTA button */}
          <button
            onClick={() => navigate("/design2")}
            className={`w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 sm:py-3 rounded-xl text-white font-semibold text-sm transition-all bg-gradient-to-r ${
              isAIGenerated
                ? "from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700"
                : "from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
            }`}
          >
            <RotateCcw className="w-4 h-4 sm:w-5 sm:h-5" />
            Analyze Another File
          </button>

        </div>
      </div>
    </div>
  );
}