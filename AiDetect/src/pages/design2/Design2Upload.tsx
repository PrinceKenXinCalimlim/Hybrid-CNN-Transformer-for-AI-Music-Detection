import { useState } from "react";
import { useNavigate } from "react-router";
import { Upload, ArrowLeft } from "lucide-react";

export default function Design2Upload() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setSelectedFile(e.target.files[0]);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) setSelectedFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = () => {
    if (selectedFile)
      navigate("/design2/results", { state: { fileName: selectedFile.name } });
  };

  return (
    <div className="min-h-screen flex">

      {/* LEFT PANEL — fixed */}
      <div className="fixed top-0 left-0 h-full w-1/2 bg-gradient-to-br from-emerald-600 to-teal-600 flex flex-col justify-center p-6 sm:p-8 lg:p-12 overflow-hidden z-10">

        {/* Back button */}
        <button
          onClick={() => navigate("/")}
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
          <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10 sm:w-12 sm:h-12 lg:w-16 lg:h-16 text-white mb-4 lg:mb-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="2"/>
            <path d="M8.56 2.9A7 7 0 0 1 19.07 19"/>
            <path d="M4.93 4.93a10 10 0 0 0 0 14.14"/>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
          </svg>
          <h1 className="text-2xl sm:text-3xl lg:text-5xl font-bold text-white mb-3 lg:mb-4 leading-tight">
            AI Music<br />Detection System
          </h1>
          <p className="text-sm sm:text-base lg:text-xl text-emerald-100">
            Advanced audio analysis powered by machine learning
          </p>
        </div>
      </div>

      {/* RIGHT PANEL — scrollable, responsive */}
      <div className="ml-[50%] w-1/2 min-h-screen bg-white flex items-center justify-center p-6 sm:p-8 lg:p-12">
        <div className="w-full max-w-md">
          <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold text-gray-900 mb-2">Upload Audio</h2>
          <p className="text-sm sm:text-base text-gray-600 mb-6 lg:mb-8">Select an audio file to begin analysis</p>

          {/* Drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            className={`border-2 border-dashed rounded-2xl p-6 sm:p-8 lg:p-12 text-center transition-all mb-6 lg:mb-8 cursor-pointer ${
              isDragging || selectedFile
                ? "border-emerald-500 bg-emerald-50"
                : "border-gray-300 hover:border-gray-400"
            }`}
          >
            <Upload className={`w-8 h-8 sm:w-10 sm:h-10 lg:w-12 lg:h-12 mx-auto mb-3 lg:mb-4 ${selectedFile ? "text-emerald-600" : "text-gray-400"}`} />
            {selectedFile ? (
              <>
                <p className="font-semibold text-gray-900 mb-1 text-sm sm:text-base truncate">{selectedFile.name}</p>
                <p className="text-xs sm:text-sm text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
              </>
            ) : (
              <>
                <p className="text-gray-900 font-medium mb-1 text-sm sm:text-base">Drop your audio file here</p>
                <p className="text-xs sm:text-sm text-gray-500">MP3, WAV, FLAC, M4A</p>
              </>
            )}
          </div>

          <input type="file" accept="audio/*" onChange={handleFileSelect} className="hidden" id="file-upload" />

          <div className="space-y-3">
            <label htmlFor="file-upload" className="block">
              <span className="flex items-center justify-center w-full px-4 py-2.5 sm:py-3 border-2 border-gray-300 rounded-xl text-gray-700 font-semibold text-sm cursor-pointer hover:border-gray-400 hover:bg-gray-50 transition-all">
                Browse Files
              </span>
            </label>
            <button
              onClick={handleAnalyze}
              disabled={!selectedFile}
              className="w-full px-4 py-2.5 sm:py-3 rounded-xl text-white font-semibold text-sm bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              Start Analysis
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}