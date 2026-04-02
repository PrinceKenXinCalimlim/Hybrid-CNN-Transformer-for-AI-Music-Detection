import { BrowserRouter, Routes, Route, Navigate } from "react-router";
import Design2Upload from "./pages/Upload";
import Design2Results from "./pages/Results";
import Design2Landing from "./pages/LandingPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/Landingpage" replace />} />
        <Route path="/landingPage" element={<Design2Landing />} />
        <Route path="/upload" element={<Design2Upload/>} />
        <Route path="/results" element={<Design2Results />} />
      </Routes>
    </BrowserRouter>
  );
}