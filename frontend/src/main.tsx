import "../stylesheets/main.css";
import { StrictMode } from "react";
import { Chat } from "./pages/chat";
import { Landing } from "./pages/landing";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/chat/:id" element={<Chat />} />
        <Route path="/" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
