import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { BrowserPreviewBenchmarkRoute, isBrowserPreviewBenchmarkPath } from "./bench/browserPreviewBenchmarkRoute";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {isBrowserPreviewBenchmarkPath() ? <BrowserPreviewBenchmarkRoute /> : <App />}
  </React.StrictMode>
);
