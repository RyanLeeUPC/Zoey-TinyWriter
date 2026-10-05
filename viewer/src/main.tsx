import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router";
import "./index.css";
import { ThemeProvider } from "./lib/theme";
import { Layout } from "./components/Layout";
import { Explore } from "./pages/Explore";
import { RunPage } from "./pages/RunPage";
import { Dictionary } from "./pages/Dictionary";

// HashRouter so the site works on static hosting (GitHub Pages) with no server config.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <HashRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Explore />} />
            <Route path="/run/:runId" element={<RunPage />} />
            <Route path="/dictionary" element={<Dictionary />} />
            <Route path="/dictionary/:tokenId" element={<Dictionary />} />
          </Routes>
        </Layout>
      </HashRouter>
    </ThemeProvider>
  </StrictMode>,
);
