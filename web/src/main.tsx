import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { Hub } from "./Hub";
import { SCENE } from "./api";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {SCENE ? <App /> : <Hub />}
  </StrictMode>,
);
