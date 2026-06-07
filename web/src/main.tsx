import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";
import "./index.css";
import RunList from "./pages/RunList";
import LiveDashboard from "./pages/LiveDashboard";
import ReportViewer from "./pages/ReportViewer";

const router = createBrowserRouter([
  { path: "/", element: <RunList /> },
  { path: "/live/:id", element: <LiveDashboard /> },
  { path: "/report/:id", element: <ReportViewer /> },
  { path: "*", element: <Navigate to="/" replace /> },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
