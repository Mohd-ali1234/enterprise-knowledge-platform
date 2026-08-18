import { Routes, Route, Navigate } from "react-router-dom";
import AppLayout from "@/components/layout/AppLayout.jsx";
import Home from "@/pages/Home.jsx";
import AskKnowledge from "@/pages/AskKnowledge.jsx";
import Documents from "@/pages/Documents.jsx";
import Collections from "@/pages/Collections.jsx";
import KnowledgeGraph from "@/pages/KnowledgeGraph.jsx";
import Analytics from "@/pages/Analytics.jsx";
import Workflows from "@/pages/Workflows.jsx";
import Integrations from "@/pages/Integrations.jsx";
import UsersRoles from "@/pages/UsersRoles.jsx";
import Permissions from "@/pages/Permissions.jsx";
import AuditLogs from "@/pages/AuditLogs.jsx";
import DataGovernance from "@/pages/DataGovernance.jsx";
import ModelManagement from "@/pages/ModelManagement.jsx";
import DataConnectors from "@/pages/DataConnectors.jsx";
import SystemSettings from "@/pages/SystemSettings.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Home />} />
        <Route path="/ask" element={<AskKnowledge />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/collections" element={<Collections />} />
        <Route path="/graph" element={<KnowledgeGraph />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/workflows" element={<Workflows />} />
        <Route path="/integrations" element={<Integrations />} />
        <Route path="/users" element={<UsersRoles />} />
        <Route path="/permissions" element={<Permissions />} />
        <Route path="/audit" element={<AuditLogs />} />
        <Route path="/governance" element={<DataGovernance />} />
        <Route path="/models" element={<ModelManagement />} />
        <Route path="/connectors" element={<DataConnectors />} />
        <Route path="/settings" element={<SystemSettings />} />
        <Route path="*" element={<Navigate to="/ask" replace />} />
      </Route>
    </Routes>
  );
}
