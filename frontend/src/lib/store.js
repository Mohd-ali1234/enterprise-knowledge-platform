import { create } from "zustand";

export const useAppStore = create((set) => ({
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  topK: 8,
  setTopK: (n) => set({ topK: n }),
  reranking: true,
  toggleReranking: () => set((s) => ({ reranking: !s.reranking })),
  queryRewriting: true,
  toggleQueryRewriting: () =>
    set((s) => ({ queryRewriting: !s.queryRewriting })),
  hybridSearch: true,
  toggleHybrid: () => set((s) => ({ hybridSearch: !s.hybridSearch })),
  semanticSearch: true,
  toggleSemantic: () => set((s) => ({ semanticSearch: !s.semanticSearch })),
}));
