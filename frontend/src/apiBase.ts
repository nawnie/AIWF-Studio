import { resolveLoopbackApiBase } from './apiBasePolicy'

export const API_BASE = resolveLoopbackApiBase(import.meta.env.VITE_AIWF_API_BASE)
