# AIWF Studio Pro frontend

This directory contains the React, TypeScript, and Vite frontend served by the AIWF Studio Pro FastAPI app.

## Local development

Install and run the frontend from this directory:

```powershell
npm ci
npm run dev
```

The Vite server listens on loopback and proxies `/api/pro` to `http://127.0.0.1:7861` by default. Set `AIWF_PRO_API_TARGET` to another loopback URL when the local backend uses a different port.

## API origin policy

The production build uses same-origin `/api/pro` requests. `VITE_AIWF_API_BASE` is only for a frontend and backend running on separate loopback ports during local development. The build rejects non-loopback values.

Remote and mobile API clients use the backend pairing flow and must send `X-AIWF-Token`. The React shell does not collect or store that token, so `VITE_AIWF_API_BASE` must not point at a LAN or internet host.

## Validation

```powershell
npm run test:api-base
npm run lint
npm run build
```
