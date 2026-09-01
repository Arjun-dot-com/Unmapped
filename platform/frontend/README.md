# Phase 5 viewer

Install with `npm install`, start with `npm run dev`, and point the frontend at the FastAPI server. The viewer loads the georeferenced model, includes click-to-measure and confidence-layer controls, and uses OpenStreetMap imagery by default. Copy `.env.example` to `.env.local` and add `NEXT_PUBLIC_CESIUM_ION_TOKEN` to enable Cesium Ion world terrain.
