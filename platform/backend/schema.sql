-- PostgreSQL + PostGIS production schema.
-- Local development uses the SQLite fallback in app/services/state.py.
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS flights (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  origin GEOGRAPHY(POINTZ, 4326),
  status TEXT NOT NULL DEFAULT 'uploaded'
);

CREATE TABLE IF NOT EXISTS frames (
  id BIGSERIAL PRIMARY KEY,
  flight_id UUID NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
  frame_id TEXT NOT NULL,
  timestamp_seconds DOUBLE PRECISION NOT NULL,
  image_path TEXT NOT NULL,
  mask_path TEXT,
  blur_score DOUBLE PRECISION,
  gps GEOGRAPHY(POINTZ, 4326),
  UNIQUE (flight_id, frame_id)
);

CREATE TABLE IF NOT EXISTS poses (
  frame_id BIGINT PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
  rotation DOUBLE PRECISION[] NOT NULL,
  translation DOUBLE PRECISION[] NOT NULL,
  confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE TABLE IF NOT EXISTS model_assets (
  id BIGSERIAL PRIMARY KEY,
  flight_id UUID NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
  asset_type TEXT NOT NULL,
  asset_path TEXT NOT NULL,
  crs TEXT NOT NULL DEFAULT 'EPSG:4326',
  bbox GEOMETRY(POLYGONZ, 4326),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (flight_id, asset_type)
);

CREATE TABLE IF NOT EXISTS measurements (
  id BIGSERIAL PRIMARY KEY,
  flight_id UUID NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
  camera_position GEOGRAPHY(POINTZ, 4326),
  target_position GEOGRAPHY(POINTZ, 4326),
  distance_m DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS frames_flight_idx ON frames(flight_id);
CREATE INDEX IF NOT EXISTS model_assets_flight_idx ON model_assets(flight_id);
