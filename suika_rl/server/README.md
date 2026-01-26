# Suika Game Server

Headless Suika Game server for RL training using Node.js.

## Installation

```bash
npm install
```

## Usage

```bash
# Start server
npm start

# Development mode (auto-reload)
npm run dev
```

Server runs on port 8924 by default.

## API Endpoints

### POST /api/reset
Reset the game environment.

**Request:**
```json
{
  "game_id": "env_0",
  "seed": 12345
}
```

**Response:**
```json
{
  "success": true,
  "observation": {
    "image": [...],  // RGBA array [128x128x4]
    "score": 0,
    "done": false,
    "stateIndex": 1
  },
  "game_id": "env_0"
}
```

### POST /api/step
Execute an action in the environment.

**Request:**
```json
{
  "game_id": "env_0",
  "action": 320,
  "auto_reset": true
}
```

**Response:**
```json
{
  "success": true,
  "observation": {
    "image": [...],
    "score": 10,
    "done": false,
    "stateIndex": 1
  }
}
```

### GET /health
Check server health.

**Response:**
```json
{
  "status": "ok",
  "activeGames": 4,
  "timestamp": 1706280000000
}
```

## Performance

- ~50-70ms per step (vs ~1000ms with Selenium)
- Supports multiple parallel game instances
- No memory leaks from browser processes
