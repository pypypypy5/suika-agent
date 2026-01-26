/**
 * Suika Game HTTP Server
 * Provides REST API for RL training
 */

const express = require('express');
const cors = require('cors');
const { SuikaGame } = require('./game');

const app = express();
const PORT = process.env.PORT || 8924;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Game instances storage (one per game_id)
const games = new Map();

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    activeGames: games.size,
    timestamp: Date.now()
  });
});

// Reset environment
app.post('/api/reset', (req, res) => {
  try {
    const { game_id = 'default', seed = null } = req.body;

    // Create or reset game
    const game = new SuikaGame(640, 960, seed);
    games.set(game_id, game);

    const obs = game.getObservation();

    res.json({
      success: true,
      observation: obs,
      game_id
    });
  } catch (error) {
    console.error('Error in /api/reset:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Step environment
app.post('/api/step', (req, res) => {
  try {
    const { game_id = 'default', action } = req.body;

    if (action === undefined || action === null) {
      return res.status(400).json({
        success: false,
        error: 'Action is required'
      });
    }

    // Get game instance
    let game = games.get(game_id);

    if (!game) {
      // Auto-create if not exists
      game = new SuikaGame(640, 960);
      games.set(game_id, game);
    }

    // Execute step
    const obs = game.step(action);

    // If game over, auto-reset (optional behavior)
    const autoReset = req.body.auto_reset !== false;
    if (obs.done && autoReset) {
      game.reset();
    }

    res.json({
      success: true,
      observation: obs
    });
  } catch (error) {
    console.error('Error in /api/step:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get current state
app.get('/api/state/:game_id', (req, res) => {
  try {
    const { game_id } = req.params;
    const game = games.get(game_id);

    if (!game) {
      return res.status(404).json({
        success: false,
        error: 'Game not found'
      });
    }

    const obs = game.getObservation();

    res.json({
      success: true,
      observation: obs
    });
  } catch (error) {
    console.error('Error in /api/state:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Delete game instance
app.delete('/api/game/:game_id', (req, res) => {
  try {
    const { game_id } = req.params;
    const deleted = games.delete(game_id);

    res.json({
      success: deleted,
      game_id,
      remainingGames: games.size
    });
  } catch (error) {
    console.error('Error in /api/game:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Suika Game Server running on port ${PORT}`);
  console.log(`API endpoints:`);
  console.log(`  POST /api/reset  - Reset environment`);
  console.log(`  POST /api/step   - Execute action`);
  console.log(`  GET  /api/state/:game_id - Get current state`);
  console.log(`  GET  /health     - Health check`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  games.clear();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\nShutting down server...');
  games.clear();
  process.exit(0);
});
