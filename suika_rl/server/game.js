/**
 * Suika Game Logic (Headless)
 * Ported from index.js for Node.js server
 */

const Matter = require('matter-js');

const GameStates = {
  MENU: 0,
  READY: 1,
  DROP: 2,
  LOSE: 3,
};

class SuikaGame {
  constructor(width = 640, height = 960, seed = null) {
    this.width = width;
    this.height = height;
    this.wallPad = 64;
    this.loseHeight = 84;

    // Random number generator (seeded)
    this.seed = seed || Date.now();
    this.rng = this.mulberry32(this.seed);

    // Game state
    this.stateIndex = GameStates.MENU;
    this.score = 0;
    this.fruitsMerged = Array(11).fill(0);
    this.currentFruitSize = 0;
    this.nextFruitSize = 0;
    this.fastMode = true;

    // Fruit sizes and scores
    this.fruitSizes = [
      { radius: 24,  scoreValue: 1  },
      { radius: 32,  scoreValue: 3  },
      { radius: 40,  scoreValue: 6  },
      { radius: 56,  scoreValue: 10 },
      { radius: 64,  scoreValue: 15 },
      { radius: 72,  scoreValue: 21 },
      { radius: 84,  scoreValue: 28 },
      { radius: 96,  scoreValue: 36 },
      { radius: 128, scoreValue: 45 },
      { radius: 160, scoreValue: 55 },
      { radius: 192, scoreValue: 66 },
    ];

    this.friction = {
      friction: 0.006,
      frictionStatic: 0.006,
      frictionAir: 0,
      restitution: 0.1
    };

    // Canvas rendering (simplified - no actual rendering)
    // We'll generate synthetic observations based on game state
    this.canvas = null;
    this.ctx = null;

    // Matter.js engine
    const { Engine, Composite, Bodies, Events } = Matter;
    this.engine = Engine.create();
    this.engine.gravity.y = 1;

    // Create walls
    this.gameStatics = this._createWalls();
    Composite.add(this.engine.world, this.gameStatics);

    // Collision handler for fruit merging
    Events.on(this.engine, 'collisionStart', (event) => {
      this._handleCollisions(event);
    });

    // Preview ball
    this.previewBall = null;

    this.startGame();
  }

  // Seeded random number generator
  mulberry32(a) {
    return () => {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  _createWalls() {
    const { Bodies } = Matter;
    const wallThickness = 50;
    const wallHeight = this.height - this.loseHeight;

    return [
      // Left wall
      Bodies.rectangle(
        this.wallPad / 2,
        wallHeight / 2 + this.loseHeight,
        this.wallPad,
        wallHeight,
        { isStatic: true, label: 'wall-left' }
      ),
      // Right wall
      Bodies.rectangle(
        this.width - this.wallPad / 2,
        wallHeight / 2 + this.loseHeight,
        this.wallPad,
        wallHeight,
        { isStatic: true, label: 'wall-right' }
      ),
      // Bottom wall
      Bodies.rectangle(
        this.width / 2,
        this.height - wallThickness / 2,
        this.width,
        wallThickness,
        { isStatic: true, label: 'wall-bottom' }
      ),
    ];
  }

  startGame() {
    this.stateIndex = GameStates.READY;
    this.score = 0;
    this.fruitsMerged = Array(11).fill(0);

    // Clear all fruits
    const { Composite } = Matter;
    const bodies = Composite.allBodies(this.engine.world);
    for (let body of bodies) {
      if (!body.isStatic) {
        Composite.remove(this.engine.world, body);
      }
    }

    // Set initial fruits
    this.currentFruitSize = Math.floor(this.rng() * 5);
    this.nextFruitSize = Math.floor(this.rng() * 5);

    // Create preview ball
    this.previewBall = this.generateFruitBody(
      this.width / 2,
      0,
      this.currentFruitSize,
      { isStatic: true }
    );
    Composite.add(this.engine.world, this.previewBall);
  }

  reset() {
    const { Composite } = Matter;

    // Clear all non-static bodies
    const bodies = Composite.allBodies(this.engine.world);
    for (let body of bodies) {
      if (!body.isStatic) {
        Composite.remove(this.engine.world, body);
      }
    }

    this.startGame();
    this._render();

    return this.getObservation();
  }

  generateFruitBody(x, y, sizeIndex, options = {}) {
    const { Bodies } = Matter;
    const fruitSize = this.fruitSizes[sizeIndex];

    return Bodies.circle(x, y, fruitSize.radius, {
      ...this.friction,
      label: `fruit-${sizeIndex}`,
      sizeIndex: sizeIndex,
      ...options
    });
  }

  addFruit(x) {
    if (this.stateIndex !== GameStates.READY) {
      return false;
    }

    const { Composite } = Matter;

    // Change state
    this.stateIndex = GameStates.DROP;

    // Add fruit
    const latestFruit = this.generateFruitBody(x, this.loseHeight + 10, this.currentFruitSize);
    Composite.add(this.engine.world, latestFruit);

    // Update current and next fruit
    this.currentFruitSize = this.nextFruitSize;
    this.nextFruitSize = Math.floor(this.rng() * 5);

    // Remove and recreate preview ball
    if (this.previewBall) {
      Composite.remove(this.engine.world, this.previewBall);
    }
    this.previewBall = this.generateFruitBody(
      x,
      0,
      this.currentFruitSize,
      { isStatic: true, collisionFilter: { mask: 0x0040 } }
    );

    // Transition back to READY after a short delay
    setTimeout(() => {
      if (this.stateIndex === GameStates.DROP) {
        Composite.add(this.engine.world, this.previewBall);
        this.stateIndex = GameStates.READY;
      }
    }, 50);

    return true;
  }

  _handleCollisions(event) {
    const { Composite } = Matter;

    for (let pair of event.pairs) {
      const { bodyA, bodyB } = pair;

      // Check if both bodies are fruits
      if (bodyA.label.startsWith('fruit-') && bodyB.label.startsWith('fruit-')) {
        const sizeA = bodyA.sizeIndex;
        const sizeB = bodyB.sizeIndex;

        // Same size fruits merge
        if (sizeA === sizeB && sizeA < this.fruitSizes.length - 1) {
          const newSizeIndex = sizeA + 1;

          // Calculate merge position (midpoint)
          const x = (bodyA.position.x + bodyB.position.x) / 2;
          const y = (bodyA.position.y + bodyB.position.y) / 2;

          // Remove old fruits
          Composite.remove(this.engine.world, bodyA);
          Composite.remove(this.engine.world, bodyB);

          // Add new fruit
          const newFruit = this.generateFruitBody(x, y, newSizeIndex);
          Composite.add(this.engine.world, newFruit);

          // Update score
          this.fruitsMerged[newSizeIndex]++;
          this.calculateScore();
        }
      }

      // Check lose condition (fruit touches top line)
      if (bodyA.label.startsWith('fruit-') || bodyB.label.startsWith('fruit-')) {
        const fruit = bodyA.label.startsWith('fruit-') ? bodyA : bodyB;

        if (fruit.position.y - this.fruitSizes[fruit.sizeIndex].radius < this.loseHeight) {
          this.stateIndex = GameStates.LOSE;
        }
      }
    }
  }

  calculateScore() {
    this.score = this.fruitsMerged.reduce((total, count, sizeIndex) => {
      return total + this.fruitSizes[sizeIndex].scoreValue * count;
    }, 0);
  }

  isStable(threshold = 0.01) {
    const { Composite } = Matter;
    const bodies = Composite.allBodies(this.engine.world);

    for (let body of bodies) {
      if (body.isStatic) continue;

      const speed = Math.sqrt(
        body.velocity.x * body.velocity.x +
        body.velocity.y * body.velocity.y
      );

      const angularSpeed = Math.abs(body.angularVelocity || 0);

      if (speed > threshold || angularSpeed > threshold) {
        return false;
      }
    }

    return true;
  }

  fastForwardUntilStable(maxSteps = 300, deltaMs = 16.67, threshold = 0.01) {
    const { Engine } = Matter;
    let steps = 0;
    let stableCount = 0;
    const requiredStableSteps = 12;

    while (steps < maxSteps) {
      Engine.update(this.engine, deltaMs);
      steps++;

      if (this.isStable(threshold)) {
        stableCount++;
        if (stableCount >= requiredStableSteps) {
          return { success: true, steps, time: steps * deltaMs };
        }
      } else {
        stableCount = 0;
      }
    }

    return { success: false, steps, time: steps * deltaMs };
  }

  _render() {
    // Simplified rendering: Create a synthetic image based on game state
    // Image size: 260x384 (maintains 2:3 aspect ratio of game area)
    // This preserves spatial information better than 128x128

    const { Composite } = Matter;
    const width = 260;   // ~half of 520px play area width
    const height = 384;  // maintains 2:3 ratio (260 * 1.477 ≈ 384)
    const imageSize = width * height * 4;  // RGBA
    const imageData = new Uint8Array(imageSize);

    // Fill with background color (RGB: 255, 213, 157)
    for (let i = 0; i < imageSize; i += 4) {
      imageData[i] = 255;     // R
      imageData[i + 1] = 213; // G
      imageData[i + 2] = 157; // B
      imageData[i + 3] = 255; // A
    }

    // Draw simplified fruit representations
    const bodies = Composite.allBodies(this.engine.world);
    for (let body of bodies) {
      if (body.label.startsWith('fruit-')) {
        // Map from world coordinates (640x960) to image coordinates (128x128)
        const x = Math.floor(body.position.x * width / this.width);
        const y = Math.floor(body.position.y * height / this.height);
        const radius = Math.floor(this.fruitSizes[body.sizeIndex].radius * width / this.width);

        // Draw a simple filled circle
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy <= radius * radius) {
              const px = x + dx;
              const py = y + dy;

              if (px >= 0 && px < width && py >= 0 && py < height) {
                const idx = (py * width + px) * 4;

                // Color based on fruit size
                const hue = (body.sizeIndex / this.fruitSizes.length);
                imageData[idx] = Math.floor(255 * (1 - hue));     // R
                imageData[idx + 1] = Math.floor(255 * hue);       // G
                imageData[idx + 2] = Math.floor(150);             // B
                imageData[idx + 3] = 255;                         // A
              }
            }
          }
        }
      }
    }

    return imageData;
  }

  getObservation() {
    // Render current state (returns RGBA array)
    const imageData = this._render();

    return {
      image: Array.from(imageData),  // RGBA array [128x128x4]
      score: this.score,
      done: this.stateIndex === GameStates.LOSE,
      stateIndex: this.stateIndex
    };
  }

  step(action) {
    // action is x position (0-640)
    const success = this.addFruit(action);

    if (!success) {
      // Game not ready, return current state
      return this.getObservation();
    }

    // Fast forward until stable
    this.fastForwardUntilStable();

    // Return observation
    return this.getObservation();
  }
}

module.exports = { SuikaGame, GameStates };
