# ADR-0001: Use Gymnasium Environment Interface

## Status
Accepted

## Context
The project needs a standardized interface for reinforcement learning environments. The tokamak plasma control system requires integration with various RL algorithms and frameworks. Two main options exist: OpenAI Gym (legacy) and Gymnasium (maintained fork). Additionally, we could create a custom interface.

The RL community has largely migrated to Gymnasium due to better maintenance, API improvements, and active development. Most modern RL libraries (Stable-Baselines3, Ray RLlib) support Gymnasium natively.

## Decision
We will use the Gymnasium environment interface as the standard API for our tokamak plasma control environment.

Key aspects:
- Implement `gymnasium.Env` base class
- Use Box spaces for continuous control actions and observations
- Follow Gymnasium's step() API: `observation, reward, terminated, truncated, info`
- Implement proper environment registration via `gymnasium.register()`
- Support vectorized environments for parallel training

## Consequences

**Positive:**
- Seamless integration with existing RL frameworks and algorithms
- Access to extensive ecosystem of RL tools and utilities
- Standardized API familiar to RL researchers
- Built-in support for environment vectorization and parallel training
- Better documentation and community support

**Negative:**
- Dependency on external package (though very stable)
- Must conform to Gymnasium's API constraints
- Limited flexibility in observation/action space design

**Neutral:**
- Standard environment lifecycle management (reset, step, close)
- Need to implement proper seeding for reproducibility

## Alternatives Considered

1. **OpenAI Gym (Legacy)**: Rejected due to deprecated status and lack of maintenance
2. **Custom Interface**: Rejected due to increased development overhead and lack of ecosystem compatibility
3. **PettingZoo**: Considered for multi-agent scenarios but not needed for current single-agent control
4. **DeepMind Lab**: Too heavyweight and game-focused for physics simulation