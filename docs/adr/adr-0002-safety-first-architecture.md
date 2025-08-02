# ADR-0002: Safety-First Architecture Design

## Status
Accepted

## Context
Tokamak plasma control systems are safety-critical. Plasma disruptions can cause:
- Multi-million dollar damage to reactor components
- Personnel safety risks from electromagnetic forces
- Loss of experimental data and operational time
- Potential regulatory violations and shutdown

Traditional RL approaches prioritize performance optimization, but fusion control requires absolute safety guarantees. Any control system must prevent dangerous configurations even during exploration or model failures.

## Decision
We implement a safety-first architecture with multiple layers of protection:

1. **Physics-Based Constraints**: Hard limits on safety factor (q > 1.5), density (< Greenwald limit), beta
2. **Safety Shield**: Real-time action filtering before execution
3. **Disruption Predictor**: ML-based early warning system (LSTM on diagnostic signals)
4. **Emergency Fallback**: Classical PID control as last resort
5. **Hardware Interlocks**: Direct integration with tokamak safety systems

Architecture layers (innermost to outermost):
```
RL Agent → Safety Shield → Emergency Controller → Hardware Interlocks → Actuators
```

## Consequences

**Positive:**
- Guaranteed safe operation even with untrained/failing RL agents
- Regulatory compliance and operator confidence
- Graceful degradation during system failures
- Allows aggressive RL exploration within safe bounds
- Audit trail for post-incident analysis

**Negative:**
- Reduced RL agent action space and autonomy
- Additional computational overhead (safety checks)
- More complex system architecture and testing
- Potential performance limitations from conservative safety margins

**Neutral:**
- Need for extensive safety validation and testing
- Integration complexity with existing tokamak systems
- Regular safety system calibration and maintenance

## Alternatives Considered

1. **Unconstrained RL**: Rejected due to unacceptable disruption risk
2. **Soft Constraints via Rewards**: Rejected as insufficient for hard safety requirements
3. **Classical Control Only**: Rejected due to performance limitations
4. **Model-Predictive Safety**: Considered but computationally expensive for real-time operation
5. **Learned Safety Policies**: Rejected due to lack of safety guarantees during training