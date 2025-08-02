# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the tokamak-rl-control-suite project.

## What are ADRs?

Architecture Decision Records (ADRs) are short documents that capture important architectural decisions made during the project, along with their context and consequences.

## ADR Format

Each ADR follows this template:

```markdown
# ADR-XXXX: [Decision Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing or have agreed to implement?

## Consequences
What becomes easier or more difficult to do because of this change?

## Alternatives Considered
What other options were considered and why were they rejected?
```

## Current ADRs

| Number | Title | Status | Date |
|--------|-------|--------|------|
| [0001](./adr-0001-gymnasium-environment-interface.md) | Use Gymnasium Environment Interface | Accepted | 2025-01-XX |
| [0002](./adr-0002-safety-first-architecture.md) | Safety-First Architecture Design | Accepted | 2025-01-XX |
| [0003](./adr-0003-modular-physics-engine.md) | Modular Physics Engine Design | Accepted | 2025-01-XX |

## Creating New ADRs

1. Copy the template from `adr-template.md`
2. Number the ADR sequentially (XXXX format)
3. Use kebab-case for the filename
4. Update this README with the new ADR entry
5. Commit the ADR with your implementation

## Guidelines

- ADRs should be concise (1-2 pages maximum)
- Focus on the decision, not implementation details
- Include the business/technical context
- Document alternatives that were considered
- ADRs are immutable once accepted (create new ADRs to supersede)