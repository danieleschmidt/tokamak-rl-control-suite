# Monitoring & Observability

This directory contains monitoring and observability configuration and documentation for the tokamak-rl-control-suite.

## Contents

- [`health-checks.md`](health-checks.md) - Health check endpoint configurations
- [`logging.md`](logging.md) - Structured logging configuration
- [`metrics.md`](metrics.md) - Prometheus metrics configuration
- [`alerting.md`](alerting.md) - Alerting configuration templates
- [`observability-best-practices.md`](observability-best-practices.md) - Best practices guide

## Overview

The monitoring setup is designed for production-ready observability with:

- **Health Checks**: HTTP endpoints for liveness/readiness probes
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Metrics**: Prometheus-compatible metrics for system and business KPIs
- **Alerting**: Template configurations for critical system alerts
- **Distributed Tracing**: OpenTelemetry integration for request tracing

## Quick Start

1. Review the [health-checks.md](health-checks.md) for endpoint implementation
2. Configure logging using templates in [logging.md](logging.md)
3. Set up metrics collection per [metrics.md](metrics.md)
4. Implement alerting based on [alerting.md](alerting.md)

## Production Readiness

This monitoring setup is designed to meet enterprise observability requirements:

- SLI/SLO tracking for plasma control performance
- MTTR optimization through effective alerting
- Compliance with reliability engineering practices
- Integration with external monitoring systems