# Helios Sync Platform Specification

## Overview

Helios Sync is the telemetry synchronization platform built by Helios Labs.
It collects high-frequency sensor streams from customer sites and reconciles
them into a single ordered timeline. Typical deployments range from wind
farms to container terminals and port cranes. Daniel Okafor is the executive
owner of Helios Sync and reviews the roadmap with the platform leads monthly.

## Architecture

Helios Sync is composed of three subsystems: Pulse Engine, Beacon API, and
Sentinel Dashboard. Pulse Engine is the stream processor that orders,
deduplicates, and compacts telemetry. Beacon API is the public ingestion
surface used by customer agents. Sentinel Dashboard is the operations console
used by on-call engineers.

## Deployment

Cascade Campus hosts the primary Pulse Engine cluster, which runs on HX-9
compute nodes with local NVMe storage. A warm standby cluster runs in a
colocation facility near Lisbon Hub and is promoted automatically when the
primary site fails two consecutive health checks.

## Beacon API

Beacon API enforces a rate limit of 1200 requests per minute per customer
agent. Authentication uses signed agent tokens, and tokens are rotated every
12 hours. Malformed batches are rejected with a structured error envelope so
agents can repair and resubmit without operator intervention.

## Service Levels and On-Call

Helios Sync carries a 99.9 percent monthly uptime commitment for ingestion.
When an availability alarm fires, Sentinel Dashboard pages the on-call
engineer within five minutes. A post-incident review is mandatory for every
customer-visible disruption.

## Data Governance

Telemetry and request data stored by Helios Sync are governed by the Meridian
Policy. Engineers adding a new data store must register it with the
compliance office before it receives production traffic.

## Roadmap

Planned work includes schema-aware compaction in Pulse Engine, mutual-TLS
support in Beacon API, and customer-facing latency analytics in Sentinel
Dashboard. Dates are reviewed quarterly and tracked in the platform plan.
