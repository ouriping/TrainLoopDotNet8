# RFC-0001: TrainLoop Architecture

**Status:** Accepted
**Date:** 2026-03-25
**Author:** Jerrett Davis

---

## Overview

TrainLoop is a human-in-the-loop ML training platform designed to produce high-quality, auditable labeled datasets through structured human review, continuous reviewer calibration, and a meta-model layer that learns from reviewer behavior patterns.

---

## The 3-Loop Concept

### Loop 1: Data Labeling

The primary loop. Human reviewers annotate `DataItem` records within a `Dataset`. Every annotation requires a written `Rationale` — this is a first-class field, not an afterthought. The rationale requirement forces reviewers to articulate their reasoning, which:

- Surfaces disagreements earlier
- Provides training signal for the meta-model
- Creates an auditable record of human intent

Labels flow through a consensus resolution step before being finalized. Items without consensus are escalated for re-review or adjudication.

### Loop 2: Reviewer Calibration

Reviewers are not treated as uniform oracles. Each `Reviewer` carries a `QualityScore` that is continuously updated via:

- **Quiz injection:** Gold-standard `QuizItem` records with known labels are silently mixed into the review queue. Reviewer answers are scored against the ground truth.
- **Inter-annotator agreement:** Agreement rates between reviewers on the same items feed back into quality scoring.
- **Calibration events:** `ReviewerCalibrated` domain events record every score change with old and new values, creating a complete audit trail.

Reviewer quality scores are used as weights in consensus resolution.

### Loop 3: Meta-Model

A lightweight ML model (ML.NET / ONNX) that observes annotation patterns and learns to:

- Predict which items are likely to be contentious (routing to more experienced reviewers)
- Flag potentially mislabeled items for re-review
- Estimate reviewer fatigue from `TimeToLabel` drift
- Surface systematic labeling biases across the reviewer pool

The meta-model does not replace human judgment — it routes and prioritizes work to make human review more effective.

---

## Design Decisions

### Storage: PostgreSQL

PostgreSQL is the primary data store via EF Core. Rationale:

- Mature JSONB support for semi-structured annotation data
- Strong transaction semantics critical for consensus writes
- Excellent .NET ecosystem support (Npgsql)
- Full-text search for rationale mining without a separate search index at early scale

### Event-Sourced Audit Log

All significant state transitions emit `DomainEvent` records:

- `AnnotationSubmitted` — every label with full rationale
- `QuizAnswered` — calibration probe results
- `ConsensusResolved` — when a final label is committed and by what strategy
- `ReviewerCalibrated` — quality score deltas

Events are appended to an `OutboxEvents` table and relayed to a message bus (MassTransit + RabbitMQ or Azure Service Bus). This gives us:

- Complete audit trail for compliance
- Replay capability to recompute derived state
- Clean integration boundary between `TrainLoop.Worker` consumers and the core domain

### Consensus Strategy

Consensus is pluggable via a `IConsensusStrategy` abstraction. Initial implementations:

- **Majority vote** — simple threshold (>= 3 of 5 reviewers agree)
- **Weighted vote** — votes weighted by reviewer `QualityScore`
- **Quorum with confidence** — requires both agreement count and minimum average confidence

The `Strategy` field on `ConsensusResolved` records which algorithm produced the final label, enabling retrospective analysis.

### ML.NET / ONNX for Meta-Model

- ML.NET for training the meta-model within the .NET process (no Python runtime dependency)
- ONNX export for portability and serving via `Microsoft.ML.OnnxRuntime`
- Initial feature set: `TimeToLabel`, confidence scores, reviewer quality scores, label entropy across annotators
- Model retrained periodically by `TrainLoop.Worker` as new annotation data accumulates

### MCP Hooks for Agent Integration

TrainLoop exposes a Model Context Protocol (MCP) interface to allow AI agents to participate as reviewers or to drive bulk annotation workflows. Design principles:

- Agents must still provide `Rationale` — no special-casing
- Agent annotations are tracked separately, never silently mixed with human annotations in consensus without explicit configuration
- Agent `QualityScore` is calibrated the same way as humans — via quiz injection

The MCP surface lives in `TrainLoop.Api` behind a feature flag, initially disabled.

---

## Phase Breakdown

### Phase 1: Foundation (Current)

- Solution scaffold with `TrainLoop.Api`, `TrainLoop.Core`, `TrainLoop.Infrastructure`, `TrainLoop.Worker`, `TrainLoop.Web`
- Domain entities: `Dataset`, `DataItem`, `Annotation`, `Reviewer`, `QuizItem`
- Domain event contracts
- EF Core DbContext with PostgreSQL, migrations
- Basic CRUD REST endpoints
- CI pipeline

### Phase 2: Labeling Loop

- Full annotation submission flow with rationale validation
- Quiz item injection into review queues
- Consensus resolution (majority vote first)
- `ReviewerCalibrated` event emission and score updates
- Blazor review UI for human annotators (item queue, annotation form, confidence slider)

### Phase 3: Meta-Model

- Feature extraction pipeline in `TrainLoop.Worker`
- ML.NET training job triggered on dataset milestones
- Contentiousness prediction integrated into item routing
- ONNX model versioning and rollback

### Phase 4: Agent Integration

- MCP endpoint implementation
- Agent reviewer registration and quota controls
- Hybrid human+agent consensus strategies
- Dashboard metrics: human vs. agent agreement rates, cost per label

---

## Open Questions

1. **Label taxonomy management** — who defines valid labels per dataset? How are schema changes versioned without invalidating existing annotations?
2. **Adjudication workflow** — when consensus cannot be reached after N rounds, what escalation path exists?
3. **Data privacy** — how do we handle PII in `Content` fields, particularly for export and agent visibility?
4. **Reviewer onboarding** — what minimum quiz pass rate is required before a reviewer enters the production queue?
