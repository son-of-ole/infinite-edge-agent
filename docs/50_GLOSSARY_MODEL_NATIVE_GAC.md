# 50 — Glossary: Model-Native GAC

## Active context

The memory and instructions available to the model during a specific inference call.

## Centroid

An artificial vector representing the average of a cluster. Useful for tight clusters but risky for identity-critical facts.

## Cell

A persistent agent/runtime entity with memory, capabilities, tools, and state.

## Consolidation

The process of reducing memory surface area while preserving useful retrieval and exact identity where required.

## Context pack

Structured set of instructions, memories, sources, and task state sent to the model or SSA runtime.

## Effective dimension

A local estimate of how complex or spread out a cluster is in embedding space.

## GAC

Geometry-Aware Memory Consolidation. The subsystem that uses cluster geometry and policy to decide what memory can be compressed.

## Identity collapse

Failure mode where a compressed memory preserves broad topic but loses the exact fact.

## Identity pin

A raw memory marked as exact and protected from compression.

## KVSwap

Runtime layer that tiers key-value cache across hot, warm, and cold memory.

## LanceDB

Embedded vector/columnar database used as the primary memory engine.

## Lineage

Mapping from derived memory to exact raw memory sources.

## Medoid

A real cluster member used as representative. Safer than centroid when exactness matters.

## Memory action head

Model-native component that predicts memory operations such as write, pin, split, or fetch raw lineage.

## Memory representative

Derived vector or text object used to retrieve a group of memories more efficiently.

## Model-native memory

A model/runtime architecture where memory operations are first-class behavior, not just external RAG.

## Raw memory

Exact immutable stored memory record.

## Residual representative

A vector representing remaining variation after choosing a primary medoid.

## Retrieval audit

Test record showing whether retrieval found the expected exact memory.

## Sleep cycle

Background process that consolidates memory and prepares wake context when a cell becomes idle or hibernates.

## Source grounding

Requirement that claims are backed by raw memory, file, or source lineage.

## SSA

Subquadratic Selective Attention. Sparse attention runtime that selects relevant blocks instead of full dense attention.

## TSP

Folding Tensor and Sequence Parallelism. Strategy for managing memory pressure for long-context execution.

## Wake context

Derived context generated during sleep cycle to help a cell resume work quickly.
