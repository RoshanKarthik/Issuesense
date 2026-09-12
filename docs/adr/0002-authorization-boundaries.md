# ADR 0002: Authorization Boundaries and Access Control

## Status
Proposed

## Context
IssueSense interacts with sensitive organizational data (GitHub, Notion, etc.). It is not sufficient to rely on the global credentials of the service account; the system must ensure that the specific user requesting information has the right to see it.

## Decision
We will implement a strict authorization boundary at the connector level:
1. **Authorization Context**: Every request to the API must provide an `AuthorizationContext` containing the user's identity and permission scopes.
2. **Application-Level Checks**: Connectors must perform an access check using the `AuthorizationContext` *before* performing any retrieval operation.
3. **Scope-Based Access**: Access will be managed via scopes (e.g., `read:repo`, `read:wiki`).
4. **No Implicit Trust**: The system will never assume that because a connector has access to a resource, the user does.

## Consequences
- **Pros**: Prevents unauthorized data leakage and ensures compliance with organizational security policies.
- **Cons**: Requires a robust identity provider integration and adds latency to the retrieval path for permission verification.
