# IssueSense

IssueSense is an evidence-grounded engineering intelligence platform. It answers engineering questions across code, documentation, delivery systems, and decision records while preserving source locations, authorization context, uncertainty, and verifiable citations.


## Core rule

Current, authorized source evidence is truth for an answer. Memory is context, never an automatic override. Claims must map to normalized evidence and citations must be verified before output.

## Local setup

Copy `.env.example` to `.env`, configure only the credentials needed for the active phase, create a Python 3.12 environment, then install the relevant optional dependency group. No external connector is implemented by this scaffold.

