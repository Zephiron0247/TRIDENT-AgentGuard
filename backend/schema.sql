CREATE TABLE sessions (
    session_id      VARCHAR(50) PRIMARY KEY,
    agent_id        VARCHAR(100),
    stated_goal     VARCHAR(500),
    status          VARCHAR(20),   -- active / blocked / killed
    started_at      TIMESTAMP,
    ended_at        TIMESTAMP
);

CREATE TABLE tool_calls (
    call_id         VARCHAR(50) PRIMARY KEY,
    session_id      VARCHAR(50),
    tool_name       VARCHAR(100),
    tool_args       VARCHAR(2000),
    risk_score      DECIMAL(4,3),
    decision        VARCHAR(20),   -- allow / block / killswitch
    entailment_flag BOOLEAN,
    called_at       TIMESTAMP
);

CREATE TABLE incidents (
    incident_id     VARCHAR(50) PRIMARY KEY,
    session_id      VARCHAR(50),
    trigger_call_id VARCHAR(50),
    final_risk_score DECIMAL(4,3),
    created_at      TIMESTAMP
);