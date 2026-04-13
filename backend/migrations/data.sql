-- VLA-Shield database schema for MySQL 8.0+
-- Aligned with: backend/vlashield/schemas.py, runtime/vlashield-core/src/*.rs

-- ---------------------------------------------------------------------------
-- robots: registered robot instances
-- Maps to: RuntimeConfig.robot_id (Rust), API path param robot_id (Python)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS robots (
    id              CHAR(36)        NOT NULL PRIMARY KEY,
    name            VARCHAR(255)    NOT NULL,
    ros_namespace   VARCHAR(255)    NOT NULL DEFAULT '',
    model_id        VARCHAR(128)    NOT NULL DEFAULT '' COMMENT 'VLA model running on this robot',
    run_mode        ENUM('production','physics_only','monitor','disabled')
                                    NOT NULL DEFAULT 'production',
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------------
-- ontology_nodes: safety ontology definitions
-- Maps to: OntologyNode (schemas.py), OntologyNode (ontology.rs)
-- Source of truth: dataset/ontology/physical.json, semantic.json
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ontology_nodes (
    id              VARCHAR(64)     NOT NULL PRIMARY KEY COMMENT 'e.g. PHY.COLLISION, SEM.FRAGILE',
    severity        ENUM('info','low','medium','high','critical')
                                    NOT NULL,
    hard_block      BOOLEAN         NOT NULL DEFAULT FALSE,
    title           VARCHAR(255)    NOT NULL,
    description     TEXT            NOT NULL,
    parents         JSON            DEFAULT NULL COMMENT 'Array of parent ontology IDs',
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------------
-- actions_log: every action evaluated by the safety runtime
-- Maps to: ActionVector + ArbiterDecision + LatencyBreakdown (schemas.py / arbiter.rs)
-- Written by: vlashield-io MySqlStore.insert_action_log (Rust)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS actions_log (
    id              BIGINT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
    robot_id        CHAR(36)        NOT NULL,
    sequence_id     BIGINT          NOT NULL COMMENT 'Monotonic per-robot counter from ActionVector',
    t_ns            BIGINT          NOT NULL COMMENT 'Action timestamp in nanoseconds',
    decision        ENUM('PASS','BLOCK') NOT NULL,
    risk_score      FLOAT           NOT NULL DEFAULT 0.0 COMMENT 'Composite risk 0.0~1.0',
    model_id        VARCHAR(128)    NOT NULL DEFAULT '' COMMENT 'VLA model that produced this action',
    action_dim      INT             NOT NULL DEFAULT 0 COMMENT 'Dimension of action vector',
    action_hash     VARCHAR(128)    DEFAULT NULL COMMENT 'sha256:hex for dedup/integrity',
    run_mode        ENUM('production','physics_only','monitor','disabled')
                                    NOT NULL DEFAULT 'production',
    latency_total_ms FLOAT          NOT NULL DEFAULT 0.0,
    latency_detail  JSON            DEFAULT NULL COMMENT 'Full LatencyBreakdown: ingest/physics/collision/semantic/total',
    meta            JSON            DEFAULT NULL COMMENT 'Model version, scene revision, etc.',
    created_at      TIMESTAMP(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

    INDEX idx_al_robot_ts       (robot_id, t_ns DESC),
    INDEX idx_al_robot_seq      (robot_id, sequence_id DESC),
    INDEX idx_al_decision       (decision),
    INDEX idx_al_action_hash    (action_hash),
    CONSTRAINT fk_al_robot FOREIGN KEY (robot_id) REFERENCES robots(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------------
-- safety_events: structured safety events (primarily blocks)
-- Maps to: SafetyEvent + ArbiterReason (schemas.py / arbiter.rs)
-- Written by: vlashield-io MySqlStore.insert_safety_event (Rust), queried by API (Python)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS safety_events (
    id              CHAR(36)        NOT NULL PRIMARY KEY COMMENT 'UUID event_id',
    robot_id        CHAR(36)        NOT NULL,
    sequence_id     BIGINT          NOT NULL,
    ts_ns           BIGINT          NOT NULL COMMENT 'Event timestamp in nanoseconds',
    decision        ENUM('PASS','BLOCK') NOT NULL,
    risk_score      FLOAT           NOT NULL DEFAULT 0.0,
    action_hash     VARCHAR(128)    DEFAULT NULL,
    run_mode        ENUM('production','physics_only','monitor','disabled')
                                    NOT NULL DEFAULT 'production',
    ontology_ids    JSON            NOT NULL COMMENT 'Array of triggered OntologyId strings',
    reasons         JSON            NOT NULL COMMENT 'Array of ArbiterReason {ontology_id, detail, score}',
    latency_total_ms FLOAT          NOT NULL DEFAULT 0.0,
    payload         JSON            NOT NULL COMMENT 'Full serialized SafetyEvent for archival',
    created_at      TIMESTAMP(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

    INDEX idx_se_robot_ts       (robot_id, ts_ns DESC),
    INDEX idx_se_robot_seq      (robot_id, sequence_id DESC),
    INDEX idx_se_decision       (decision),
    CONSTRAINT fk_se_robot FOREIGN KEY (robot_id) REFERENCES robots(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------------
-- red_team_entries: imported red-team benchmark dataset
-- Maps to: RedTeamEntry (schemas.py)
-- Written by: vlashield.data.download (Python)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS red_team_entries (
    id              VARCHAR(32)     NOT NULL PRIMARY KEY COMMENT 'e.g. rt-000001, sr-a3f2c1',
    split           ENUM('train','val','test') NOT NULL DEFAULT 'train',
    locale          VARCHAR(16)     NOT NULL DEFAULT 'en',
    instruction     TEXT            NOT NULL,
    task_context    VARCHAR(64)     NOT NULL DEFAULT '',
    risk_tags       JSON            NOT NULL COMMENT 'Array of ontology ID strings',
    expected_outcome ENUM('refuse_or_safe_alternative','proceed_with_caution','proceed_safe')
                                    NOT NULL DEFAULT 'refuse_or_safe_alternative',
    action_gold     JSON            DEFAULT NULL COMMENT 'Optional gold-standard safe action vector',
    annotator       VARCHAR(64)     NOT NULL DEFAULT '',
    source          VARCHAR(64)     NOT NULL DEFAULT 'manual' COMMENT 'BeaverTails, PKU-SafeRLHF, manual',
    version         VARCHAR(16)     NOT NULL DEFAULT 'v0.1',
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_rte_split         (split),
    INDEX idx_rte_source        (source),
    INDEX idx_rte_context       (task_context)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ---------------------------------------------------------------------------
-- Seed data for development
-- ---------------------------------------------------------------------------
INSERT IGNORE INTO robots (id, name, ros_namespace, model_id, run_mode)
VALUES ('00000000-0000-0000-0000-000000000001', 'ur5e-lab-01', '/ur5e', 'openvla-7b', 'production');

INSERT IGNORE INTO ontology_nodes (id, severity, hard_block, title, description, parents) VALUES
('PHY.COLLISION',       'high',     TRUE,  'Collision',              'Imminent link-object or object-human impact above force/velocity threshold.', '[]'),
('PHY.TIPOVER',         'critical', TRUE,  'Tip-over',              'Support polygon / ZMP-style heuristic violation indicating platform instability.', '[]'),
('PHY.OVERLOAD',        'high',     TRUE,  'Overload',              'Torque, current, or gripper force exceeds the nominal envelope of the actuator.', '[]'),
('PHY.VELOCITY_LIMIT',  'medium',   FALSE, 'Velocity limit',        'Proposed joint or end-effector velocity exceeds the configured safety cap.', '[]'),
('SEM.FRAGILE',         'medium',   FALSE, 'Fragile object',        'Action likely to damage fragile entities such as glass, ceramics, or electronics.', '[]'),
('SEM.HEAT_SOURCE',     'high',     TRUE,  'Heat source proximity', 'End-effector proximity or contact with dangerous heat sources.', '[]'),
('SEM.FORBIDDEN_REGION','high',     TRUE,  'Forbidden region',      'Workspace or semantic zone violation.', '[]'),
('SEM.LIQUID_ELECTRICAL','critical', TRUE,  'Liquid near electrical','Hazardous combination: liquid near electrical outlets or devices.', '[]'),
('SEM.HUMAN_PROXIMITY', 'high',     FALSE, 'Human proximity',       'Fast or forceful motion while a human is within the safety perimeter.', '[]'),
('SEM.SHARP_OBJECT',    'medium',   FALSE, 'Sharp object handling', 'Manipulating knives, scissors, or other sharp objects toward a person or fragile surface.', '[]');
