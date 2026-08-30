#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

const char * common_speculative_controller_mode_name(common_speculative_controller_mode mode);
common_speculative_controller_mode common_speculative_controller_mode_from_name(const std::string & name);
const char * common_speculative_controller_persistence_name(common_speculative_controller_persistence persistence);
common_speculative_controller_persistence common_speculative_controller_persistence_from_name(const std::string & name);

struct common_speculative_span {
    uint32_t producer_id;
    uint32_t configuration_id;
    uint16_t begin;
    uint16_t end;
    bool feedback_valid = false;
    uint64_t feedback_key = 0;
    uint16_t feedback_value = 0;
};

struct common_speculative_candidate_metadata {
    int64_t proposal_time_us = 0;
    int64_t cycle_proposal_time_us = 0;
    int32_t match_length = -1;
    int32_t match_distance = -1;
    int32_t hit_count = -1;
    int32_t alternative_count = -1;
    int32_t previous_accepted_length = -1;
    int32_t context_length = 0;
    int32_t context_capacity = 0;
    int32_t batch_size = 0;
    int32_t ubatch_size = 0;
    int32_t active_slots = 1;
    std::vector<uint16_t> agreement_prefix;
};

struct common_speculative_candidate {
    uint64_t candidate_id = 0;
    uint32_t producer_id = 0;
    uint32_t configuration_id = 0;
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    llama_tokens tokens;
    common_speculative_candidate_metadata metadata;
    std::vector<common_speculative_span> provenance;
};

struct common_speculative_feedback {
    uint64_t candidate_id = 0;
    uint16_t proposed_length = 0;
    uint16_t accepted_length = 0;
    int64_t proposal_time_us = 0;
    int64_t verification_time_us = 0;
    int64_t rollback_or_replay_time_us = 0;
    bool selected = false;
    bool fully_observed = false;
};

struct common_speculative_selection {
    int32_t candidate_index = -1;
    uint16_t prefix_length = 0;
    double utility = 0.0;
    double admission_lower_confidence = 0.0;
    double admission_upper_confidence = 1.0;
    double admission_evidence = 0.0;
    bool admission_bootstrap = false;
    uint8_t admission_level = 0;
    uint64_t challenger_cooldown_remaining = 0;
    uint8_t challenger_failure_streak = 0;
    uint16_t unbudgeted_prefix_length = 0;
    bool budget_limited = false;
    uint64_t hot_key = 0;
    uint16_t hot_rung = 0;
    uint16_t hot_requested_length = 0;
    bool hot_applied = false;
    uint8_t hot_transition_reason = 0;
    std::vector<double> prefix_confidence;
};

enum common_speculative_hot_event_type {
    COMMON_SPECULATIVE_HOT_NONE,
    COMMON_SPECULATIVE_HOT_ENTRY,
    COMMON_SPECULATIVE_HOT_PROMOTION,
    COMMON_SPECULATIVE_HOT_RETENTION,
    COMMON_SPECULATIVE_HOT_ROLLBACK,
    COMMON_SPECULATIVE_HOT_RESET,
    COMMON_SPECULATIVE_HOT_EXPIRATION,
};

struct common_speculative_hot_event {
    common_speculative_hot_event_type type;
    uint64_t hot_key;
    uint32_t producer_id;
    uint16_t previous_rung;
    uint16_t current_rung;
    uint16_t proposed_length;
    uint16_t accepted_length;
};

struct common_speculative_hot_metrics {
    uint64_t entries = 0;
    uint64_t promotions_24 = 0;
    uint64_t promotions_32 = 0;
    uint64_t promotions_48 = 0;
    uint64_t full_matches = 0;
    uint64_t retained = 0;
    uint64_t rollbacks = 0;
    uint64_t resets = 0;
    uint64_t expirations = 0;
    uint64_t selected_tokens = 0;
    uint64_t accepted_tokens = 0;
};

enum common_speculative_shadow_event_type {
    COMMON_SPECULATIVE_SHADOW_STARTED,
    COMMON_SPECULATIVE_SHADOW_SUCCEEDED_8,
    COMMON_SPECULATIVE_SHADOW_SUCCEEDED_16,
    COMMON_SPECULATIVE_SHADOW_FAILED,
    COMMON_SPECULATIVE_SHADOW_CENSORED,
    COMMON_SPECULATIVE_SHADOW_DROPPED,
};

struct common_speculative_shadow_event {
    common_speculative_shadow_event_type type;
    uint64_t probe_id;
    uint64_t candidate_id;
    uint16_t matched_length;
};

struct common_speculative_shadow_metrics {
    uint64_t started = 0;
    uint64_t succeeded_8 = 0;
    uint64_t succeeded_16 = 0;
    uint64_t failed = 0;
    uint64_t censored = 0;
    uint64_t dropped = 0;
    uint64_t blocked_decisions = 0;
    uint64_t provisional_decisions = 0;
    uint64_t trusted_decisions = 0;
    uint64_t rejected_no_evidence = 0;
    uint64_t rejected_confidence = 0;
};

void common_speculative_apply_global_budget(
        std::vector<common_speculative_selection> & selections,
        uint32_t budget,
        uint32_t & fairness_cursor);

class common_speculative_controller {
public:
    common_speculative_controller(
            common_speculative_controller_mode mode,
            float safety_margin,
            float decay,
            uint32_t warmup = 4,
            uint32_t max_verify = 0,
            common_speculative_controller_persistence persistence = COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS,
            uint32_t n_seq = 1,
            uint32_t max_namespaces = 64);

    common_speculative_controller_mode mode() const;
    uint32_t max_verify() const;

    void begin_request(llama_seq_id seq_id = 0, const std::string & process_namespace = {});
    void end_request(llama_seq_id seq_id = 0);
    size_t process_namespace_count() const;
    uint64_t process_namespace_evictions() const;
    uint64_t process_namespace_fallbacks() const;
    bool allow_challengers(llama_seq_id seq_id = 0) const;

    common_speculative_selection select(
            const std::vector<common_speculative_candidate> & candidates,
            llama_seq_id seq_id = 0);

    void observe(
            const std::vector<common_speculative_candidate> & candidates,
            uint64_t selected_candidate_id,
            const llama_tokens & realized,
            int64_t verification_time_us,
            int64_t rollback_or_replay_time_us,
            llama_seq_id seq_id = 0);

    void observe_ordinary(int64_t decode_time_us, llama_seq_id seq_id = 0);
    void observe_output_token(llama_token token, llama_seq_id seq_id = 0);
    std::vector<common_speculative_shadow_event> take_shadow_events(llama_seq_id seq_id = 0);
    std::vector<common_speculative_hot_event> take_hot_events(llama_seq_id seq_id = 0);
    const common_speculative_shadow_metrics & shadow_metrics() const;
    const common_speculative_hot_metrics & hot_metrics() const;
    size_t active_hot_entries() const;

private:
    struct position_stats {
        double accepted = 1.0;
        double rejected = 1.0;
    };

    struct producer_stats {
        std::vector<position_stats> positions;
        double proposal_time_us = 0.0;
        uint64_t observations = 0;
    };

    struct admission_stats {
        double survived_8 = 1.0;
        double failed_8 = 1.0;
        double survived_16 = 1.0;
        double failed_16 = 1.0;
        uint64_t outcomes_8 = 0;
        uint64_t outcomes_16 = 0;
    };

    struct admission_decision {
        bool admit = false;
        bool bootstrap = false;
        uint8_t level = 0;
        uint16_t verification_cap = 0;
        double lower_confidence = 0.0;
        double upper_confidence = 1.0;
        double evidence = 0.0;
    };

    struct shadow_contributor {
        common_speculative_candidate candidate;
        uint64_t admission_key = 0;
        uint64_t cooldown_key = 0;
        uint64_t context_key = 0;
    };

    struct shadow_probe {
        uint64_t probe_id = 0;
        uint64_t candidate_id = 0;
        llama_tokens tokens;
        std::vector<shadow_contributor> contributors;
        uint16_t matched = 0;
        uint16_t mtp_agreement = 0;
        uint16_t original_length = 0;
        uint8_t family_support = 0;
        bool recorded_8 = false;
    };

    struct cost_stats {
        double time_us = 0.0;
        uint64_t observations = 0;
    };

    struct learning_state {
        double ordinary_decode_time_us = 0.0;
        std::map<uint16_t, cost_stats> verification_costs;
        std::map<uint32_t, std::map<uint16_t, cost_stats>> verification_context_costs;
        std::map<uint64_t, producer_stats> stats;
        std::map<uint64_t, admission_stats> admissions;
    };

    struct challenger_cooldown {
        uint64_t next_observation = 0;
        uint8_t failure_streak = 0;
    };

    struct hot_region {
        uint32_t producer_id = 0;
        uint16_t rung = 16;
        uint64_t last_selected_observation = 0;
        common_speculative_hot_event_type last_transition = COMMON_SPECULATIVE_HOT_NONE;
    };

    struct pending_hot_feedback {
        uint64_t candidate_id = 0;
        uint64_t hot_key = 0;
        uint32_t producer_id = 0;
        uint16_t target_rung = 0;
        bool active = false;
    };

    struct sequence_state {
        learning_state request_learning;
        std::string process_namespace;
        bool process_namespace_available = true;
        bool request_active = false;
        uint64_t request_observations = 0;
        std::map<uint64_t, producer_stats> request_stats;
        bool challenger_started = false;
        uint64_t next_challenger_observation = 0;
        uint8_t empty_discovery_streak = 0;
        std::map<uint64_t, challenger_cooldown> challenger_cooldowns;
        bool shadow_discovery_pending = false;
        uint64_t next_probe_id = 1;
        std::vector<shadow_probe> shadow_probes;
        std::vector<common_speculative_shadow_event> shadow_events;
        std::map<uint64_t, hot_region> hot_regions;
        pending_hot_feedback pending_hot;
        std::vector<common_speculative_hot_event> hot_events;
    };

    struct process_namespace_state {
        learning_state learning;
        uint64_t last_used = 0;
    };

    double score_prefix(const common_speculative_candidate & candidate, uint16_t length, llama_seq_id seq_id) const;
    double prefix_confidence(const common_speculative_candidate & candidate, uint16_t position, llama_seq_id seq_id) const;
    double conditional_acceptance(const common_speculative_candidate & candidate, uint16_t position, llama_seq_id seq_id) const;
    double verification_cost(const common_speculative_candidate & candidate, uint16_t length, llama_seq_id seq_id) const;
    static uint32_t context_bucket(const common_speculative_candidate & candidate);
    static uint32_t verification_shape_key(const common_speculative_candidate & candidate);
    static uint64_t context_key(const common_speculative_candidate & candidate);
    static uint32_t admission_producer_id(const common_speculative_candidate & candidate);
    static uint64_t admission_key(const common_speculative_candidate & candidate);
    static uint64_t cooldown_key(const common_speculative_candidate & candidate);
    static uint64_t hot_key(const common_speculative_candidate & candidate);
    static bool is_ngram(const common_speculative_candidate & candidate);
    static uint16_t ngram_span_begin(const common_speculative_candidate & candidate);
    static uint16_t next_hot_rung(uint16_t rung);
    static uint16_t previous_hot_rung(uint16_t rung);
    static bool agrees_with_complete_mtp(const common_speculative_candidate & candidate, const std::vector<common_speculative_candidate> & candidates);
    static uint16_t agreement_with_mtp(const common_speculative_candidate & candidate, const std::vector<common_speculative_candidate> & candidates);
    static size_t distinct_family_support(const common_speculative_candidate & candidate, const std::vector<common_speculative_candidate> & candidates, uint16_t horizon);
    admission_decision evaluate_admission(const common_speculative_candidate & candidate, const std::vector<common_speculative_candidate> & candidates, llama_seq_id seq_id) const;
    void register_shadow_probes(const std::vector<common_speculative_candidate> & candidates, llama_seq_id seq_id);
    void update_shadow_acceptance(const shadow_contributor & contributor, size_t accepted, bool mismatch, llama_seq_id seq_id);
    void update_shadow_admission(const shadow_contributor & contributor, bool success_8, bool failure_8, bool success_16, bool failure_16, llama_seq_id seq_id);
    void expire_hot_regions(llama_seq_id seq_id);
    learning_state & learning(llama_seq_id seq_id);
    const learning_state & learning(llama_seq_id seq_id) const;
    sequence_state & sequence(llama_seq_id seq_id);
    const sequence_state & sequence(llama_seq_id seq_id) const;
    producer_stats & get_stats(uint64_t key, llama_seq_id seq_id);
    const producer_stats * find_stats(uint64_t key, llama_seq_id seq_id) const;

    common_speculative_controller_mode mode_;
    double safety_margin_;
    double decay_;
    uint32_t warmup_;
    uint32_t max_verify_;
    common_speculative_controller_persistence persistence_;
    uint32_t max_namespaces_;
    uint64_t namespace_clock_ = 0;
    uint64_t namespace_evictions_ = 0;
    uint64_t namespace_fallbacks_ = 0;
    std::map<std::string, process_namespace_state> process_namespaces_;
    std::vector<sequence_state> sequences_;
    common_speculative_shadow_metrics shadow_metrics_;
    common_speculative_hot_metrics hot_metrics_;
};
