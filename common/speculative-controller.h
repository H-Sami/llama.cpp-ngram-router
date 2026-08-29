#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <map>
#include <set>
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
    uint64_t challenger_cooldown_remaining = 0;
    uint8_t challenger_failure_streak = 0;
    uint16_t unbudgeted_prefix_length = 0;
    bool budget_limited = false;
    std::vector<double> prefix_confidence;
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
        double survived = 1.0;
        double failed = 1.0;
    };

    struct admission_decision {
        bool admit = false;
        bool bootstrap = false;
        double lower_confidence = 0.0;
        double upper_confidence = 1.0;
        double evidence = 0.0;
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

    struct sequence_state {
        learning_state request_learning;
        std::string process_namespace;
        bool process_namespace_available = true;
        bool request_active = false;
        uint64_t request_observations = 0;
        std::map<uint64_t, producer_stats> request_stats;
        bool challenger_started = false;
        bool challenger_failed = false;
        uint64_t next_challenger_observation = 0;
        uint8_t empty_discovery_streak = 0;
        uint8_t challenger_failure_streak = 0;
    };

    struct process_namespace_state {
        learning_state learning;
        uint64_t last_used = 0;
    };

    double score_prefix(const common_speculative_candidate & candidate, uint16_t length, llama_seq_id seq_id) const;
    double prefix_confidence(const common_speculative_candidate & candidate, uint16_t position, llama_seq_id seq_id) const;
    double verification_cost(const common_speculative_candidate & candidate, uint16_t length, llama_seq_id seq_id) const;
    static uint32_t context_bucket(const common_speculative_candidate & candidate);
    static uint32_t verification_shape_key(const common_speculative_candidate & candidate);
    static uint64_t context_key(const common_speculative_candidate & candidate);
    static uint64_t admission_key(const common_speculative_candidate & candidate);
    static bool has_bootstrap_consensus(const common_speculative_candidate & candidate);
    admission_decision evaluate_admission(const common_speculative_candidate & candidate, llama_seq_id seq_id) const;
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
};
