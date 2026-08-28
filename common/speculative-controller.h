#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

const char * common_speculative_controller_mode_name(common_speculative_controller_mode mode);
common_speculative_controller_mode common_speculative_controller_mode_from_name(const std::string & name);

struct common_speculative_span {
    uint32_t producer_id;
    uint32_t configuration_id;
    uint16_t begin;
    uint16_t end;
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
};

class common_speculative_controller {
public:
    common_speculative_controller(
            common_speculative_controller_mode mode,
            float safety_margin,
            float decay,
            uint32_t warmup = 4);

    common_speculative_controller_mode mode() const;

    void begin_request();
    bool allow_challengers() const;

    common_speculative_selection select(const std::vector<common_speculative_candidate> & candidates);

    void observe(
            const std::vector<common_speculative_candidate> & candidates,
            uint64_t selected_candidate_id,
            const llama_tokens & realized,
            int64_t verification_time_us,
            int64_t rollback_or_replay_time_us);

    void observe_ordinary(int64_t decode_time_us);

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

    double score_prefix(const common_speculative_candidate & candidate, uint16_t length) const;
    double verification_cost(uint16_t length) const;
    static uint64_t context_key(const common_speculative_candidate & candidate);
    producer_stats & get_stats(uint64_t key);
    const producer_stats * find_stats(uint64_t key) const;

    common_speculative_controller_mode mode_;
    double safety_margin_;
    double decay_;
    double ordinary_decode_time_us_ = 0.0;
    struct cost_stats {
        double time_us = 0.0;
        uint64_t observations = 0;
    };

    std::map<uint16_t, cost_stats> verification_costs_;
    uint32_t warmup_;
    uint64_t observations_ = 0;
    uint64_t request_observations_ = 0;
    std::map<uint64_t, producer_stats> stats_;
    std::map<uint64_t, producer_stats> request_stats_;
    bool challenger_probed_ = false;
    bool challenger_failed_ = false;
};
