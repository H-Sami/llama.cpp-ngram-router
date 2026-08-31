#pragma once

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

enum common_ngram_retrieval_policy {
    COMMON_NGRAM_RETRIEVAL_LONGEST_RECENT,
    COMMON_NGRAM_RETRIEVAL_MAJORITY_PREFIX,
    COMMON_NGRAM_RETRIEVAL_WEIGHTED_PREFIX,
    COMMON_NGRAM_RETRIEVAL_POLICY_COUNT,
};

struct common_ngram_retrieval_evidence {
    uint16_t match_length = 0;
    uint16_t source_support = 0;
    uint16_t dominance_permille = 0;
    uint32_t source_distance = 0;
    bool mtp_agreement = false;
};

struct common_ngram_retrieval_result {
    std::vector<llama_token> tokens;
    std::vector<common_ngram_retrieval_evidence> evidence;
    common_ngram_retrieval_policy policy = COMMON_NGRAM_RETRIEVAL_WEIGHTED_PREFIX;
    uint64_t evidence_key = 0;
    uint16_t verified_matches = 0;
    uint16_t alternatives = 0;
};

struct common_ngram_retrieval_metrics {
    uint64_t queries = 0;
    uint64_t verified_matches = 0;
    uint64_t candidates = 0;
    uint64_t accepted_tokens = 0;
    uint64_t selected_prefixes = 0;
    uint64_t rejected_short_history = 0;
    uint64_t rejected_no_hash_hit = 0;
    uint64_t rejected_collision = 0;
    uint64_t rejected_no_continuation = 0;
    std::array<uint64_t, 4> match_tiers = {};
    std::array<uint64_t, 4> support_tiers = {};
};

class common_ngram_retrieval {
public:
    explicit common_ngram_retrieval(uint16_t n_max = 48, uint8_t hash_bits = 64);

    void reset(const std::vector<llama_token> & tokens = {});
    void synchronize(const std::vector<llama_token> & tokens);
    common_ngram_retrieval_result query(
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & mtp_prefix = {},
            common_ngram_retrieval_policy policy = COMMON_NGRAM_RETRIEVAL_POLICY_COUNT);
    common_ngram_retrieval_result query(
            const std::vector<llama_token> & prefix,
            llama_token last,
            const std::vector<llama_token> & mtp_prefix = {},
            common_ngram_retrieval_policy policy = COMMON_NGRAM_RETRIEVAL_POLICY_COUNT);
    common_ngram_retrieval_result query_transient(
            const std::vector<llama_token> & prefix,
            llama_token last,
            const std::vector<llama_token> & mtp_prefix = {},
            common_ngram_retrieval_policy policy = COMMON_NGRAM_RETRIEVAL_POLICY_COUNT);
    void observe(common_ngram_retrieval_policy policy, uint16_t proposed, uint16_t accepted, int64_t elapsed_us = 0);

    size_t posting_count(uint64_t hash) const;
    size_t max_posting_count() const;
    size_t indexed_tokens() const;
    const common_ngram_retrieval_metrics & metrics() const;

private:
    struct policy_stats {
        double accepted = 1.0;
        double proposed = 2.0;
        double time_us = 1.0;
        uint64_t observations = 0;
    };

    uint64_t hash_anchor(const llama_token * tokens, size_t length) const;
    void append_from(size_t old_size);
    void synchronize(const std::vector<llama_token> & prefix, llama_token last);
    common_ngram_retrieval_result query_view(
            const std::vector<llama_token> & prefix,
            const llama_token * last,
            const std::vector<llama_token> & mtp_prefix,
            common_ngram_retrieval_policy policy);
    common_ngram_retrieval_policy best_policy() const;

    uint16_t n_max_;
    uint8_t hash_bits_;
    std::vector<llama_token> tokens_;
    std::array<std::unordered_map<uint64_t, std::vector<uint32_t>>, 3> postings_;
    std::array<policy_stats, COMMON_NGRAM_RETRIEVAL_POLICY_COUNT> policy_stats_;
    common_ngram_retrieval_metrics metrics_;
};
