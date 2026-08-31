#include "ngram-retrieval.h"
#include "ngram-retrieval-policy.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace {

constexpr std::array<size_t, 3> anchor_sizes = { 4, 8, 16 };
constexpr size_t anchor_size = anchor_sizes.front();
constexpr size_t max_postings = 32;
constexpr size_t max_match = 48;

struct source_match {
    uint32_t position;
    uint16_t match_length;
    uint32_t distance;
    std::vector<llama_token> continuation;
};

struct trie_node {
    std::map<llama_token, size_t> children;
    uint32_t support = 0;
    uint64_t weight = 0;
    uint32_t nearest = std::numeric_limits<uint32_t>::max();
    uint16_t match_length = 0;
};

uint16_t tier(uint32_t value, uint32_t one, uint32_t two, uint32_t three) {
    if (value >= three) return 3;
    if (value >= two) return 2;
    if (value >= one) return 1;
    return 0;
}

}

common_ngram_retrieval::common_ngram_retrieval(uint16_t n_max, uint8_t hash_bits)
    : n_max_(n_max), hash_bits_(hash_bits) {
    if (n_max < 8 || n_max > 48) {
        throw std::invalid_argument("ngram retrieval n-max must be between 8 and 48");
    }
    if (hash_bits > 64) {
        throw std::invalid_argument("ngram retrieval hash bits must not exceed 64");
    }
}

uint64_t common_ngram_retrieval::hash_anchor(const llama_token * tokens, size_t length) const {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < length; ++i) {
        uint32_t value = (uint32_t) tokens[i];
        for (size_t byte = 0; byte < sizeof(value); ++byte) {
            hash ^= (value >> (byte * 8)) & 0xffu;
            hash *= 1099511628211ull;
        }
    }
    if (hash_bits_ == 64) {
        return hash;
    }
    return hash_bits_ == 0 ? 0 : hash & ((1ull << hash_bits_) - 1);
}

void common_ngram_retrieval::append_from(size_t old_size) {
    for (size_t anchor_index = 0; anchor_index < anchor_sizes.size(); ++anchor_index) {
        const size_t length = anchor_sizes[anchor_index];
        const size_t first = old_size >= length - 1 ? old_size - (length - 1) : 0;
        for (size_t position = first; position + length <= tokens_.size(); ++position) {
            auto & posting = postings_[anchor_index][hash_anchor(tokens_.data() + position, length)];
            if (!posting.empty() && posting.back() == position) continue;
            posting.push_back(position);
            if (posting.size() > max_postings) {
                posting.erase(posting.begin(), posting.begin() + (posting.size() - max_postings));
            }
        }
    }
}

void common_ngram_retrieval::reset(const std::vector<llama_token> & tokens) {
    tokens_.clear();
    for (auto & index : postings_) index.clear();
    tokens_ = tokens;
    append_from(0);
}

void common_ngram_retrieval::synchronize(const std::vector<llama_token> & tokens) {
    const size_t common = std::min(tokens_.size(), tokens.size());
    if (!std::equal(tokens_.begin(), tokens_.begin() + common, tokens.begin()) || tokens.size() < tokens_.size()) {
        reset(tokens);
        return;
    }
    if (tokens.size() == tokens_.size()) {
        return;
    }
    const size_t old_size = tokens_.size();
    tokens_.insert(tokens_.end(), tokens.begin() + old_size, tokens.end());
    append_from(old_size);
}

void common_ngram_retrieval::synchronize(const std::vector<llama_token> & prefix, llama_token last) {
    const size_t size = prefix.size() + 1;
    const size_t common = std::min(tokens_.size(), size);
    for (size_t i = 0; i < common; ++i) {
        const llama_token token = i < prefix.size() ? prefix[i] : last;
        if (tokens_[i] != token) {
            tokens_.assign(prefix.begin(), prefix.end());
            tokens_.push_back(last);
            for (auto & index : postings_) index.clear();
            append_from(0);
            return;
        }
    }
    if (size < tokens_.size()) {
        tokens_.assign(prefix.begin(), prefix.end());
        tokens_.push_back(last);
        for (auto & index : postings_) index.clear();
        append_from(0);
        return;
    }
    if (size == tokens_.size()) {
        return;
    }

    const size_t old_size = tokens_.size();
    if (old_size < prefix.size()) {
        tokens_.insert(tokens_.end(), prefix.begin() + old_size, prefix.end());
    }
    tokens_.push_back(last);
    append_from(old_size);
}

common_ngram_retrieval_policy common_ngram_retrieval::best_policy() const {
    common_ngram_retrieval_policy result = COMMON_NGRAM_RETRIEVAL_DEVELOPMENT_POLICY;
    double best = -1.0;
    for (size_t i = 0; i < policy_stats_.size(); ++i) {
        const auto & stats = policy_stats_[i];
        if (stats.observations < 3) {
            continue;
        }
        const double score = stats.accepted / std::max(1.0, stats.time_us);
        if (score > best) {
            best = score;
            result = (common_ngram_retrieval_policy) i;
        }
    }
    return result;
}

common_ngram_retrieval_result common_ngram_retrieval::query(
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & mtp_prefix,
        common_ngram_retrieval_policy policy) {
    synchronize(tokens);
    return query_view(tokens, nullptr, mtp_prefix, policy);
}

common_ngram_retrieval_result common_ngram_retrieval::query(
        const std::vector<llama_token> & prefix,
        llama_token last,
        const std::vector<llama_token> & mtp_prefix,
        common_ngram_retrieval_policy policy) {
    synchronize(prefix, last);
    return query_view(prefix, &last, mtp_prefix, policy);
}

common_ngram_retrieval_result common_ngram_retrieval::query_transient(
        const std::vector<llama_token> & prefix,
        llama_token last,
        const std::vector<llama_token> & mtp_prefix,
        common_ngram_retrieval_policy policy) {
    return query_view(prefix, &last, mtp_prefix, policy);
}

common_ngram_retrieval_result common_ngram_retrieval::query_view(
        const std::vector<llama_token> & prefix,
        const llama_token * last,
        const std::vector<llama_token> & mtp_prefix,
        common_ngram_retrieval_policy policy) {
    metrics_.queries++;
    common_ngram_retrieval_result result;
    result.policy = policy == COMMON_NGRAM_RETRIEVAL_POLICY_COUNT ? best_policy() : policy;
    const size_t view_size = prefix.size() + (last != nullptr ? 1 : 0);
    const auto view_token = [&](size_t position) {
        return position < prefix.size() ? prefix[position] : *last;
    };
    if (view_size < anchor_size + 1) {
        metrics_.rejected_short_history++;
        return result;
    }

    bool had_hash_hit = false;
    std::map<uint32_t, source_match> unique_matches;
    for (size_t anchor_index = anchor_sizes.size(); anchor_index-- > 0;) {
        const size_t length = anchor_sizes[anchor_index];
        if (view_size < length + 1) continue;
        const size_t current = view_size - length;
        llama_token anchor[anchor_sizes.back()];
        for (size_t i = 0; i < length; ++i) anchor[i] = view_token(current + i);
        const auto found = postings_[anchor_index].find(hash_anchor(anchor, length));
        if (found == postings_[anchor_index].end()) continue;
        had_hash_hit = true;
        for (const uint32_t position : found->second) {
            if (position >= current || position + length > tokens_.size()) continue;
            bool equal = true;
            for (size_t i = 0; i < length; ++i) equal &= tokens_[position + i] == view_token(current + i);
            if (!equal) {
                metrics_.rejected_collision++;
                continue;
            }
            size_t backward = 0;
            while (backward + length < max_match && backward < position && backward < current &&
                    tokens_[position - backward - 1] == view_token(current - backward - 1)) {
                backward++;
            }
            const size_t continuation_begin = position + length;
            const size_t continuation_end = std::min({ current, continuation_begin + n_max_, tokens_.size() });
            if (continuation_begin >= continuation_end) continue;
            source_match match;
            match.position = position;
            match.match_length = length + backward;
            match.distance = view_size - continuation_begin;
            match.continuation.assign(tokens_.begin() + continuation_begin, tokens_.begin() + continuation_end);
            auto existing = unique_matches.find(continuation_begin);
            if (existing == unique_matches.end() || existing->second.match_length < match.match_length) {
                unique_matches[continuation_begin] = std::move(match);
            }
        }
    }
    if (!had_hash_hit) {
        metrics_.rejected_no_hash_hit++;
        return result;
    }
    std::vector<source_match> matches;
    matches.reserve(unique_matches.size());
    for (auto & item : unique_matches) matches.push_back(std::move(item.second));
    if (matches.empty()) {
        metrics_.rejected_no_continuation++;
        return result;
    }

    std::vector<trie_node> trie(1);
    for (const auto & match : matches) {
        size_t node = 0;
        for (const llama_token token : match.continuation) {
            auto inserted = trie[node].children.emplace(token, trie.size());
            if (inserted.second) {
                trie.emplace_back();
            }
            node = inserted.first->second;
            trie[node].support++;
            trie[node].weight += match.match_length;
            trie[node].nearest = std::min(trie[node].nearest, match.distance);
            trie[node].match_length = std::max(trie[node].match_length, match.match_length);
        }
    }

    size_t node = 0;
    while (result.tokens.size() < n_max_ && !trie[node].children.empty()) {
        auto best = trie[node].children.begin();
        for (auto it = std::next(best); it != trie[node].children.end(); ++it) {
            const auto & lhs = trie[it->second];
            const auto & rhs = trie[best->second];
            bool take = false;
            if (result.policy == COMMON_NGRAM_RETRIEVAL_LONGEST_RECENT) {
                take = lhs.match_length > rhs.match_length ||
                    (lhs.match_length == rhs.match_length && lhs.nearest < rhs.nearest);
            } else if (result.policy == COMMON_NGRAM_RETRIEVAL_MAJORITY_PREFIX) {
                take = lhs.support > rhs.support ||
                    (lhs.support == rhs.support && lhs.nearest < rhs.nearest);
            } else {
                take = lhs.weight > rhs.weight ||
                    (lhs.weight == rhs.weight && lhs.nearest < rhs.nearest);
            }
            if (take) {
                best = it;
            }
        }

        uint32_t total_support = 0;
        for (const auto & child : trie[node].children) {
            total_support += trie[child.second].support;
        }
        node = best->second;
        const auto & chosen = trie[node];
        const size_t position = result.tokens.size();
        const uint16_t dominance = (uint16_t) (1000u * chosen.support / std::max(1u, total_support));
        if (position >= 3 && !result.evidence.empty()) {
            const auto & previous = result.evidence.back();
            const bool context_collapse = chosen.match_length + 8 <= previous.match_length;
            const bool source_jump = chosen.nearest > previous.source_distance &&
                chosen.nearest - previous.source_distance > 32;
            const bool support_collapse = chosen.support < previous.source_support;
            if (source_jump && (context_collapse || support_collapse)) {
                break;
            }
        }
        result.tokens.push_back(best->first);
        result.evidence.push_back({
            chosen.match_length,
            (uint16_t) std::min<uint32_t>(chosen.support, std::numeric_limits<uint16_t>::max()),
            dominance,
            chosen.nearest,
            position < mtp_prefix.size() && mtp_prefix[position] == best->first,
        });
    }

    if (result.tokens.empty()) {
        metrics_.rejected_no_continuation++;
        return result;
    }
    result.verified_matches = matches.size();
    result.alternatives = trie[0].children.size();
    const auto & first = result.evidence.front();
    const uint64_t match_tier = tier(first.match_length, 8, 16, 32);
    const uint64_t support_tier = tier(first.source_support, 2, 4, 8);
    const uint64_t dominance_tier = tier(first.dominance_permille, 500, 750, 900);
    const uint64_t recency_tier = first.source_distance <= 32 ? 3 : first.source_distance <= 128 ? 2 : first.source_distance <= 512 ? 1 : 0;
    result.evidence_key = match_tier | (support_tier << 2) | (dominance_tier << 4) | (recency_tier << 6);
    metrics_.verified_matches += matches.size();
    metrics_.candidates++;
    metrics_.match_tiers[match_tier]++;
    metrics_.support_tiers[support_tier]++;
    return result;
}

void common_ngram_retrieval::observe(
        common_ngram_retrieval_policy policy,
        uint16_t proposed,
        uint16_t accepted,
        int64_t elapsed_us) {
    if (policy >= COMMON_NGRAM_RETRIEVAL_POLICY_COUNT || proposed == 0) {
        return;
    }
    auto & stats = policy_stats_[policy];
    constexpr double decay = 0.98;
    stats.accepted = 1.0 + decay * (stats.accepted - 1.0) + accepted;
    stats.proposed = 2.0 + decay * (stats.proposed - 2.0) + proposed;
    stats.time_us = decay * stats.time_us + std::max<int64_t>(1, elapsed_us);
    stats.observations++;
    metrics_.selected_prefixes++;
    metrics_.accepted_tokens += accepted;
}

size_t common_ngram_retrieval::posting_count(uint64_t hash) const {
    size_t result = 0;
    for (const auto & index : postings_) {
        const auto it = index.find(hash);
        if (it != index.end()) result += it->second.size();
    }
    return result;
}

size_t common_ngram_retrieval::max_posting_count() const {
    size_t result = 0;
    for (const auto & index : postings_) {
        for (const auto & posting : index) {
            result = std::max(result, posting.second.size());
        }
    }
    return result;
}

size_t common_ngram_retrieval::indexed_tokens() const {
    return tokens_.size();
}

const common_ngram_retrieval_metrics & common_ngram_retrieval::metrics() const {
    return metrics_;
}
