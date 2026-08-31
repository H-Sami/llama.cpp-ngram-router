#include "speculative-controller.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

void common_speculative_apply_global_budget(
        std::vector<common_speculative_selection> & selections,
        uint32_t budget,
        uint32_t & fairness_cursor) {
    if (selections.empty()) {
        return;
    }

    uint64_t requested = 0;
    for (auto & selection : selections) {
        selection.unbudgeted_prefix_length = selection.prefix_length;
        selection.budget_limited = false;
        requested += selection.prefix_length;
    }
    if (budget == 0 || requested <= budget) {
        return;
    }

    std::vector<uint16_t> allocated(selections.size(), 0);
    const size_t start = fairness_cursor % selections.size();
    for (size_t offset = 0; offset < selections.size() && budget > 0; ++offset) {
        const size_t i = (start + offset) % selections.size();
        if (selections[i].candidate_index >= 0 && selections[i].prefix_length > 0) {
            allocated[i] = 1;
            budget--;
        }
    }
    fairness_cursor = (start + 1) % selections.size();

    while (budget > 0) {
        size_t best = selections.size();
        double best_confidence = -1.0;
        for (size_t offset = 0; offset < selections.size(); ++offset) {
            const size_t i = (start + offset) % selections.size();
            if (allocated[i] >= selections[i].prefix_length) {
                continue;
            }
            const size_t position = allocated[i];
            const double confidence = position < selections[i].prefix_confidence.size()
                ? selections[i].prefix_confidence[position]
                : 0.0;
            if (confidence > best_confidence) {
                best = i;
                best_confidence = confidence;
            }
        }
        if (best == selections.size()) {
            break;
        }
        allocated[best]++;
        budget--;
    }

    for (size_t i = 0; i < selections.size(); ++i) {
        selections[i].budget_limited = allocated[i] < selections[i].prefix_length;
        selections[i].prefix_length = allocated[i];
        if (allocated[i] == 0) {
            selections[i].candidate_index = -1;
            selections[i].utility = 0.0;
        }
    }
}

const char * common_speculative_controller_mode_name(common_speculative_controller_mode mode) {
    switch (mode) {
        case COMMON_SPECULATIVE_CONTROLLER_MODE_OFF:      return "off";
        case COMMON_SPECULATIVE_CONTROLLER_MODE_SHADOW:   return "shadow";
        case COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE: return "adaptive";
        case COMMON_SPECULATIVE_CONTROLLER_MODE_REPLAY:   return "replay";
    }

    return "unknown";
}

common_speculative_controller_mode common_speculative_controller_mode_from_name(const std::string & name) {
    if (name == "off") {
        return COMMON_SPECULATIVE_CONTROLLER_MODE_OFF;
    }
    if (name == "shadow") {
        return COMMON_SPECULATIVE_CONTROLLER_MODE_SHADOW;
    }
    if (name == "adaptive") {
        return COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE;
    }
    if (name == "replay") {
        return COMMON_SPECULATIVE_CONTROLLER_MODE_REPLAY;
    }

    throw std::invalid_argument("unknown speculative controller mode: " + name);
}

const char * common_speculative_controller_persistence_name(common_speculative_controller_persistence persistence) {
    switch (persistence) {
        case COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_REQUEST: return "request";
        case COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS: return "process";
    }

    return "unknown";
}

common_speculative_controller_persistence common_speculative_controller_persistence_from_name(const std::string & name) {
    if (name == "request") {
        return COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_REQUEST;
    }
    if (name == "process") {
        return COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS;
    }

    throw std::invalid_argument("unknown speculative controller persistence: " + name);
}

common_speculative_controller::common_speculative_controller(
        common_speculative_controller_mode mode,
        float safety_margin,
        float decay,
        uint32_t warmup,
        uint32_t max_verify,
        common_speculative_controller_persistence persistence,
        uint32_t n_seq,
        uint32_t max_namespaces)
    : mode_(mode)
    , safety_margin_(std::max(0.0f, safety_margin))
    , decay_(std::clamp((double) decay, 0.0, 1.0))
    , warmup_(warmup)
    , max_verify_(max_verify)
    , persistence_(persistence)
    , max_namespaces_(std::max(1u, max_namespaces))
    , sequences_(std::max(1u, n_seq)) {
    process_namespaces_[{}].last_used = ++namespace_clock_;
}

common_speculative_controller_mode common_speculative_controller::mode() const {
    return mode_;
}

uint32_t common_speculative_controller::max_verify() const {
    return max_verify_;
}

void common_speculative_controller::begin_request(llama_seq_id seq_id, const std::string & process_namespace) {
    auto & state = sequence(seq_id);
    state = {};
    state.process_namespace = process_namespace;
    state.request_active = true;

    if (persistence_ != COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS) {
        return;
    }

    auto it = process_namespaces_.find(process_namespace);
    if (it == process_namespaces_.end() && process_namespaces_.size() >= max_namespaces_) {
        auto victim = process_namespaces_.end();
        for (auto candidate = process_namespaces_.begin(); candidate != process_namespaces_.end(); ++candidate) {
            bool active = false;
            for (const auto & sequence_state : sequences_) {
                active |= sequence_state.request_active && sequence_state.process_namespace_available &&
                    sequence_state.process_namespace == candidate->first;
            }
            if (active) {
                continue;
            }
            if (victim == process_namespaces_.end() || candidate->second.last_used < victim->second.last_used ||
                    (candidate->second.last_used == victim->second.last_used && candidate->first < victim->first)) {
                victim = candidate;
            }
        }
        if (victim == process_namespaces_.end()) {
            state.process_namespace_available = false;
            namespace_fallbacks_++;
            return;
        }
        process_namespaces_.erase(victim);
        namespace_evictions_++;
    }

    process_namespaces_[process_namespace].last_used = ++namespace_clock_;
}

void common_speculative_controller::end_request(llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    for (const auto & probe : state.shadow_probes) {
        state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_CENSORED, probe.probe_id, probe.candidate_id, probe.matched });
        shadow_metrics_.censored++;
    }
    state.shadow_probes.clear();
    state.hot_regions.clear();
    state.pending_hot = {};
    state.request_active = false;
}

size_t common_speculative_controller::process_namespace_count() const {
    return persistence_ == COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS ? process_namespaces_.size() : 0;
}

uint64_t common_speculative_controller::process_namespace_evictions() const {
    return namespace_evictions_;
}

uint64_t common_speculative_controller::process_namespace_fallbacks() const {
    return namespace_fallbacks_;
}

bool common_speculative_controller::allow_challengers(llama_seq_id seq_id) const {
    const auto & state = sequence(seq_id);
    if (mode_ != COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE) {
        return true;
    }
    if (state.shadow_discovery_pending) {
        return true;
    }
    if (state.challenger_started) {
        return true;
    }

    // One early shadow sample provides request-local evidence without paying
    // for every n-gram arm throughout the MTP warm-up. Recheck at the warm-up
    // boundary, then periodically until a challenger is admitted.
    if (state.request_observations < warmup_) {
        return state.request_observations == 0;
    }

    return state.request_observations >= state.next_challenger_observation;
}

uint32_t common_speculative_controller::context_bucket(const common_speculative_candidate & candidate) {
    uint32_t context_bucket = 0;
    for (int32_t n = std::max(1, candidate.metadata.context_length); n > 1; n >>= 1) {
        context_bucket++;
    }

    return context_bucket;
}

uint32_t common_speculative_controller::verification_shape_key(const common_speculative_candidate & candidate) {
    const auto bucket = [](int32_t value) {
        uint32_t result = 0;
        for (int32_t n = std::max(1, value); n > 1; n >>= 1) {
            result++;
        }
        return std::min(result, 31u);
    };
    const uint32_t active_slots = std::clamp(candidate.metadata.active_slots, 1, 255);
    return context_bucket(candidate) |
        (bucket(candidate.metadata.context_capacity) << 6) |
        (bucket(candidate.metadata.batch_size) << 11) |
        (bucket(candidate.metadata.ubatch_size) << 16) |
        (active_slots << 21);
}

uint64_t common_speculative_controller::context_key(const common_speculative_candidate & candidate) {
    if (is_retrieval(candidate)) {
        return (0x52ull << 56) | candidate.metadata.retrieval_evidence_key;
    }
    const uint32_t context_bucket = common_speculative_controller::context_bucket(candidate);

    uint32_t agreement_bucket = 0;
    for (const uint16_t common : candidate.metadata.agreement_prefix) {
        if (common >= 4) {
            agreement_bucket = std::max(agreement_bucket, 2u);
        } else if (common > 0) {
            agreement_bucket = std::max(agreement_bucket, 1u);
        }
    }

    return (1ull << 63) | candidate.producer_id | ((uint64_t) context_bucket << 32) | ((uint64_t) agreement_bucket << 48);
}

uint32_t common_speculative_controller::admission_producer_id(const common_speculative_candidate & candidate) {
    if (is_ngram(candidate) && !candidate.provenance.empty()) {
        return candidate.provenance.back().producer_id;
    }
    return candidate.producer_id;
}

uint64_t common_speculative_controller::admission_key(const common_speculative_candidate & candidate) {
    if (is_retrieval(candidate)) {
        return (0x53ull << 56) | candidate.metadata.retrieval_evidence_key;
    }
    const uint32_t context_bucket = common_speculative_controller::context_bucket(candidate);

    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    mix(admission_producer_id(candidate));
    mix(context_bucket);
    return key;
}

uint64_t common_speculative_controller::cooldown_key(const common_speculative_candidate & candidate) {
    return admission_key(candidate);
}

uint64_t common_speculative_controller::hot_key(const common_speculative_candidate & candidate) {
    return admission_key(candidate);
}

bool common_speculative_controller::is_ngram(const common_speculative_candidate & candidate) {
    return candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE ||
        candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K ||
        candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V ||
        candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD ||
        candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE ||
        candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_RETRIEVAL;
}

bool common_speculative_controller::is_retrieval(const common_speculative_candidate & candidate) {
    return candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_RETRIEVAL;
}

uint16_t common_speculative_controller::ngram_span_begin(const common_speculative_candidate & candidate) {
    return candidate.provenance.size() > 1 ? candidate.provenance.back().begin : 0;
}

uint16_t common_speculative_controller::next_hot_rung(uint16_t rung) {
    if (rung < 24) return 24;
    if (rung < 32) return 32;
    return 48;
}

uint16_t common_speculative_controller::previous_hot_rung(uint16_t rung) {
    if (rung > 32) return 32;
    if (rung > 24) return 24;
    return 16;
}

bool common_speculative_controller::agrees_with_complete_mtp(
        const common_speculative_candidate & candidate,
        const std::vector<common_speculative_candidate> & candidates) {
    bool has_mtp = false;
    for (const auto & peer : candidates) {
        if (peer.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            continue;
        }
        has_mtp = true;
        if (candidate.tokens.size() < peer.tokens.size()) {
            continue;
        }
        if (std::equal(peer.tokens.begin(), peer.tokens.end(), candidate.tokens.begin())) {
            return true;
        }
    }
    return !has_mtp;
}

void common_speculative_controller::expire_hot_regions(llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    for (auto it = state.hot_regions.begin(); it != state.hot_regions.end();) {
        if (state.request_observations - it->second.last_selected_observation < 64) {
            ++it;
            continue;
        }
        state.hot_events.push_back({
            COMMON_SPECULATIVE_HOT_EXPIRATION,
            it->first,
            it->second.producer_id,
            it->second.rung,
            0,
            0,
            0,
        });
        hot_metrics_.expirations++;
        it = state.hot_regions.erase(it);
    }
}

uint16_t common_speculative_controller::agreement_with_mtp(
        const common_speculative_candidate & candidate,
        const std::vector<common_speculative_candidate> & candidates) {
    uint16_t result = 0;
    for (const auto & peer : candidates) {
        if (peer.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            continue;
        }
        const size_t n = std::min(candidate.tokens.size(), peer.tokens.size());
        size_t i = 0;
        while (i < n && candidate.tokens[i] == peer.tokens[i]) {
            i++;
        }
        result = std::max<uint16_t>(result, i);
    }
    return result;
}

size_t common_speculative_controller::distinct_family_support(
        const common_speculative_candidate & candidate,
        const std::vector<common_speculative_candidate> & candidates,
        uint16_t horizon) {
    if (candidate.tokens.size() < horizon) {
        return 0;
    }
    std::set<common_speculative_type> families;
    for (const auto & peer : candidates) {
        if (!is_ngram(peer) || peer.tokens.size() < horizon) {
            continue;
        }
        bool equal = true;
        for (uint16_t i = 0; i < horizon; ++i) {
            equal &= candidate.tokens[i] == peer.tokens[i];
        }
        if (equal) {
            families.insert(peer.type);
        }
    }
    return families.size();
}

common_speculative_controller::admission_decision common_speculative_controller::evaluate_admission(
        const common_speculative_candidate & candidate,
        const std::vector<common_speculative_candidate> & candidates,
        llama_seq_id seq_id) const {
    admission_decision result;
    const uint16_t mtp_agreement = agreement_with_mtp(candidate, candidates);
    const bool mtp_complete = std::any_of(candidates.begin(), candidates.end(), [&](const auto & peer) {
        return peer.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP &&
            mtp_agreement >= std::min(peer.tokens.size(), candidate.tokens.size());
    });
    // Retrieval earns request-local trust from passive target outcomes before Vulkan verification.
    const bool exploratory_trial = !is_retrieval(candidate) && candidate.tokens.size() >= 16;
    const bool provisional_consensus = mtp_complete && distinct_family_support(candidate, candidates, 8) >= 2;
    const bool trusted_consensus = mtp_complete && distinct_family_support(candidate, candidates, 32) >= 3;
    result.bootstrap = exploratory_trial || provisional_consensus || trusted_consensus;
    if (trusted_consensus) {
        result.level = 2;
        result.admit = true;
    } else if (provisional_consensus) {
        result.level = 1;
        result.verification_cap = 16;
        result.admit = true;
    } else if (exploratory_trial) {
        result.level = 1;
        result.verification_cap = 8;
        result.admit = true;
    }

    const auto interval = [](double survived, double failed) {
        const double total = survived + failed;
        const double mean = survived / total;
        const double deviation = 1.96 * std::sqrt((survived * failed) / (total * total * (total + 1.0)));
        return std::pair<double, double>(std::max(0.0, mean - deviation), std::min(1.0, mean + deviation));
    };

    struct evaluated_admission {
        bool blocked = false;
        bool succeeded_8 = false;
        bool succeeded_16 = false;
        bool trusted = false;
        double lower_confidence = 0.0;
        double upper_confidence = 1.0;
        double evidence = 0.0;
    };
    const auto evaluate = [&](const admission_stats & stats) {
        evaluated_admission evaluated;
        const double evidence_8 = stats.outcomes_8;
        const double evidence_16 = stats.outcomes_16;
        const auto confidence_8 = interval(stats.survived_8, stats.failed_8);
        const auto confidence_16 = interval(stats.survived_16, stats.failed_16);
        evaluated.blocked = evidence_8 >= 3.0 && confidence_8.second < 0.5;
        evaluated.succeeded_8 = stats.survived_8 > 1.0;
        evaluated.succeeded_16 = stats.survived_16 > 1.0;
        evaluated.trusted = evidence_16 >= 6.0 && confidence_16.first >= 0.6;
        evaluated.lower_confidence = confidence_16.first;
        evaluated.upper_confidence = confidence_8.second;
        evaluated.evidence = std::max(evidence_8, evidence_16);
        return evaluated;
    };

    const uint64_t key = admission_key(candidate);
    const auto set_diagnostics = [&](const evaluated_admission & evaluated) {
        result.evidence = evaluated.evidence;
        result.lower_confidence = evaluated.lower_confidence;
        result.upper_confidence = evaluated.upper_confidence;
    };
    const auto apply_persistent = [&](const evaluated_admission & evaluated) {
        set_diagnostics(evaluated);
        if (evaluated.blocked && !provisional_consensus && !trusted_consensus) {
            result.level = 0;
            result.verification_cap = 0;
            result.admit = false;
        } else if (!provisional_consensus && !trusted_consensus &&
                candidate.tokens.size() >= 16 &&
                (evaluated.succeeded_8 || evaluated.succeeded_16 || evaluated.trusted)) {
            result.level = 1;
            result.verification_cap = 8;
            result.admit = true;
        }
    };
    const auto apply_request = [&](const evaluated_admission & evaluated) {
        set_diagnostics(evaluated);
        if (evaluated.blocked && !provisional_consensus && !trusted_consensus) {
            result.level = 0;
            result.verification_cap = 0;
            result.admit = false;
        } else if (evaluated.succeeded_16 || evaluated.trusted) {
            result.level = 2;
            result.verification_cap = 0;
            result.admit = true;
        } else if (evaluated.succeeded_8 && !trusted_consensus) {
            result.level = std::max<uint8_t>(result.level, 1);
            result.verification_cap = 16;
            result.admit = true;
        }
    };

    const auto & persistent = learning(seq_id).admissions;
    const auto persistent_it = persistent.find(key);
    if (persistent_it != persistent.end()) {
        apply_persistent(evaluate(persistent_it->second));
    }

    const auto & request = sequence(seq_id).request_learning.admissions;
    const auto request_it = request.find(key);
    if (request_it != request.end()) {
        apply_request(evaluate(request_it->second));
    }
    if (is_retrieval(candidate) &&
            !candidate.metadata.retrieval_match_length.empty() &&
            !candidate.metadata.retrieval_source_support.empty() &&
            !candidate.metadata.retrieval_dominance_permille.empty() &&
            candidate.metadata.retrieval_match_length.front() >= 16 &&
            (candidate.metadata.retrieval_source_support.front() >= 2 ||
                candidate.metadata.retrieval_match_length.front() >= 32) &&
            candidate.metadata.retrieval_dominance_permille.front() >= 900) {
        // Carry target-proven trust across evidence tiers within one request.
        apply_request(evaluate(sequence(seq_id).request_retrieval_admission));
    }
    return result;
}

common_speculative_controller::learning_state & common_speculative_controller::learning(llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    if (persistence_ != COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS || !state.process_namespace_available) {
        return state.request_learning;
    }
    return process_namespaces_.at(state.process_namespace).learning;
}

const common_speculative_controller::learning_state & common_speculative_controller::learning(llama_seq_id seq_id) const {
    const auto & state = sequence(seq_id);
    if (persistence_ != COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS || !state.process_namespace_available) {
        return state.request_learning;
    }
    return process_namespaces_.at(state.process_namespace).learning;
}

common_speculative_controller::sequence_state & common_speculative_controller::sequence(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) sequences_.size()) {
        throw std::out_of_range("speculative controller sequence ID is out of range");
    }
    return sequences_[seq_id];
}

const common_speculative_controller::sequence_state & common_speculative_controller::sequence(llama_seq_id seq_id) const {
    if (seq_id < 0 || seq_id >= (llama_seq_id) sequences_.size()) {
        throw std::out_of_range("speculative controller sequence ID is out of range");
    }
    return sequences_[seq_id];
}

const common_speculative_controller::producer_stats * common_speculative_controller::find_stats(
        uint64_t key,
        llama_seq_id seq_id) const {
    const auto & state = learning(seq_id);
    const auto it = state.stats.find(key);
    return it == state.stats.end() ? nullptr : &it->second;
}

common_speculative_controller::producer_stats & common_speculative_controller::get_stats(
        uint64_t key,
        llama_seq_id seq_id) {
    return learning(seq_id).stats[key];
}

double common_speculative_controller::score_prefix(
        const common_speculative_candidate & candidate,
        uint16_t length,
        llama_seq_id seq_id) const {
    double expected_advance = 1.0;
    double survival = 1.0;

    for (uint16_t i = 0; i < length; ++i) {
        survival *= conditional_acceptance(candidate, i, seq_id);
        expected_advance += survival;
    }

    const int64_t measured_proposal_us = candidate.metadata.cycle_proposal_time_us > 0
        ? candidate.metadata.cycle_proposal_time_us
        : candidate.metadata.proposal_time_us;
    double proposal_us = std::max<int64_t>(1, measured_proposal_us);
    double verification_us = verification_cost(candidate, length, seq_id);
    const producer_stats * global = find_stats(
            is_ngram(candidate) ? admission_producer_id(candidate) : candidate.producer_id,
            seq_id);
    if (global != nullptr) {
        if (candidate.metadata.cycle_proposal_time_us <= 0 && global->proposal_time_us > 0.0) {
            proposal_us = global->proposal_time_us;
        }
    }
    return expected_advance / std::max(1.0, proposal_us + verification_us);
}

double common_speculative_controller::prefix_confidence(
        const common_speculative_candidate & candidate,
        uint16_t position,
        llama_seq_id seq_id) const {
    double survival = 1.0;
    for (uint16_t i = 0; i <= position; ++i) {
        survival *= conditional_acceptance(candidate, i, seq_id);
    }
    return survival;
}

double common_speculative_controller::conditional_acceptance(
        const common_speculative_candidate & candidate,
        uint16_t position,
        llama_seq_id seq_id) const {
    const auto & state = sequence(seq_id);
    const uint64_t producer_key = is_ngram(candidate) ? admission_producer_id(candidate) : candidate.producer_id;
    const producer_stats * global = find_stats(producer_key, seq_id);
    const producer_stats * contextual = find_stats(context_key(candidate), seq_id);
    const auto request_contextual_it = state.request_stats.find(context_key(candidate));
    const auto request_global_it = state.request_stats.find(producer_key);
    const producer_stats * request_contextual = request_contextual_it == state.request_stats.end() ? nullptr : &request_contextual_it->second;
    const producer_stats * request_global = request_global_it == state.request_stats.end() ? nullptr : &request_global_it->second;

    const auto confidence = [&](const producer_stats * stats, uint64_t minimum_observations) {
        if (stats == nullptr || stats->observations < minimum_observations || position >= stats->positions.size()) {
            return -1.0;
        }
        const auto & value = stats->positions[position];
        const double accepted = std::max(0.0, value.accepted - 1.0);
        const double rejected = std::max(0.0, value.rejected - 1.0);
        return accepted + rejected > 0.0 ? accepted / (accepted + rejected) : -1.0;
    };

    double value = confidence(request_contextual, 2);
    if (value >= 0.0) return value;
    value = confidence(request_global, 2);
    if (value >= 0.0) return value;
    value = confidence(contextual, 1);
    if (value >= 0.0) return value;
    value = confidence(global, 1);
    if (value >= 0.0) return value;
    if (is_retrieval(candidate)) {
        const size_t begin = ngram_span_begin(candidate);
        if (position >= begin) {
            const size_t retrieval_position = position - begin;
            if (retrieval_position < candidate.metadata.retrieval_dominance_permille.size()) {
                const double dominance = candidate.metadata.retrieval_dominance_permille[retrieval_position] / 1000.0;
                const double match = std::min(1.0, candidate.metadata.retrieval_match_length[retrieval_position] / 24.0);
                const double support = std::min(1.0, candidate.metadata.retrieval_source_support[retrieval_position] / 4.0);
                const double mtp = candidate.metadata.retrieval_mtp_agreement[retrieval_position] ? 0.10 : 0.0;
                if (candidate.metadata.retrieval_match_length[retrieval_position] >= 32 &&
                        (candidate.metadata.retrieval_source_support[retrieval_position] >= 2 ||
                            candidate.metadata.retrieval_match_length[retrieval_position] >= 48) &&
                        candidate.metadata.retrieval_dominance_permille[retrieval_position] >= 950) {
                    // Long unambiguous matches use a stronger survival prior.
                    return candidate.metadata.retrieval_mtp_agreement[retrieval_position] ? 0.99 : 0.985;
                }
                return std::clamp(0.30 + 0.30 * dominance + 0.15 * match + 0.10 * support + mtp, 0.35, 0.95);
            }
        }
    }
    return 0.5;
}

double common_speculative_controller::verification_cost(
        const common_speculative_candidate & candidate,
        uint16_t length,
        llama_seq_id seq_id) const {
    const auto & state = learning(seq_id);
    const auto contextual = state.verification_context_costs.find(verification_shape_key(candidate));
    const auto & costs = contextual != state.verification_context_costs.end() && !contextual->second.empty()
        ? contextual->second
        : state.verification_costs;
    if (costs.empty()) {
        return 1.0 + length;
    }

    auto upper = costs.lower_bound(length);
    if (upper == costs.begin()) {
        return upper->second.time_us;
    }
    if (upper == costs.end()) {
        return std::prev(upper)->second.time_us;
    }
    if (upper->first == length) {
        return upper->second.time_us;
    }

    const auto lower = std::prev(upper);
    const double position = (double) (length - lower->first) / (upper->first - lower->first);
    return lower->second.time_us + position * (upper->second.time_us - lower->second.time_us);
}

common_speculative_selection common_speculative_controller::select(
        const std::vector<common_speculative_candidate> & candidates,
        llama_seq_id seq_id) {
    common_speculative_selection result;
    auto & state = sequence(seq_id);
    const auto & learned = learning(seq_id);
    state.pending_hot = {};
    expire_hot_regions(seq_id);

    for (auto & item : state.challenger_cooldowns) {
        if (state.request_observations >= item.second.next_observation) {
            item.second.next_observation = 0;
        }
    }

    if (candidates.empty()) {
        return result;
    }

    if (mode_ == COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE) {
        register_shadow_probes(candidates, seq_id);
        state.shadow_discovery_pending = false;
    }

    if (mode_ == COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE &&
            !state.challenger_started && allow_challengers(seq_id)) {
        const bool has_ngram_candidate = std::any_of(candidates.begin(), candidates.end(), [](const auto & candidate) {
            return candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
        });

        if (state.request_observations < warmup_) {
            state.next_challenger_observation = warmup_;
        } else if (has_ngram_candidate) {
            state.empty_discovery_streak = 0;
            state.next_challenger_observation = state.request_observations + 8;
        } else {
            state.empty_discovery_streak = std::min<uint8_t>(state.empty_discovery_streak + 1, 3);
            const uint64_t interval = 8ull << (state.empty_discovery_streak - 1);
            state.next_challenger_observation = state.request_observations + interval;
        }
    }

    if (mode_ != COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE) {
        result.candidate_index = 0;
        result.prefix_length = max_verify_ > 0
            ? std::min<size_t>(candidates[0].tokens.size(), max_verify_)
            : candidates[0].tokens.size();
        result.prefix_confidence.assign(result.prefix_length, 0.5);
        return result;
    }

    if (state.request_observations < warmup_) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
                result.candidate_index = i;
                result.prefix_length = candidates[i].tokens.size();
                result.prefix_confidence.assign(result.prefix_length, 0.5);
                return result;
            }
        }
        result.candidate_index = 0;
        result.prefix_length = candidates[0].tokens.size();
        result.prefix_confidence.assign(result.prefix_length, 0.5);
        return result;
    }

    double best = -std::numeric_limits<double>::infinity();
    int32_t mtp_index = -1;
    double mtp_utility = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & candidate = candidates[i];
        admission_decision admission;
        if (candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            const auto cooldown = state.challenger_cooldowns.find(cooldown_key(candidate));
            if (cooldown != state.challenger_cooldowns.end() &&
                    state.request_observations < cooldown->second.next_observation) {
                result.challenger_cooldown_remaining = std::max(
                        result.challenger_cooldown_remaining,
                        cooldown->second.next_observation - state.request_observations);
                result.challenger_failure_streak = std::max(
                        result.challenger_failure_streak,
                        cooldown->second.failure_streak);
                continue;
            }
            admission = evaluate_admission(candidate, candidates, seq_id);
            if (!admission.admit) {
                if (admission.evidence > 0.0 && admission.upper_confidence < 0.5) {
                    shadow_metrics_.rejected_confidence++;
                } else {
                    shadow_metrics_.rejected_no_evidence++;
                }
                continue;
            }
        }
        if (candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            mtp_index = i;
            mtp_utility = score_prefix(candidate, candidate.tokens.size(), seq_id);
        }
        uint16_t maximum_length = max_verify_ > 0
            ? std::min<size_t>(candidate.tokens.size(), max_verify_)
            : candidate.tokens.size();
        if (candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP && admission.verification_cap > 0) {
            maximum_length = std::min<uint16_t>(maximum_length, admission.verification_cap);
        }
        std::vector<uint16_t> lengths;
        if (is_retrieval(candidate)) {
            const size_t evidence_length = std::min({
                candidate.metadata.retrieval_dominance_permille.size(),
                candidate.metadata.retrieval_source_support.size(),
                candidate.metadata.retrieval_source_distance.size(),
                (size_t) maximum_length,
            });
            for (size_t position = 1; position < evidence_length; ++position) {
                const bool ambiguous = candidate.metadata.retrieval_dominance_permille[position] < 600;
                const bool support_collapse = position >= 3 &&
                    candidate.metadata.retrieval_source_support[position] <
                        candidate.metadata.retrieval_source_support[position - 1];
                const bool source_jump = position >= 3 &&
                    candidate.metadata.retrieval_source_distance[position] >
                        candidate.metadata.retrieval_source_distance[position - 1] + 32;
                if (ambiguous || (support_collapse && source_jump)) {
                    maximum_length = std::min<uint16_t>(maximum_length, position);
                    break;
                }
            }
            if (maximum_length < 3) {
                continue;
            }
            static const uint16_t rungs[] = { 3, 8, 12, 16, 24, 32, 48 };
            for (const uint16_t rung : rungs) {
                if (rung <= maximum_length && (admission.verification_cap == 0 || rung <= admission.verification_cap)) {
                    lengths.push_back(rung);
                }
            }
            if (lengths.empty() && maximum_length > 0) {
                lengths.push_back(maximum_length);
            }
        } else {
            const uint16_t minimum_length = candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || admission.level == 1
                ? maximum_length
                : 1;
            for (uint16_t length = minimum_length; length <= maximum_length; ++length) {
                lengths.push_back(length);
            }
        }
        for (const uint16_t length : lengths) {
            const double utility = score_prefix(candidate, length, seq_id);
            if (utility > best) {
                best = utility;
                result.candidate_index = i;
                result.prefix_length = length;
                result.utility = utility;
            }
        }
    }

    if (mtp_index >= 0 &&
            (result.candidate_index != mtp_index || result.prefix_length != candidates[mtp_index].tokens.size()) &&
            result.utility < mtp_utility * (1.0 + safety_margin_)) {
        result.candidate_index = mtp_index;
        result.prefix_length = candidates[mtp_index].tokens.size();
        result.utility = mtp_utility;
    }

    if (learned.ordinary_decode_time_us > 0.0) {
        const double ordinary_utility = 1.0 / learned.ordinary_decode_time_us;
        if (result.utility < ordinary_utility * (1.0 + safety_margin_)) {
            return {};
        }
    }

    if (result.candidate_index >= 0 &&
            candidates[result.candidate_index].type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
        const auto & selected = candidates[result.candidate_index];
        const auto admission = evaluate_admission(selected, candidates, seq_id);
        result.admission_lower_confidence = admission.lower_confidence;
        result.admission_upper_confidence = admission.upper_confidence;
        result.admission_evidence = admission.evidence;
        result.admission_bootstrap = admission.bootstrap;
        result.admission_level = admission.level;
        if (admission.verification_cap > 0) {
            result.prefix_length = std::min<uint16_t>(result.prefix_length, admission.verification_cap);
        }
        state.challenger_started = true;
        if (admission.level == 1) {
            shadow_metrics_.provisional_decisions++;
        } else {
            shadow_metrics_.trusted_decisions++;
        }

        if (admission.level == 2 && !is_retrieval(selected)) {
            const uint64_t key = hot_key(selected);
            result.hot_key = key;
            const auto hot = state.hot_regions.find(key);
            if (hot == state.hot_regions.end()) {
                if (result.prefix_length >= 16 && selected.tokens.size() >= 16) {
                    state.pending_hot = { selected.candidate_id, key, admission_producer_id(selected), 16, false };
                    result.hot_requested_length = 16;
                }
            } else {
                hot->second.last_selected_observation = state.request_observations;
                result.hot_rung = hot->second.rung;
                result.hot_transition_reason = hot->second.last_transition;
                uint16_t available = max_verify_ > 0
                    ? std::min<size_t>(selected.tokens.size(), max_verify_)
                    : selected.tokens.size();
                if (available >= hot->second.rung && agrees_with_complete_mtp(selected, candidates)) {
                    state.pending_hot = { selected.candidate_id, key, admission_producer_id(selected), hot->second.rung, true };
                    result.hot_requested_length = hot->second.rung;
                    const uint16_t normal_length = result.prefix_length;
                    result.prefix_length = std::max(result.prefix_length, hot->second.rung);
                    result.hot_applied = result.prefix_length > normal_length;
                }
            }
        }
    } else if (std::any_of(candidates.begin(), candidates.end(), is_ngram)) {
        shadow_metrics_.blocked_decisions++;
    }

    if (result.candidate_index >= 0) {
        const auto & selected = candidates[result.candidate_index];
        result.prefix_confidence.reserve(result.prefix_length);
        for (uint16_t position = 0; position < result.prefix_length; ++position) {
            result.prefix_confidence.push_back(prefix_confidence(selected, position, seq_id));
        }
    }

    return result;
}

void common_speculative_controller::observe(
        const std::vector<common_speculative_candidate> & candidates,
        uint64_t selected_candidate_id,
        const llama_tokens & realized,
        int64_t verification_time_us,
        int64_t rollback_or_replay_time_us,
        llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    auto & learned = learning(seq_id);
    state.request_observations++;

    size_t mtp_length = 0;
    for (const auto & candidate : candidates) {
        if (candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            mtp_length = std::max(mtp_length, candidate.tokens.size());
        }
    }

    for (const auto & candidate : candidates) {
        const size_t n_observed = std::min(candidate.tokens.size(), realized.size());
        size_t n_accepted = 0;
        while (n_accepted < n_observed && candidate.tokens[n_accepted] == realized[n_accepted]) {
            n_accepted++;
        }

        const bool mismatch_observed = n_accepted < n_observed;
        if (candidate.candidate_id == selected_candidate_id &&
                candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            const uint64_t key = cooldown_key(candidate);
            if (candidate.tokens.size() >= 8 && n_accepted <= mtp_length) {
                auto & failure = state.challenger_cooldowns[key];
                failure.failure_streak = std::min<uint8_t>(failure.failure_streak + 1, 3);
                const uint64_t cooldown = 32ull << (failure.failure_streak - 1);
                failure.next_observation = state.request_observations + cooldown;
            } else if (n_accepted > mtp_length) {
                state.challenger_cooldowns.erase(key);
            }

            if (is_retrieval(candidate)) {
                const size_t begin = std::min<size_t>(ngram_span_begin(candidate), candidate.tokens.size());
                if (n_accepted >= begin) {
                    const size_t proposed = candidate.tokens.size() - begin;
                    const size_t accepted = std::min(n_accepted, candidate.tokens.size()) - begin;
                    const bool retrieval_mismatch = mismatch_observed && n_accepted >= begin;
                    const bool success_8 = proposed >= 8 && accepted >= 8;
                    const bool failure_8 = proposed >= 8 && retrieval_mismatch && accepted < 8;
                    const bool success_16 = proposed >= 16 && accepted >= 16;
                    const bool failure_16 = proposed >= 16 && retrieval_mismatch && accepted < 16;
                    const auto update_admission = [&](admission_stats & admission) {
                        if (success_8 || failure_8) {
                            admission.survived_8 = 1.0 + decay_ * (admission.survived_8 - 1.0) + (success_8 ? 1.0 : 0.0);
                            admission.failed_8 = 1.0 + decay_ * (admission.failed_8 - 1.0) + (failure_8 ? 1.0 : 0.0);
                            admission.outcomes_8++;
                        }
                        if (success_16 || failure_16) {
                            admission.survived_16 = 1.0 + decay_ * (admission.survived_16 - 1.0) + (success_16 ? 1.0 : 0.0);
                            admission.failed_16 = 1.0 + decay_ * (admission.failed_16 - 1.0) + (failure_16 ? 1.0 : 0.0);
                            admission.outcomes_16++;
                        }
                    };
                    if (success_8 || failure_8 || success_16 || failure_16) {
                        auto & request = state.request_learning.admissions[admission_key(candidate)];
                        update_admission(request);
                        update_admission(state.request_retrieval_admission);
                        auto & persistent = learning(seq_id).admissions[admission_key(candidate)];
                        if (&persistent != &request) {
                            update_admission(persistent);
                        }
                    }
                }
            }

            const auto pending = state.pending_hot;
            state.pending_hot = {};
            if (pending.candidate_id == candidate.candidate_id && candidate.tokens.size() >= pending.target_rung) {
                const uint16_t span_begin = std::min<uint16_t>(ngram_span_begin(candidate), candidate.tokens.size());
                const uint16_t proposed = candidate.tokens.size() - span_begin;
                const uint16_t accepted = n_accepted > span_begin
                    ? std::min<size_t>(n_accepted, candidate.tokens.size()) - span_begin
                    : 0;
                const bool fully_observed = mismatch_observed || n_observed >= candidate.tokens.size();

                if (pending.active) {
                    hot_metrics_.selected_tokens += proposed;
                    hot_metrics_.accepted_tokens += accepted;
                }

                if (fully_observed && proposed > 0) {
                    auto hot = state.hot_regions.find(pending.hot_key);
                    if (!pending.active) {
                        if (n_accepted == candidate.tokens.size()) {
                            const uint16_t next = next_hot_rung(16);
                            state.hot_regions[pending.hot_key] = { pending.producer_id, next, state.request_observations, COMMON_SPECULATIVE_HOT_ENTRY };
                            state.hot_events.push_back({ COMMON_SPECULATIVE_HOT_ENTRY, pending.hot_key, pending.producer_id, 16, next, proposed, accepted });
                            hot_metrics_.entries++;
                            hot_metrics_.promotions_24++;
                            hot_metrics_.full_matches++;
                        }
                    } else if (hot != state.hot_regions.end()) {
                        const uint16_t previous = hot->second.rung;
                        hot->second.last_selected_observation = state.request_observations;
                        if (accepted == 0) {
                            hot->second.rung = 16;
                            hot->second.last_transition = COMMON_SPECULATIVE_HOT_RESET;
                            state.hot_events.push_back({ COMMON_SPECULATIVE_HOT_RESET, pending.hot_key, pending.producer_id, previous, 16, proposed, accepted });
                            hot_metrics_.resets++;
                        } else if (n_accepted == candidate.tokens.size()) {
                            const uint16_t next = next_hot_rung(previous);
                            hot->second.rung = next;
                            hot_metrics_.full_matches++;
                            if (next != previous) {
                                hot->second.last_transition = COMMON_SPECULATIVE_HOT_PROMOTION;
                                state.hot_events.push_back({ COMMON_SPECULATIVE_HOT_PROMOTION, pending.hot_key, pending.producer_id, previous, next, proposed, accepted });
                                if (next == 24) hot_metrics_.promotions_24++;
                                if (next == 32) hot_metrics_.promotions_32++;
                                if (next == 48) hot_metrics_.promotions_48++;
                            } else {
                                hot->second.last_transition = COMMON_SPECULATIVE_HOT_RETENTION;
                                state.hot_events.push_back({ COMMON_SPECULATIVE_HOT_RETENTION, pending.hot_key, pending.producer_id, previous, previous, proposed, accepted });
                                hot_metrics_.retained++;
                            }
                        } else if ((uint32_t) accepted * 4 >= (uint32_t) proposed * 3) {
                            hot->second.last_transition = COMMON_SPECULATIVE_HOT_RETENTION;
                            state.hot_events.push_back({ COMMON_SPECULATIVE_HOT_RETENTION, pending.hot_key, pending.producer_id, previous, previous, proposed, accepted });
                            hot_metrics_.retained++;
                        } else {
                            hot->second.rung = previous_hot_rung(previous);
                            hot->second.last_transition = COMMON_SPECULATIVE_HOT_ROLLBACK;
                            state.hot_events.push_back({ COMMON_SPECULATIVE_HOT_ROLLBACK, pending.hot_key, pending.producer_id, previous, hot->second.rung, proposed, accepted });
                            hot_metrics_.rollbacks++;
                        }
                    }
                }
            }
        }
        auto update_acceptance = [&](producer_stats & stats, size_t begin) {
            if (n_accepted < begin) {
                return;
            }
            stats.observations++;
            const size_t n_positions = n_accepted + (mismatch_observed ? 1 : 0);
            if (stats.positions.size() < n_positions) {
                stats.positions.resize(n_positions);
            }

            for (size_t i = begin; i < n_accepted; ++i) {
                auto & pos = stats.positions[i];
                pos.accepted = 1.0 + decay_ * (pos.accepted - 1.0) + 1.0;
                pos.rejected = 1.0 + decay_ * (pos.rejected - 1.0);
            }
            if (mismatch_observed && n_accepted >= begin) {
                auto & pos = stats.positions[n_accepted];
                pos.accepted = 1.0 + decay_ * (pos.accepted - 1.0);
                pos.rejected = 1.0 + decay_ * (pos.rejected - 1.0) + 1.0;
            }
        };

        auto & global = get_stats(candidate.producer_id, seq_id);
        if (candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            update_acceptance(global, 0);
            update_acceptance(get_stats(context_key(candidate), seq_id), 0);
            update_acceptance(state.request_stats[candidate.producer_id], 0);
            update_acceptance(state.request_stats[context_key(candidate)], 0);
        } else if (candidate.candidate_id == selected_candidate_id && (is_retrieval(candidate) || n_observed > 16)) {
            const uint64_t producer_key = admission_producer_id(candidate);
            const size_t begin = is_retrieval(candidate) ? ngram_span_begin(candidate) : 16;
            update_acceptance(get_stats(producer_key, seq_id), begin);
            update_acceptance(get_stats(context_key(candidate), seq_id), begin);
            update_acceptance(state.request_stats[producer_key], begin);
            update_acceptance(state.request_stats[context_key(candidate)], begin);
        }

        if (candidate.metadata.proposal_time_us > 0) {
            global.proposal_time_us = global.proposal_time_us == 0.0
                ? candidate.metadata.proposal_time_us
                : decay_ * global.proposal_time_us + (1.0 - decay_) * candidate.metadata.proposal_time_us;
        }

        if (candidate.candidate_id == selected_candidate_id && verification_time_us > 0) {
            const double total_us = verification_time_us + std::max<int64_t>(0, rollback_or_replay_time_us);
            auto & cost = learned.verification_costs[(uint16_t) candidate.tokens.size()];
            cost.time_us = cost.observations == 0
                ? total_us
                : decay_ * cost.time_us + (1.0 - decay_) * total_us;
            cost.observations++;
            auto & contextual_cost = learned.verification_context_costs[verification_shape_key(candidate)][(uint16_t) candidate.tokens.size()];
            contextual_cost.time_us = contextual_cost.observations == 0
                ? total_us
                : decay_ * contextual_cost.time_us + (1.0 - decay_) * total_us;
            contextual_cost.observations++;
        }
    }
}

void common_speculative_controller::observe_ordinary(int64_t decode_time_us, llama_seq_id seq_id) {
    if (decode_time_us <= 0) {
        return;
    }

    auto & ordinary_decode_time_us = learning(seq_id).ordinary_decode_time_us;
    ordinary_decode_time_us = ordinary_decode_time_us == 0.0
        ? decode_time_us
        : decay_ * ordinary_decode_time_us + (1.0 - decay_) * decode_time_us;
}

void common_speculative_controller::register_shadow_probes(
        const std::vector<common_speculative_candidate> & candidates,
        llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    std::vector<shadow_probe> pending;

    for (const auto & candidate : candidates) {
        if (!is_ngram(candidate) || candidate.tokens.size() < 8) {
            continue;
        }

        llama_tokens tokens(candidate.tokens.begin(), candidate.tokens.begin() + std::min<size_t>(16, candidate.tokens.size()));
        auto group = std::find_if(pending.begin(), pending.end(), [&](const auto & probe) {
            return probe.tokens == tokens;
        });
        if (group == pending.end()) {
            shadow_probe probe;
            probe.probe_id = state.next_probe_id++;
            probe.candidate_id = candidate.candidate_id;
            probe.tokens = std::move(tokens);
            probe.mtp_agreement = agreement_with_mtp(candidate, candidates);
            probe.original_length = candidate.tokens.size();
            pending.push_back(std::move(probe));
            group = std::prev(pending.end());
        }

        shadow_contributor contributor;
        contributor.candidate = candidate;
        contributor.candidate.tokens.resize(std::min<size_t>(16, contributor.candidate.tokens.size()));
        contributor.admission_key = admission_key(candidate);
        contributor.cooldown_key = cooldown_key(candidate);
        contributor.context_key = context_key(candidate);
        group->contributors.push_back(std::move(contributor));
        std::set<common_speculative_type> families;
        for (const auto & item : group->contributors) {
            families.insert(item.candidate.type);
        }
        group->family_support = families.size();
    }

    for (auto & probe : pending) {
        state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_STARTED, probe.probe_id, probe.candidate_id, 0 });
        shadow_metrics_.started++;
        state.shadow_probes.push_back(std::move(probe));
    }

    const auto better = [](const shadow_probe & a, const shadow_probe & b) {
        if (a.mtp_agreement != b.mtp_agreement) return a.mtp_agreement > b.mtp_agreement;
        if (a.family_support != b.family_support) return a.family_support > b.family_support;
        if (a.original_length != b.original_length) return a.original_length > b.original_length;
        return a.candidate_id < b.candidate_id;
    };
    std::stable_sort(state.shadow_probes.begin(), state.shadow_probes.end(), better);
    while (state.shadow_probes.size() > 64) {
        const auto & probe = state.shadow_probes.back();
        state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_DROPPED, probe.probe_id, probe.candidate_id, probe.matched });
        shadow_metrics_.dropped++;
        state.shadow_probes.pop_back();
    }
}

void common_speculative_controller::update_shadow_acceptance(
        const shadow_contributor & contributor,
        size_t accepted,
        bool mismatch,
        llama_seq_id seq_id) {
    auto update = [&](producer_stats & stats) {
        stats.observations++;
        const size_t n_positions = accepted + (mismatch ? 1 : 0);
        if (stats.positions.size() < n_positions) {
            stats.positions.resize(n_positions);
        }
        for (size_t i = 0; i < accepted; ++i) {
            auto & position = stats.positions[i];
            position.accepted = 1.0 + decay_ * (position.accepted - 1.0) + 1.0;
            position.rejected = 1.0 + decay_ * (position.rejected - 1.0);
        }
        if (mismatch) {
            auto & position = stats.positions[accepted];
            position.accepted = 1.0 + decay_ * (position.accepted - 1.0);
            position.rejected = 1.0 + decay_ * (position.rejected - 1.0) + 1.0;
        }
    };

    auto & state = sequence(seq_id);
    const uint64_t producer_key = admission_producer_id(contributor.candidate);
    update(get_stats(producer_key, seq_id));
    update(get_stats(contributor.context_key, seq_id));
    update(state.request_stats[producer_key]);
    update(state.request_stats[contributor.context_key]);
}

void common_speculative_controller::update_shadow_admission(
        const shadow_contributor & contributor,
        bool success_8,
        bool failure_8,
        bool success_16,
        bool failure_16,
        llama_seq_id seq_id) {
    const auto update = [&](admission_stats & admission) {
        if (success_8 || failure_8) {
            admission.survived_8 = 1.0 + decay_ * (admission.survived_8 - 1.0) + (success_8 ? 1.0 : 0.0);
            admission.failed_8 = 1.0 + decay_ * (admission.failed_8 - 1.0) + (failure_8 ? 1.0 : 0.0);
            admission.outcomes_8++;
        }
        if (success_16 || failure_16) {
            admission.survived_16 = 1.0 + decay_ * (admission.survived_16 - 1.0) + (success_16 ? 1.0 : 0.0);
            admission.failed_16 = 1.0 + decay_ * (admission.failed_16 - 1.0) + (failure_16 ? 1.0 : 0.0);
            admission.outcomes_16++;
        }
    };

    auto & request = sequence(seq_id).request_learning.admissions[contributor.admission_key];
    update(request);
    if (is_retrieval(contributor.candidate)) {
        update(sequence(seq_id).request_retrieval_admission);
    }
    auto & persistent = learning(seq_id).admissions[contributor.admission_key];
    if (&persistent != &request) {
        update(persistent);
    }
}

void common_speculative_controller::observe_output_token(llama_token token, llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    for (size_t i = 0; i < state.shadow_probes.size();) {
        auto & probe = state.shadow_probes[i];
        const bool matches = probe.matched < probe.tokens.size() && probe.tokens[probe.matched] == token;
        if (!matches) {
            const bool failure_8 = probe.matched < 8;
            const bool failure_16 = probe.tokens.size() >= 16 && probe.matched < 16;
            std::set<uint64_t> admission_keys;
            for (const auto & contributor : probe.contributors) {
                update_shadow_acceptance(contributor, probe.matched, true, seq_id);
                if (admission_keys.insert(contributor.admission_key).second) {
                    update_shadow_admission(contributor, false, failure_8, false, failure_16, seq_id);
                }
            }
            state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_FAILED, probe.probe_id, probe.candidate_id, probe.matched });
            shadow_metrics_.failed++;
            state.shadow_probes.erase(state.shadow_probes.begin() + i);
            continue;
        }

        probe.matched++;
        if (probe.matched == 8) {
            std::set<uint64_t> admission_keys;
            for (const auto & contributor : probe.contributors) {
                if (admission_keys.insert(contributor.admission_key).second) {
                    update_shadow_admission(contributor, true, false, false, false, seq_id);
                }
            }
            probe.recorded_8 = true;
            state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_SUCCEEDED_8, probe.probe_id, probe.candidate_id, probe.matched });
            shadow_metrics_.succeeded_8++;
            for (const auto & contributor : probe.contributors) {
                state.challenger_cooldowns.erase(contributor.cooldown_key);
            }
            state.next_challenger_observation = state.request_observations;
            state.shadow_discovery_pending = true;
        }
        if (probe.matched == probe.tokens.size()) {
            const bool success_16 = probe.tokens.size() == 16;
            std::set<uint64_t> admission_keys;
            for (const auto & contributor : probe.contributors) {
                update_shadow_acceptance(contributor, probe.matched, false, seq_id);
                if (admission_keys.insert(contributor.admission_key).second) {
                    update_shadow_admission(contributor, false, false, success_16, false, seq_id);
                }
            }
            if (success_16) {
                state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_SUCCEEDED_16, probe.probe_id, probe.candidate_id, probe.matched });
                shadow_metrics_.succeeded_16++;
            }
            state.shadow_probes.erase(state.shadow_probes.begin() + i);
            continue;
        }
        i++;
    }
}

void common_speculative_controller::exclude_selected_shadow(
        const common_speculative_candidate & candidate,
        llama_seq_id seq_id) {
    auto & state = sequence(seq_id);
    const uint64_t selected_key = admission_key(candidate);
    for (size_t i = 0; i < state.shadow_probes.size();) {
        auto & probe = state.shadow_probes[i];
        probe.contributors.erase(
                std::remove_if(probe.contributors.begin(), probe.contributors.end(), [&](const auto & contributor) {
                    return contributor.admission_key == selected_key;
                }),
                probe.contributors.end());
        if (probe.contributors.empty()) {
            state.shadow_events.push_back({ COMMON_SPECULATIVE_SHADOW_DROPPED, probe.probe_id, probe.candidate_id, probe.matched });
            shadow_metrics_.dropped++;
            state.shadow_probes.erase(state.shadow_probes.begin() + i);
            continue;
        }
        probe.candidate_id = probe.contributors.front().candidate.candidate_id;
        i++;
    }
}

std::vector<common_speculative_shadow_event> common_speculative_controller::take_shadow_events(llama_seq_id seq_id) {
    auto & events = sequence(seq_id).shadow_events;
    auto result = std::move(events);
    events.clear();
    return result;
}

std::vector<common_speculative_hot_event> common_speculative_controller::take_hot_events(llama_seq_id seq_id) {
    auto & events = sequence(seq_id).hot_events;
    auto result = std::move(events);
    events.clear();
    return result;
}

const common_speculative_shadow_metrics & common_speculative_controller::shadow_metrics() const {
    return shadow_metrics_;
}

const common_speculative_hot_metrics & common_speculative_controller::hot_metrics() const {
    return hot_metrics_;
}

size_t common_speculative_controller::active_hot_entries() const {
    size_t result = 0;
    for (const auto & state : sequences_) {
        result += state.hot_regions.size();
    }
    return result;
}
