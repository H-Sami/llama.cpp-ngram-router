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
    sequence(seq_id).request_active = false;
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
    if (state.challenger_failed) {
        return state.request_observations >= state.next_challenger_observation;
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

uint64_t common_speculative_controller::admission_key(const common_speculative_candidate & candidate) {
    const uint32_t context_bucket = common_speculative_controller::context_bucket(candidate);

    constexpr uint16_t horizon = 16;
    size_t supporters = 0;
    uint16_t longest = 0;
    for (const uint16_t agreement : candidate.metadata.agreement_prefix) {
        supporters += agreement >= horizon;
        longest = std::max(longest, agreement);
    }

    const size_t peers = candidate.metadata.agreement_prefix.empty() ? 0 : candidate.metadata.agreement_prefix.size() - 1;
    const uint32_t support_bucket = peers == 0 ? 0 : std::min<uint32_t>(4, supporters * 4 / peers);
    const uint32_t agreement_bucket = longest >= 32 ? 5 : longest >= 16 ? 4 : longest >= 8 ? 3 : longest >= 4 ? 2 : longest > 0 ? 1 : 0;
    const uint32_t length_bucket = candidate.tokens.size() >= 32 ? 2 : candidate.tokens.size() >= 16 ? 1 : 0;

    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    mix(candidate.producer_id);
    mix(context_bucket);
    mix(support_bucket);
    mix(agreement_bucket);
    mix(length_bucket);
    return key;
}

bool common_speculative_controller::has_bootstrap_consensus(const common_speculative_candidate & candidate) {
    constexpr uint16_t consensus_prefix = 32;
    constexpr size_t minimum_supporters = 8;

    if (candidate.tokens.size() < consensus_prefix || candidate.metadata.agreement_prefix.size() < 2) {
        return false;
    }

    size_t supporters = 0;
    const size_t peers = candidate.metadata.agreement_prefix.size() - 1;
    for (const uint16_t agreement : candidate.metadata.agreement_prefix) {
        supporters += agreement >= consensus_prefix;
    }

    return supporters >= minimum_supporters && supporters * 5 >= peers * 4;
}

common_speculative_controller::admission_decision common_speculative_controller::evaluate_admission(
        const common_speculative_candidate & candidate,
        llama_seq_id seq_id) const {
    admission_decision result;
    result.bootstrap = has_bootstrap_consensus(candidate);
    result.admit = result.bootstrap;

    const auto & state = learning(seq_id);
    const auto it = state.admissions.find(admission_key(candidate));
    if (it == state.admissions.end()) {
        return result;
    }

    const double survived = std::max(1.0, it->second.survived);
    const double failed = std::max(1.0, it->second.failed);
    const double total = survived + failed;
    result.evidence = std::max(0.0, total - 2.0);
    const double mean = survived / total;
    const double deviation = 1.96 * std::sqrt((survived * failed) / (total * total * (total + 1.0)));
    result.lower_confidence = std::max(0.0, mean - deviation);
    result.upper_confidence = std::min(1.0, mean + deviation);

    if (result.evidence < 5.5) {
        return result;
    }
    if (result.upper_confidence < 0.5) {
        result.admit = false;
    } else if (candidate.tokens.size() >= 8 && result.lower_confidence >= 0.6) {
        result.admit = true;
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
    const auto & state = sequence(seq_id);
    const producer_stats * global = find_stats(candidate.producer_id, seq_id);
    const producer_stats * contextual = find_stats(context_key(candidate), seq_id);
    const auto request_contextual_it = state.request_stats.find(context_key(candidate));
    const auto request_global_it = state.request_stats.find(candidate.producer_id);
    const producer_stats * request_contextual = request_contextual_it == state.request_stats.end() ? nullptr : &request_contextual_it->second;
    const producer_stats * request_global = request_global_it == state.request_stats.end() ? nullptr : &request_global_it->second;
    const producer_stats * acceptance = request_contextual != nullptr && request_contextual->observations >= 2
        ? request_contextual
        : request_global != nullptr && request_global->observations >= 2
            ? request_global
            : contextual != nullptr && contextual->observations > 0 ? contextual : global;

    double expected_advance = 1.0;
    double survival = 1.0;

    for (uint16_t i = 0; i < length; ++i) {
        double conditional = 0.5;
        if (acceptance != nullptr && i < acceptance->positions.size()) {
            const auto & pos = acceptance->positions[i];
            const double accepted = std::max(0.0, pos.accepted - 1.0);
            const double rejected = std::max(0.0, pos.rejected - 1.0);
            if (accepted + rejected > 0.0) {
                conditional = accepted / (accepted + rejected);
            }
        }
        survival *= conditional;
        expected_advance += survival;
    }

    const int64_t measured_proposal_us = candidate.metadata.cycle_proposal_time_us > 0
        ? candidate.metadata.cycle_proposal_time_us
        : candidate.metadata.proposal_time_us;
    double proposal_us = std::max<int64_t>(1, measured_proposal_us);
    double verification_us = verification_cost(candidate, length, seq_id);
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
    const auto & state = sequence(seq_id);
    const producer_stats * global = find_stats(candidate.producer_id, seq_id);
    const producer_stats * contextual = find_stats(context_key(candidate), seq_id);
    const auto request_contextual_it = state.request_stats.find(context_key(candidate));
    const auto request_global_it = state.request_stats.find(candidate.producer_id);
    const producer_stats * request_contextual = request_contextual_it == state.request_stats.end() ? nullptr : &request_contextual_it->second;
    const producer_stats * request_global = request_global_it == state.request_stats.end() ? nullptr : &request_global_it->second;
    const producer_stats * acceptance = request_contextual != nullptr && request_contextual->observations >= 2
        ? request_contextual
        : request_global != nullptr && request_global->observations >= 2
            ? request_global
            : contextual != nullptr && contextual->observations > 0 ? contextual : global;

    double survival = 1.0;
    for (uint16_t i = 0; i <= position; ++i) {
        double conditional = 0.5;
        if (acceptance != nullptr && i < acceptance->positions.size()) {
            const auto & pos = acceptance->positions[i];
            const double accepted = std::max(0.0, pos.accepted - 1.0);
            const double rejected = std::max(0.0, pos.rejected - 1.0);
            if (accepted + rejected > 0.0) {
                conditional = accepted / (accepted + rejected);
            }
        }
        survival *= conditional;
    }
    return survival;
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

    if (state.challenger_failed && state.request_observations >= state.next_challenger_observation) {
        state.challenger_failed = false;
    }
    result.challenger_cooldown_remaining = state.challenger_failed
        ? state.next_challenger_observation - state.request_observations
        : 0;
    result.challenger_failure_streak = state.challenger_failure_streak;

    if (candidates.empty()) {
        return result;
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

    if (state.challenger_failed) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
                result.candidate_index = i;
                result.prefix_length = candidates[i].tokens.size();
                result.utility = score_prefix(candidates[i], result.prefix_length, seq_id);
                for (uint16_t position = 0; position < result.prefix_length; ++position) {
                    result.prefix_confidence.push_back(prefix_confidence(candidates[i], position, seq_id));
                }
                return result;
            }
        }
    }

    double best = -std::numeric_limits<double>::infinity();
    int32_t mtp_index = -1;
    double mtp_utility = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & candidate = candidates[i];
        if (!state.challenger_started && candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP &&
                !evaluate_admission(candidate, seq_id).admit) {
            continue;
        }
        if (candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            mtp_index = i;
            mtp_utility = score_prefix(candidate, candidate.tokens.size(), seq_id);
        }
        const uint16_t maximum_length = max_verify_ > 0
            ? std::min<size_t>(candidate.tokens.size(), max_verify_)
            : candidate.tokens.size();
        const uint16_t minimum_length = candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP ? maximum_length : 1;
        for (uint16_t length = minimum_length; length <= maximum_length; ++length) {
            if (!learned.verification_costs.empty() &&
                    length != candidate.tokens.size() &&
                    learned.verification_costs.find(length) == learned.verification_costs.end()) {
                continue;
            }
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

    if (max_verify_ > 0 && !state.challenger_started) {
        double strongest_utility = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto & candidate = candidates[i];
            if (candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || !evaluate_admission(candidate, seq_id).admit) {
                continue;
            }
            const uint16_t length = std::min<size_t>(candidate.tokens.size(), max_verify_);
            const double utility = score_prefix(candidate, length, seq_id);
            if (utility > strongest_utility) {
                strongest_utility = utility;
                result.candidate_index = i;
                result.prefix_length = length;
                result.utility = utility;
            }
        }
    }

    if (result.candidate_index >= 0 &&
            candidates[result.candidate_index].type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP &&
            !state.challenger_started) {
        const auto admission = evaluate_admission(candidates[result.candidate_index], seq_id);
        result.admission_lower_confidence = admission.lower_confidence;
        result.admission_upper_confidence = admission.upper_confidence;
        result.admission_evidence = admission.evidence;
        result.admission_bootstrap = admission.bootstrap;
        result.prefix_length = max_verify_ > 0
            ? std::min<size_t>(candidates[result.candidate_index].tokens.size(), max_verify_)
            : candidates[result.candidate_index].tokens.size();
        state.challenger_started = true;
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
            if (candidate.tokens.size() >= 8 && n_accepted <= mtp_length) {
                state.challenger_failure_streak = std::min<uint8_t>(state.challenger_failure_streak + 1, 3);
                const uint64_t cooldown = 32ull << (state.challenger_failure_streak - 1);
                state.challenger_failed = true;
                state.challenger_started = false;
                state.next_challenger_observation = state.request_observations + cooldown;
            } else if (n_accepted > mtp_length) {
                state.challenger_failure_streak = 0;
            }
        }
        auto update_acceptance = [&](producer_stats & stats) {
            stats.observations++;
            const size_t n_positions = n_accepted + (mismatch_observed ? 1 : 0);
            if (stats.positions.size() < n_positions) {
                stats.positions.resize(n_positions);
            }

            for (size_t i = 0; i < n_accepted; ++i) {
                auto & pos = stats.positions[i];
                pos.accepted = 1.0 + decay_ * (pos.accepted - 1.0) + 1.0;
                pos.rejected = 1.0 + decay_ * (pos.rejected - 1.0);
            }
            if (mismatch_observed) {
                auto & pos = stats.positions[n_accepted];
                pos.accepted = 1.0 + decay_ * (pos.accepted - 1.0);
                pos.rejected = 1.0 + decay_ * (pos.rejected - 1.0) + 1.0;
            }
        };

        auto & global = get_stats(candidate.producer_id, seq_id);
        update_acceptance(global);
        update_acceptance(get_stats(context_key(candidate), seq_id));
        update_acceptance(state.request_stats[candidate.producer_id]);
        update_acceptance(state.request_stats[context_key(candidate)]);

        if (candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP && candidate.tokens.size() >= 8) {
            const size_t horizon = std::min<size_t>(16, candidate.tokens.size());
            const bool survived = n_accepted >= horizon;
            const bool failed = mismatch_observed && n_accepted < horizon;
            if (survived || failed) {
                auto & admission = learned.admissions[admission_key(candidate)];
                admission.survived = 1.0 + decay_ * (admission.survived - 1.0) + (survived ? 1.0 : 0.0);
                admission.failed = 1.0 + decay_ * (admission.failed - 1.0) + (failed ? 1.0 : 0.0);
            }
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
