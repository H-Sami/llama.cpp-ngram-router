#include "speculative-controller.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

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

common_speculative_controller::common_speculative_controller(
        common_speculative_controller_mode mode,
        float safety_margin,
        float decay,
        uint32_t warmup)
    : mode_(mode)
    , safety_margin_(std::max(0.0f, safety_margin))
    , decay_(std::clamp((double) decay, 0.0, 1.0))
    , warmup_(warmup) {
}

common_speculative_controller_mode common_speculative_controller::mode() const {
    return mode_;
}

void common_speculative_controller::begin_request() {
    request_observations_ = 0;
    request_stats_.clear();
    challenger_probed_ = false;
    challenger_failed_ = false;
}

bool common_speculative_controller::allow_challengers() const {
    return !challenger_failed_;
}

uint64_t common_speculative_controller::context_key(const common_speculative_candidate & candidate) {
    uint32_t context_bucket = 0;
    for (int32_t n = std::max(1, candidate.metadata.context_length); n > 1; n >>= 1) {
        context_bucket++;
    }

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

const common_speculative_controller::producer_stats * common_speculative_controller::find_stats(uint64_t key) const {
    const auto it = stats_.find(key);
    return it == stats_.end() ? nullptr : &it->second;
}

common_speculative_controller::producer_stats & common_speculative_controller::get_stats(uint64_t key) {
    return stats_[key];
}

double common_speculative_controller::score_prefix(
        const common_speculative_candidate & candidate,
        uint16_t length) const {
    const producer_stats * global = find_stats(candidate.producer_id);
    const producer_stats * contextual = find_stats(context_key(candidate));
    const auto request_contextual_it = request_stats_.find(context_key(candidate));
    const auto request_global_it = request_stats_.find(candidate.producer_id);
    const producer_stats * request_contextual = request_contextual_it == request_stats_.end() ? nullptr : &request_contextual_it->second;
    const producer_stats * request_global = request_global_it == request_stats_.end() ? nullptr : &request_global_it->second;
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
    double verification_us = verification_cost(length);
    if (global != nullptr) {
        if (candidate.metadata.cycle_proposal_time_us <= 0 && global->proposal_time_us > 0.0) {
            proposal_us = global->proposal_time_us;
        }
    }
    return expected_advance / std::max(1.0, proposal_us + verification_us);
}

double common_speculative_controller::verification_cost(uint16_t length) const {
    if (verification_costs_.empty()) {
        return 1.0 + length;
    }

    auto upper = verification_costs_.lower_bound(length);
    if (upper == verification_costs_.begin()) {
        return upper->second.time_us;
    }
    if (upper == verification_costs_.end()) {
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
        const std::vector<common_speculative_candidate> & candidates) {
    common_speculative_selection result;

    if (candidates.empty()) {
        return result;
    }

    if (mode_ != COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE) {
        result.candidate_index = 0;
        result.prefix_length = candidates[0].tokens.size();
        return result;
    }

    if (request_observations_ < warmup_) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
                result.candidate_index = i;
                result.prefix_length = candidates[i].tokens.size();
                return result;
            }
        }
        result.candidate_index = 0;
        result.prefix_length = candidates[0].tokens.size();
        return result;
    }

    if (challenger_failed_) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
                result.candidate_index = i;
                result.prefix_length = candidates[i].tokens.size();
                result.utility = score_prefix(candidates[i], result.prefix_length);
                return result;
            }
        }
    }

    if (!challenger_probed_) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto & candidate = candidates[i];
            const bool is_ngram = candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE ||
                candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K ||
                candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V ||
                candidate.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
            if (!is_ngram || candidate.provenance.size() > 1 || candidate.tokens.size() < 4) {
                continue;
            }

            const auto stats_it = request_stats_.find(candidate.producer_id);
            if (stats_it == request_stats_.end() || stats_it->second.positions.size() < 3) {
                continue;
            }

            bool reliable_prefix = true;
            for (size_t pos = 0; pos < 3; ++pos) {
                const auto & evidence = stats_it->second.positions[pos];
                const double accepted = std::max(0.0, evidence.accepted - 1.0);
                const double rejected = std::max(0.0, evidence.rejected - 1.0);
                if (accepted + rejected < 2.0 || accepted / (accepted + rejected) < 0.8) {
                    reliable_prefix = false;
                    break;
                }
            }
            if (reliable_prefix) {
                challenger_probed_ = true;
                result.candidate_index = i;
                result.prefix_length = candidate.tokens.size();
                return result;
            }
        }
    }

    double best = -std::numeric_limits<double>::infinity();
    int32_t mtp_index = -1;
    double mtp_utility = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto & candidate = candidates[i];
        if (candidate.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            mtp_index = i;
            mtp_utility = score_prefix(candidate, candidate.tokens.size());
        }
        for (uint16_t length = 1; length <= candidate.tokens.size(); ++length) {
            if (!verification_costs_.empty() &&
                    length != candidate.tokens.size() &&
                    verification_costs_.find(length) == verification_costs_.end()) {
                continue;
            }
            const double utility = score_prefix(candidate, length);
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

    if (ordinary_decode_time_us_ > 0.0) {
        const double ordinary_utility = 1.0 / ordinary_decode_time_us_;
        if (result.utility < ordinary_utility * (1.0 + safety_margin_)) {
            return {};
        }
    }

    return result;
}

void common_speculative_controller::observe(
        const std::vector<common_speculative_candidate> & candidates,
        uint64_t selected_candidate_id,
        const llama_tokens & realized,
        int64_t verification_time_us,
        int64_t rollback_or_replay_time_us) {
    observations_++;
    request_observations_++;

    for (const auto & candidate : candidates) {
        const size_t n_observed = std::min(candidate.tokens.size(), realized.size());
        size_t n_accepted = 0;
        while (n_accepted < n_observed && candidate.tokens[n_accepted] == realized[n_accepted]) {
            n_accepted++;
        }

        const bool mismatch_observed = n_accepted < n_observed;
        if (candidate.candidate_id == selected_candidate_id &&
                candidate.type != COMMON_SPECULATIVE_TYPE_DRAFT_MTP &&
                candidate.tokens.size() >= 8 &&
                n_accepted * 2 < candidate.tokens.size()) {
            challenger_failed_ = true;
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

        auto & global = get_stats(candidate.producer_id);
        update_acceptance(global);
        update_acceptance(get_stats(context_key(candidate)));
        update_acceptance(request_stats_[candidate.producer_id]);
        update_acceptance(request_stats_[context_key(candidate)]);

        if (candidate.metadata.proposal_time_us > 0) {
            global.proposal_time_us = global.proposal_time_us == 0.0
                ? candidate.metadata.proposal_time_us
                : decay_ * global.proposal_time_us + (1.0 - decay_) * candidate.metadata.proposal_time_us;
        }

        if (candidate.candidate_id == selected_candidate_id && verification_time_us > 0) {
            const double total_us = verification_time_us + std::max<int64_t>(0, rollback_or_replay_time_us);
            auto & cost = verification_costs_[(uint16_t) candidate.tokens.size()];
            cost.time_us = cost.observations == 0
                ? total_us
                : decay_ * cost.time_us + (1.0 - decay_) * total_us;
            cost.observations++;
        }
    }
}

void common_speculative_controller::observe_ordinary(int64_t decode_time_us) {
    if (decode_time_us <= 0) {
        return;
    }

    ordinary_decode_time_us_ = ordinary_decode_time_us_ == 0.0
        ? decode_time_us
        : decay_ * ordinary_decode_time_us_ + (1.0 - decay_) * decode_time_us;
}
