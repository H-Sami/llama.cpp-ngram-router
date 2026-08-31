#include "speculative-controller.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #value); return 1; } } while (0)

static llama_tokens make_tokens(size_t length, llama_token first = 100) {
    llama_tokens result;
    for (size_t i = 0; i < length; ++i) {
        result.push_back(first + i);
    }
    return result;
}

static common_speculative_candidate make_candidate(
        uint64_t id,
        uint32_t producer,
        common_speculative_type type,
        const llama_tokens & tokens,
        int32_t context_length = 4096,
        int64_t proposal_us = 1) {
    common_speculative_candidate result;
    result.candidate_id = id;
    result.producer_id = producer;
    result.configuration_id = producer;
    result.type = type;
    result.tokens = tokens;
    result.metadata.proposal_time_us = proposal_us;
    result.metadata.context_length = context_length;
    result.provenance.push_back({ producer, producer, 0, (uint16_t) tokens.size() });
    return result;
}

static std::vector<common_speculative_candidate> make_candidates(
        size_t length = 48,
        uint32_t producer = 1,
        int32_t context_length = 4096,
        size_t mtp_length = 3) {
    const auto tokens = make_tokens(length);
    return {
        make_candidate(1, producer, COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, tokens, context_length),
        make_candidate(2, 20, COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
                llama_tokens(tokens.begin(), tokens.begin() + std::min(mtp_length, tokens.size())), context_length, 50000),
    };
}

static std::vector<common_speculative_candidate> make_fused_candidates(
        size_t length = 48,
        uint32_t producer = 1,
        int32_t context_length = 4096,
        size_t mtp_length = 4) {
    auto candidates = make_candidates(length, producer, context_length, mtp_length);
    auto & fused = candidates[0];
    fused.producer_id = 0x80000000u | (20u << 16) | producer;
    fused.configuration_id = fused.producer_id;
    fused.provenance = {
        { 20, 20, 0, (uint16_t) mtp_length },
        { producer, producer, (uint16_t) mtp_length, (uint16_t) length },
    };
    return candidates;
}

static std::vector<common_speculative_candidate> make_retrieval_candidates(bool fused = false) {
    const auto tokens = make_tokens(48);
    auto retrieval = make_candidate(101, 31, COMMON_SPECULATIVE_TYPE_NGRAM_RETRIEVAL, tokens, 4096, 1);
    retrieval.metadata.retrieval_evidence_key = 0xff;
    retrieval.metadata.retrieval_verified_matches = 8;
    retrieval.metadata.retrieval_alternatives = 1;
    retrieval.metadata.retrieval_match_length.assign(48, 32);
    retrieval.metadata.retrieval_source_support.assign(48, 8);
    retrieval.metadata.retrieval_dominance_permille.assign(48, 1000);
    retrieval.metadata.retrieval_source_distance.assign(48, 16);
    retrieval.metadata.retrieval_mtp_agreement.assign(48, 1);
    auto mtp = make_candidate(102, 20, COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
            llama_tokens(tokens.begin(), tokens.begin() + 3), 4096, 50000);
    if (fused) {
        retrieval.producer_id = 0x80000000u | (20u << 16) | 99u;
        retrieval.configuration_id = retrieval.producer_id;
        retrieval.provenance = {
            { 20, 20, 0, 3 },
            { 99, 99, 3, 48 },
        };
    }
    return { retrieval, mtp };
}

static void feed(common_speculative_controller & controller, const llama_tokens & tokens, llama_seq_id seq_id = 0) {
    for (const auto token : tokens) {
        controller.observe_output_token(token, seq_id);
    }
}

static common_speculative_selection select_and_observe(
        common_speculative_controller & controller,
        std::vector<common_speculative_candidate> candidates,
        size_t accepted,
        bool add_mismatch = false,
        llama_seq_id seq_id = 0,
        uint32_t global_budget = 0) {
    auto selection = controller.select(candidates, seq_id);
    if (global_budget > 0) {
        std::vector<common_speculative_selection> selections { selection };
        uint32_t cursor = 0;
        common_speculative_apply_global_budget(selections, global_budget, cursor);
        selection = selections[0];
    }
    if (selection.candidate_index < 0) {
        return selection;
    }
    auto & selected = candidates[selection.candidate_index];
    selected.tokens.resize(std::min<size_t>(selection.prefix_length, selected.tokens.size()));
    for (auto & span : selected.provenance) {
        span.end = std::min<uint16_t>(span.end, selected.tokens.size());
    }
    accepted = std::min(accepted, selected.tokens.size());
    llama_tokens realized(selected.tokens.begin(), selected.tokens.begin() + accepted);
    if (add_mismatch && accepted < selected.tokens.size()) {
        realized.push_back(-1);
    }
    controller.observe(candidates, selected.candidate_id, realized, 100, 0, seq_id);
    return selection;
}

static void make_trusted(common_speculative_controller & controller, const std::vector<common_speculative_candidate> & candidates) {
    controller.select(candidates);
    feed(controller, llama_tokens(candidates[0].tokens.begin(), candidates[0].tokens.begin() + 16));
}

static int enter_hot(common_speculative_controller & controller, const std::vector<common_speculative_candidate> & candidates) {
    make_trusted(controller, candidates);
    const auto entry = select_and_observe(controller, candidates, 16);
    CHECK(entry.admission_level == 2);
    CHECK(entry.prefix_length == 16);
    CHECK(controller.hot_metrics().entries == 1);
    CHECK(controller.hot_metrics().promotions_24 == 1);
    return 0;
}

int main() {
    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_candidates();
        CHECK(enter_hot(controller, candidates) == 0);

        const auto rung24 = select_and_observe(controller, make_candidates(24), 24);
        CHECK(rung24.hot_rung == 24);
        CHECK(rung24.hot_requested_length == 24);
        CHECK(rung24.prefix_length == 24);
        CHECK(controller.hot_metrics().promotions_32 == 1);

        const auto rung32 = select_and_observe(controller, make_candidates(32), 32);
        CHECK(rung32.hot_rung == 32);
        CHECK(rung32.prefix_length == 32);
        CHECK(controller.hot_metrics().promotions_48 == 1);

        const auto rung48 = select_and_observe(controller, candidates, 48);
        CHECK(rung48.hot_rung == 48);
        CHECK(rung48.prefix_length == 48);
        CHECK(controller.hot_metrics().full_matches == 4);
        CHECK(controller.hot_metrics().retained == 1);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto standalone = make_candidates();
        CHECK(enter_hot(controller, standalone) == 0);
        const auto fused = make_fused_candidates(24);
        const auto selected = controller.select(fused);
        CHECK(selected.admission_level == 2);
        CHECK(selected.hot_rung == 24);
        CHECK(selected.prefix_length == 24);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_fused_candidates(24);
        CHECK(enter_hot(controller, candidates) == 0);

        const auto retained = select_and_observe(controller, candidates, 19, true);
        CHECK(retained.hot_rung == 24);
        CHECK(controller.hot_metrics().retained == 1);
        CHECK(controller.select(candidates).hot_rung == 24);

        select_and_observe(controller, candidates, 20, true);
        CHECK(controller.hot_metrics().retained == 2);
        CHECK(controller.select(candidates).hot_rung == 24);

        const auto rolled_back = select_and_observe(controller, candidates, 18, true);
        CHECK(rolled_back.hot_rung == 24);
        CHECK(controller.hot_metrics().rollbacks == 1);
        CHECK(controller.select(candidates).hot_rung == 16);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_fused_candidates(24);
        CHECK(enter_hot(controller, candidates) == 0);
        select_and_observe(controller, candidates, 4, true);
        CHECK(controller.hot_metrics().resets == 1);
        const auto after_reset = controller.select(candidates);
        CHECK(after_reset.hot_rung == 0);
        CHECK(after_reset.challenger_cooldown_remaining == 32);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_candidates();
        CHECK(enter_hot(controller, candidates) == 0);
        const auto limited = select_and_observe(controller, make_candidates(24), 20, false, 0, 20);
        CHECK(limited.unbudgeted_prefix_length == 24);
        CHECK(limited.prefix_length == 20);
        CHECK(controller.hot_metrics().promotions_32 == 0);
        CHECK(controller.select(candidates).hot_rung == 24);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_candidates();
        CHECK(enter_hot(controller, candidates) == 0);
        auto active = make_candidates(24);
        const auto selection = controller.select(active);
        CHECK(selection.hot_rung == 24);
        auto & selected = active[selection.candidate_index];
        selected.tokens.resize(selection.prefix_length);
        controller.observe(active, selected.candidate_id, llama_tokens(selected.tokens.begin(), selected.tokens.begin() + 10), 100, 0);
        CHECK(controller.hot_metrics().promotions_32 == 0);
        CHECK(controller.select(candidates).hot_rung == 24);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_candidates();
        CHECK(enter_hot(controller, candidates) == 0);
        const auto short_candidates = make_candidates(20);
        const auto short_selection = select_and_observe(controller, short_candidates, 20);
        CHECK(short_selection.hot_rung == 24);
        CHECK(short_selection.hot_requested_length == 0);
        CHECK(controller.hot_metrics().promotions_32 == 0);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto producer_one = make_candidates();
        CHECK(enter_hot(controller, producer_one) == 0);
        CHECK(controller.select(make_candidates(48, 2)).hot_rung == 0);
        CHECK(controller.select(make_candidates(48, 1, 8192)).hot_rung == 0);

        auto slow = make_candidates(48);
        slow[0].metadata.proposal_time_us = 1000000;
        slow[0].metadata.cycle_proposal_time_us = 1000000;
        const auto fallback = controller.select(slow);
        CHECK(fallback.candidate_index == 1);
        CHECK(fallback.hot_rung == 0);
        CHECK(controller.active_hot_entries() == 1);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48, COMMON_SPECULATIVE_CONTROLLER_PERSISTENCE_PROCESS, 2);
        controller.observe_ordinary(100000, 0);
        const auto candidates = make_candidates();
        make_trusted(controller, candidates);
        select_and_observe(controller, candidates, 16, false, 0);
        CHECK(controller.active_hot_entries() == 1);
        CHECK(controller.select(candidates, 1).hot_rung == 0);
        controller.end_request(0);
        CHECK(controller.active_hot_entries() == 0);
        controller.begin_request(0, "other");
        CHECK(controller.select(candidates, 0).hot_rung == 0);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        const auto candidates = make_candidates();
        CHECK(enter_hot(controller, candidates) == 0);
        const auto mtp = make_candidates()[1];
        for (int i = 0; i < 64; ++i) {
            controller.observe({ mtp }, mtp.candidate_id, mtp.tokens, 100, 0);
        }
        const auto expired = controller.select(candidates);
        CHECK(expired.hot_rung == 0);
        CHECK(controller.hot_metrics().expirations == 1);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        auto candidates = make_retrieval_candidates();
        const auto shadow_only = controller.select(candidates);
        CHECK(shadow_only.candidate_index == 1);
        feed(controller, llama_tokens(candidates[0].tokens.begin(), candidates[0].tokens.begin() + 8));
        feed(controller, llama_tokens(candidates[0].tokens.begin() + 8, candidates[0].tokens.begin() + 16));
        const auto selected = controller.select(candidates);
        CHECK(selected.candidate_index == 0);
        CHECK(selected.prefix_length >= 3 && selected.prefix_length <= 48);
        CHECK(selected.hot_applied == false);

        auto migrated = candidates;
        migrated[0].metadata.retrieval_evidence_key = 0xbf;
        CHECK(controller.select(migrated).candidate_index == 0);
        controller.observe(candidates, candidates[0].candidate_id, { -1 }, 100, 0);

        const auto fused_cooldown = controller.select(make_retrieval_candidates(true));
        CHECK(fused_cooldown.candidate_index == 1);
        CHECK(fused_cooldown.challenger_cooldown_remaining == 32);

        const auto mtp = candidates[1];
        for (int i = 0; i < 32; ++i) {
            controller.observe({ mtp }, mtp.candidate_id, mtp.tokens, 100, 0);
        }
        const auto recovered = controller.select(make_retrieval_candidates(true));
        CHECK(recovered.candidate_index == 0);
        CHECK(recovered.prefix_length >= 3 && recovered.prefix_length <= 48);
        CHECK(recovered.hot_rung == 0);
    }

    {
        common_speculative_controller controller(COMMON_SPECULATIVE_CONTROLLER_MODE_ADAPTIVE, 0.0f, 1.0f, 0, 48);
        controller.observe_ordinary(100000);
        auto candidates = make_retrieval_candidates();
        candidates[0].tokens.resize(16);
        candidates[0].provenance[0].end = 16;
        controller.observe(candidates, candidates[0].candidate_id, candidates[0].tokens, 100, 0);
        const auto trusted = controller.select(make_retrieval_candidates());
        CHECK(trusted.candidate_index == 0);
        CHECK(trusted.admission_level == 2);
        CHECK(!trusted.hot_applied);

    }

    std::puts("stage 7 plus retrieval controller tests passed");
    return 0;
}
