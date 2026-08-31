#include "ngram-retrieval.h"

#include <cstdio>
#include <vector>

using test_tokens = std::vector<llama_token>;

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #value); return 1; } } while (0)

static int test_collision_rejection() {
    common_ngram_retrieval retrieval(48, 0);
    const test_tokens tokens = { 30, 31, 32, 33, 90, 91, 1, 2, 3, 4, 10, 11, 1, 2, 3, 4 };
    const auto result = retrieval.query(tokens);
    CHECK(result.tokens == test_tokens({ 10, 11 }));
    CHECK(retrieval.metrics().rejected_collision > 0);
    return 0;
}

static int test_append_rebuild_and_bounds() {
    common_ngram_retrieval retrieval;
    retrieval.reset({ 1, 2, 3, 4, 5 });
    retrieval.synchronize({ 1, 2, 3, 4, 5, 6, 7 });
    CHECK(retrieval.indexed_tokens() == 7);
    retrieval.synchronize({ 9, 8, 7, 6 });
    CHECK(retrieval.indexed_tokens() == 4);

    test_tokens repeated(256, 42);
    retrieval.reset(repeated);
    CHECK(retrieval.max_posting_count() == 32);
    return 0;
}

static int test_branch_policies() {
    const test_tokens tokens = {
        1, 2, 3, 4, 10, 11, 90,
        1, 2, 3, 4, 10, 12, 91,
        1, 2, 3, 4, 20, 21, 92,
        1, 2, 3, 4,
    };
    common_ngram_retrieval retrieval;
    const auto recent = retrieval.query(tokens, {}, COMMON_NGRAM_RETRIEVAL_LONGEST_RECENT);
    const auto majority = retrieval.query(tokens, {}, COMMON_NGRAM_RETRIEVAL_MAJORITY_PREFIX);
    const auto weighted = retrieval.query(tokens, {}, COMMON_NGRAM_RETRIEVAL_WEIGHTED_PREFIX);
    CHECK(!recent.tokens.empty() && recent.tokens[0] == 20);
    CHECK(majority.tokens.size() >= 2 && majority.tokens[0] == 10);
    CHECK(weighted.tokens.size() >= 2 && weighted.tokens[0] == 10);
    CHECK(majority.evidence[0].source_support == 2);
    CHECK(majority.evidence[0].dominance_permille == 666);
    return 0;
}

static int test_recency_tie_and_mtp() {
    const test_tokens tokens = {
        1, 2, 3, 4, 10, 90,
        1, 2, 3, 4, 20, 91,
        1, 2, 3, 4,
    };
    common_ngram_retrieval retrieval;
    const auto result = retrieval.query(tokens, { 20 }, COMMON_NGRAM_RETRIEVAL_MAJORITY_PREFIX);
    CHECK(!result.tokens.empty() && result.tokens[0] == 20);
    CHECK(result.evidence[0].mtp_agreement);
    return 0;
}

static int test_request_reset() {
    common_ngram_retrieval retrieval;
    const test_tokens first = { 1, 2, 3, 4, 10, 11, 1, 2, 3, 4 };
    CHECK(!retrieval.query(first).tokens.empty());
    retrieval.reset({ 7, 8, 9, 10 });
    CHECK(retrieval.query({ 7, 8, 9, 10 }).tokens.empty());
    CHECK(retrieval.indexed_tokens() == 4);
    return 0;
}

static int test_zero_copy_append_and_transient_extension() {
    common_ngram_retrieval retrieval;
    const test_tokens history = { 1, 2, 3, 4, 10, 11, 1, 2, 3 };
    const auto actual = retrieval.query(history, 4);
    CHECK(actual.tokens == test_tokens({ 10, 11 }));
    CHECK(retrieval.indexed_tokens() == history.size() + 1);

    const test_tokens hypothetical = { 1, 2, 3, 4, 10, 11, 1, 2, 3, 4, 10, 11, 1, 2, 3 };
    const size_t indexed = retrieval.indexed_tokens();
    const auto extension = retrieval.query_transient(hypothetical, 4);
    CHECK(!extension.tokens.empty());
    CHECK(retrieval.indexed_tokens() == indexed);

    const test_tokens continued = { 1, 2, 3, 4, 10, 11, 1, 2, 3, 4, 10 };
    retrieval.query(continued, 11);
    CHECK(retrieval.indexed_tokens() == continued.size() + 1);
    return 0;
}

static int test_long_anchor_survives_short_anchor_pressure() {
    common_ngram_retrieval retrieval;
    test_tokens tokens;
    const test_tokens anchor = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    tokens.insert(tokens.end(), anchor.begin(), anchor.end());
    tokens.insert(tokens.end(), { 99, 100 });
    for (llama_token i = 0; i < 40; ++i) {
        tokens.insert(tokens.end(), { 1000 + i, 13, 14, 15, 16, 2000 + i });
    }
    tokens.insert(tokens.end(), anchor.begin(), anchor.end());
    const auto result = retrieval.query(tokens);
    CHECK(result.tokens.size() >= 2);
    CHECK(result.tokens[0] == 99 && result.tokens[1] == 100);
    CHECK(result.evidence[0].match_length >= 16);
    return 0;
}

int main() {
    CHECK(test_collision_rejection() == 0);
    CHECK(test_append_rebuild_and_bounds() == 0);
    CHECK(test_branch_policies() == 0);
    CHECK(test_recency_tie_and_mtp() == 0);
    CHECK(test_request_reset() == 0);
    CHECK(test_zero_copy_append_and_transient_extension() == 0);
    CHECK(test_long_anchor_survives_short_anchor_pressure() == 0);
    return 0;
}
