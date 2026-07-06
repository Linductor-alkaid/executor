// P-260618-008: doc-vs-struct constraint test.
// Parses docs/API.md §7.3 "RealtimeExecutorStatus" entry and asserts the
// field names listed there match the actual members of
// `executor::RealtimeExecutorStatus` in include/executor/types.hpp.
//
// Prevents the recurrence of the bug fixed by P-260618-008: a struct gets
// new fields (dropped_task_count, failed_pushes, peak_queue_size,
// queue_capacity) but docs/API.md §7.3 is not updated, leaving users with
// stale documentation. Also covers P-008 batch performance claim sources.

#include <executor/types.hpp>
#include "executor/thread_pool/thread_pool.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string extract_status_entry(const std::string& api_md) {
    const std::string marker = "- **RealtimeExecutorStatus**";
    auto pos = api_md.find(marker);
    if (pos == std::string::npos) {
        return {};
    }
    auto end = api_md.find("\n- **", pos + marker.size());
    if (end == std::string::npos) {
        end = api_md.size();
    }
    return api_md.substr(pos, end - pos);
}

std::set<std::string> extract_field_names(const std::string& entry) {
    std::set<std::string> fields;
    std::regex re("`([a-zA-Z_][a-zA-Z0-9_]*)`");
    for (auto it = std::sregex_iterator(entry.begin(), entry.end(), re);
         it != std::sregex_iterator(); ++it) {
        fields.insert((*it)[1].str());
    }
    return fields;
}

std::set<std::string> expected_struct_fields() {
    return {
        "name",
        "is_running",
        "cycle_period_ns",
        "cycle_count",
        "cycle_timeout_count",
        "avg_cycle_time_ns",
        "max_cycle_time_ns",
        "dropped_task_count",
        "failed_pushes",
        "peak_queue_size",
        "queue_capacity",
    };
}

std::string read_doc_from_candidates(const std::vector<std::string>& candidates,
                                     std::string& path_used) {
    std::ifstream in;
    for (const auto& p : candidates) {
        in.open(p);
        if (in.good()) {
            path_used = p;
            break;
        }
        in.clear();
    }
    if (!in.good()) {
        return {};
    }

    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool contains_regex(const std::string& text, const std::string& pattern) {
    return std::regex_search(text, std::regex(pattern));
}

TEST(ApiDocStatusFields, RealtimeExecutorStatusEntryMatchesStruct) {
    const std::vector<std::string> candidates = {
        "docs/API.md",
        "../docs/API.md",
        "../../docs/API.md",
    };

    std::string path_used;
    const std::string api_md = read_doc_from_candidates(candidates, path_used);
    ASSERT_FALSE(api_md.empty()) << "Could not open docs/API.md from any candidate path";

    const auto entry = extract_status_entry(api_md);
    ASSERT_FALSE(entry.empty())
        << "RealtimeExecutorStatus entry not found in docs/API.md §7.3";

    const auto doc_fields = extract_field_names(entry);
    const auto struct_fields = expected_struct_fields();

    std::set<std::string> doc_struct_fields;
    for (const auto& f : doc_fields) {
        if (struct_fields.count(f) > 0) {
            doc_struct_fields.insert(f);
        }
    }

    EXPECT_EQ(doc_struct_fields, struct_fields)
        << "RealtimeExecutorStatus doc-vs-struct mismatch in " << path_used;
    for (const auto& f : doc_struct_fields) {
        if (struct_fields.count(f) == 0) {
            ADD_FAILURE() << "  - " << f;
        }
    }
    for (const auto& f : struct_fields) {
        if (doc_struct_fields.count(f) == 0) {
            ADD_FAILURE() << "  + " << f;
        }
    }
}

TEST(ApiDocStatusFields, ApiDocPerformanceClaimsHaveSources) {
    std::string api_path;
    const std::string api_md = read_doc_from_candidates(
        {"docs/API.md", "../docs/API.md", "../../docs/API.md"}, api_path);
    ASSERT_FALSE(api_md.empty()) << "Could not open docs/API.md from any candidate path";

    std::string readme_path;
    const std::string readme_md = read_doc_from_candidates(
        {"README.md", "../README.md", "../../README.md"}, readme_path);
    ASSERT_FALSE(readme_md.empty()) << "Could not open README.md from any candidate path";

    const std::vector<std::pair<std::string, std::string>> docs = {
        {api_path, api_md},
        {readme_path, readme_md},
    };

    for (const auto& [path, text] : docs) {
        EXPECT_TRUE(text.find("docs/performance/lockfree_task_executor_baseline.md") !=
                        std::string::npos ||
                    text.find("performance/lockfree_task_executor_baseline.md") !=
                        std::string::npos)
            << path << " batch performance claim must link the Markdown source";
        EXPECT_TRUE(text.find("docs/performance/batch_submit_baseline_2026-03-13.json") !=
                        std::string::npos ||
                    text.find("performance/batch_submit_baseline_2026-03-13.json") !=
                        std::string::npos)
            << path << " batch performance claim must link the JSON source";
        EXPECT_NE(text.find("benchmark_batch_submit_real"), std::string::npos)
            << path << " batch performance claim must include the benchmark command";
        EXPECT_TRUE(contains_regex(text, "date[^0-9]{0,20}2026-03-13") ||
                    contains_regex(text, "日期[^0-9]{0,20}2026-03-13"))
            << path << " batch performance claim must include the benchmark date";
    }

    EXPECT_NE(readme_md.find("3–5x"), std::string::npos)
        << "README.md public batch performance claim should stay conservative at 3-5x";
    EXPECT_NE(api_md.find("实测加速比 | 公开推荐范围"), std::string::npos)
        << "docs/API.md should keep measured ratios separate from public guidance";
    EXPECT_NE(api_md.find("16.47x"), std::string::npos)
        << "docs/API.md should preserve the measured 2000-task ratio";
}

TEST(ApiDocThreadPoolSnippetCompiles, FixedThreadPoolExampleInitializes) {
    executor::ThreadPoolConfig config;
    config.min_threads = 16;
    config.max_threads = 16;

    executor::ThreadPool pool;
    ASSERT_TRUE(pool.initialize(config));

    pool.shutdown(true);
}

}  // namespace
