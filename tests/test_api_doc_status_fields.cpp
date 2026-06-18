// P-260618-008: doc-vs-struct constraint test.
// Parses docs/API.md §7.3 "RealtimeExecutorStatus" entry and asserts the
// field names listed there match the actual members of
// `executor::RealtimeExecutorStatus` in include/executor/types.hpp.
//
// Prevents the recurrence of the bug fixed by P-260618-008: a struct gets
// new fields (dropped_task_count, failed_pushes, peak_queue_size,
// queue_capacity) but docs/API.md §7.3 is not updated, leaving users with
// stale documentation.

#include <executor/types.hpp>

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

TEST(ApiDocStatusFields, RealtimeExecutorStatusEntryMatchesStruct) {
    const std::vector<std::string> candidates = {
        "docs/API.md",
        "../docs/API.md",
        "../../docs/API.md",
    };

    std::ifstream in;
    std::string path_used;
    for (const auto& p : candidates) {
        in.open(p);
        if (in.good()) {
            path_used = p;
            break;
        }
        in.clear();
    }
    ASSERT_TRUE(in.good()) << "Could not open docs/API.md from any candidate path";

    std::stringstream ss;
    ss << in.rdbuf();
    const std::string api_md = ss.str();
    in.close();

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

}  // namespace
