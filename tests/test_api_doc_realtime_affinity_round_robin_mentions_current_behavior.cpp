#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

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

std::string to_lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

bool contains_regex(const std::string& text, const std::string& pattern) {
    return std::regex_search(text, std::regex(pattern));
}

}  // namespace

TEST(ApiDocRealtimeAffinity, MentionsRoundRobinAutoAffinity) {
    std::string path_used;
    const std::string api_md = read_doc_from_candidates(
        {"docs/API.md", "../docs/API.md", "../../docs/API.md"}, path_used);
    ASSERT_FALSE(api_md.empty()) << "Could not open docs/API.md from any candidate path";

    const std::string lower_api = to_lower_ascii(api_md);
    EXPECT_FALSE(contains_regex(lower_api, R"(bind(s|ing)?( to)? core 0)"))
        << "docs/API.md must not claim empty realtime cpu_affinity binds core 0";
    EXPECT_FALSE(contains_regex(lower_api, R"(pin(s|ning)?( to)? core 0)"))
        << "docs/API.md must not claim empty realtime cpu_affinity pins core 0";
    EXPECT_FALSE(contains_regex(lower_api, R"(bind(s|ing)?( to)? cpu 0)"))
        << "docs/API.md must not claim empty realtime cpu_affinity binds CPU 0";
    EXPECT_NE(lower_api.find("round-robin"), std::string::npos)
        << "docs/API.md should document round-robin realtime auto affinity";
    EXPECT_NE(api_md.find("g_next_rt_cpu_hint"), std::string::npos)
        << "docs/API.md should name the auto-affinity hint used by the implementation";
}
