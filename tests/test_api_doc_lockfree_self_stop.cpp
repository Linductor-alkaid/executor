// P-005: docs/API.md 关于 LockFreeTaskExecutor 自停止语义的一致性守护。
// 历史上 §5.7 曾声称任务内调用 exec.stop() 会“死锁”，与实现（worker_id_
// 检测后返回 false）和 §5.4「停止后的提交语义」相互矛盾；本测试防止该
// 说法回潮，并钉住正确的指导：任务内 stop 只请求自停止，join 由外部完成。
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
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

}  // namespace

TEST(ApiDocLockFreeSelfStop, DocumentsSelfStopWithoutDeadlockClaim) {
    std::string path_used;
    const std::string api_md = read_doc_from_candidates(
        {"docs/API.md", "../docs/API.md", "../../docs/API.md"}, path_used);
    ASSERT_FALSE(api_md.empty()) << "Could not open docs/API.md from any candidate path";

    // 任务内 stop()/stop_and_join() 是受支持的自停止请求，不会死锁；
    // 文档不得再宣称相反结论。
    EXPECT_EQ(api_md.find("死锁！消费者线程等待自己"), std::string::npos)
        << "docs/API.md must not claim exec.stop() inside a task deadlocks; "
           "the implementation detects worker_id_ and returns false";

    // 排障条目应说明 stop_and_join() 在工作线程内返回 false、join 由外部完成。
    EXPECT_NE(api_md.find("在工作线程内返回 `false`"), std::string::npos)
        << "docs/API.md should explain stop_and_join() returns false on the worker thread";
    EXPECT_NE(api_md.find("只请求自停止"), std::string::npos)
        << "docs/API.md should describe in-task stop as a self-stop request";

    // §5.4 的既有正确说明必须保留：任务/实时回调可安全请求自停止。
    EXPECT_NE(api_md.find("可安全调用 `stop()` / `stop_and_join()` 请求自停止"),
              std::string::npos)
        << "docs/API.md must keep the lifecycle-section statement that in-task "
           "self-stop requests are safe";
}
