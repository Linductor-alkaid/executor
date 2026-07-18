#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <executor/executor.hpp>

namespace {

struct SensorFrame {
    int id;
    int samples;
};

int score_frame(SensorFrame frame, int weight) {
    return frame.samples * weight;
}

class Planner {
public:
    explicit Planner(std::string name) : name_(std::move(name)) {}

    std::string make_plan(SensorFrame frame) const {
        return name_ + "-frame-" + std::to_string(frame.id);
    }

private:
    std::string name_;
};

void count_frame(std::atomic<int>& processed) {
    processed.fetch_add(1);
}

}

int main() {
    auto& executor = executor::Executor::instance();
    SensorFrame frame{7, 21};

    auto score = executor.submit(score_frame, frame, 2);

    auto planner = std::make_shared<Planner>("local");
    auto plan = executor.submit(&Planner::make_plan, planner, frame);

    int offset = 5;
    auto adjusted = executor.submit([frame, offset]() noexcept {
        return frame.samples + offset;
    });

    auto payload = std::make_unique<int>(9);
    auto owned = executor.submit([payload = std::move(payload)]() mutable noexcept {
        return *payload;
    });

    std::atomic<int> processed{0};
    auto counted = executor.submit(count_frame, std::ref(processed));

    std::cout << "score=" << score.get() << ", plan=" << plan.get()
              << ", adjusted=" << adjusted.get() << ", owned=" << owned.get()
              << '\n';
    counted.get();
    std::cout << "processed=" << processed.load() << '\n';

    executor.shutdown();
    return 0;
}
