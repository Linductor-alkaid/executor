#pragma once

#include <chrono>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace executor::test::harness {

struct Operation {
    enum class Type {
        Send,
        Recv,
        Publish,
        Read
    };

    std::uint64_t thread_id{0};
    std::uint64_t op_id{0};
    Type type{Type::Send};
    std::uint64_t value{0};
    bool success{false};
    std::chrono::steady_clock::time_point start{};
    std::chrono::steady_clock::time_point end{};
};

struct InvariantResult {
    bool ok{true};
    std::vector<std::string> failures{};

    void fail(std::string message) {
        ok = false;
        failures.push_back(std::move(message));
    }
};

class OperationHistory {
public:
    void record(Operation operation) {
        operations_.push_back(std::move(operation));
    }

    [[nodiscard]] const std::vector<Operation>& operations() const {
        return operations_;
    }

    [[nodiscard]] std::size_t size() const {
        return operations_.size();
    }

    [[nodiscard]] InvariantResult check_no_duplicate_successful_receives() const {
        InvariantResult result;
        std::set<std::uint64_t> received_values;

        for (const Operation& operation : operations_) {
            if (operation.type != Operation::Type::Recv || !operation.success) {
                continue;
            }
            if (!received_values.insert(operation.value).second) {
                result.fail("duplicate successful receive for value " + std::to_string(operation.value));
            }
        }

        return result;
    }

    [[nodiscard]] InvariantResult check_no_phantom_successful_receives() const {
        InvariantResult result;
        std::set<std::uint64_t> sent_values;

        for (const Operation& operation : operations_) {
            if (operation.type == Operation::Type::Send && operation.success) {
                sent_values.insert(operation.value);
            }
        }

        for (const Operation& operation : operations_) {
            if (operation.type != Operation::Type::Recv || !operation.success) {
                continue;
            }
            if (sent_values.find(operation.value) == sent_values.end()) {
                result.fail("phantom successful receive for value " + std::to_string(operation.value));
            }
        }

        return result;
    }

    [[nodiscard]] InvariantResult check_basic_channel_invariants() const {
        InvariantResult result = check_no_duplicate_successful_receives();
        InvariantResult no_phantom = check_no_phantom_successful_receives();
        if (!no_phantom.ok) {
            result.ok = false;
            result.failures.insert(result.failures.end(), no_phantom.failures.begin(), no_phantom.failures.end());
        }
        return result;
    }

private:
    std::vector<Operation> operations_;
};

} // namespace executor::test::harness
