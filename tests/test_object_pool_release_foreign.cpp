#include "util/object_pool.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

struct PoolValue {
    int value{0};
};

TEST(ObjectPoolReleaseGuard, ForeignPointerThrowsLogicError) {
    executor::util::ObjectPool<PoolValue> pool(1);
    PoolValue foreign;

    EXPECT_THROW(pool.release(&foreign), std::logic_error);
}

}  // namespace
