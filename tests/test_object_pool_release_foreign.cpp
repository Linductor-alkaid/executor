#ifdef NDEBUG
#undef NDEBUG
#endif

#include "util/object_pool.hpp"

#include <gtest/gtest.h>

namespace {

struct PoolValue {
    int value{0};
};

TEST(ObjectPoolReleaseGuard, ForeignPointerTriggersDebugAssert) {
    executor::util::ObjectPool<PoolValue> pool(1);
    PoolValue foreign;

    EXPECT_DEATH(pool.release(&foreign), "ObjectPool::release called with a foreign pointer");
}

}  // namespace
