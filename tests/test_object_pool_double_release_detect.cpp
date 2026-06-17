#ifdef NDEBUG
#undef NDEBUG
#endif

#include "util/object_pool.hpp"

#include <gtest/gtest.h>

namespace {

struct PoolValue {
    int value{0};
};

TEST(ObjectPoolReleaseGuard, DoubleReleaseTriggersDebugAssert) {
    executor::util::ObjectPool<PoolValue> pool(1);
    PoolValue* value = pool.acquire();
    ASSERT_NE(value, nullptr);

    pool.release(value);

    EXPECT_DEATH(pool.release(value), "ObjectPool::release called more than once");
}

}  // namespace
