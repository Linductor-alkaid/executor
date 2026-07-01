#include <gtest/gtest.h>

namespace {

// H-001 placeholder: keep the first harness PR focused on test infrastructure.
// Future PRs should replace these disabled cases with compiling examples once
// executor::comm facade APIs are intentionally introduced.

TEST(FacadeCommUsage, DISABLED_SensorProducerPlannerConsumer) {
    GTEST_SKIP() << "TODO: enable when executor::comm::MpscChannel<T> is introduced.";
}

TEST(FacadeCommUsage, DISABLED_ConfigThreadRealtimeControlThread) {
    GTEST_SKIP() << "TODO: enable when executor::comm::LatestMailbox<T> is introduced.";
}

TEST(FacadeCommUsage, DISABLED_InitThreadWorkerThread) {
    GTEST_SKIP() << "TODO: enable when executor::comm::PhaseGate is introduced.";
}

TEST(FacadeCommUsage, DISABLED_StateWriterMonitorReader) {
    GTEST_SKIP() << "TODO: enable when executor::comm::DoubleBuffer<T> is introduced.";
}

TEST(FacadeCommUsage, DISABLED_RealtimeCycleDrainsMessages) {
    GTEST_SKIP() << "TODO: enable when executor::comm::RealtimeChannel<T> is introduced.";
}

} // namespace
