// CR-030 诊断探针：直接区分 pthread_setschedparam 与 setpriority 各自的
// 返回值与 errno。
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <thread>

int main() {
    std::thread worker([] {});
    auto handle = worker.native_handle();

    errno = 0;
    sched_param p{};
    p.sched_priority = 0;
    int r1 = pthread_setschedparam(handle, SCHED_OTHER, &p);
    std::printf("pthread_setschedparam(worker, SCHED_OTHER, prio=0) = %d errno=%d (%s)\n",
                r1, r1 != 0 ? r1 : 0, r1 != 0 ? strerror(r1) : "ok");

    errno = 0;
    int r2 = setpriority(PRIO_PROCESS, 0, 19);
    std::printf("setpriority(PRIO_PROCESS, 0, 19) = %d errno=%d (%s)\n",
                r2, errno, r2 != 0 ? strerror(errno) : "ok");
    errno = 0;
    std::printf("main nice after = %d\n", getpriority(PRIO_PROCESS, 0));

    worker.join();
    return 0;
}
// 追加：直接调用库实现对比（重新编译时替换 main 之后的内容无效，这里用独立 main2 不行——
// 实际上下面这些行不会被编译，因为 main 已返回；本文件仅作探针 1 用途）
