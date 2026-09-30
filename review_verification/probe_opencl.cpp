// Probe: does the system OpenCL loader expose any platform/ICD device?
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef void* cl_platform_id;

int main() {
    void* h = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
        printf("dlopen libOpenCL.so.1 FAILED: %s\n", dlerror());
        return 1;
    }
    printf("dlopen libOpenCL.so.1 OK: %p\n", h);
    using Fn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
    auto clGetPlatformIDs = reinterpret_cast<Fn>(dlsym(h, "clGetPlatformIDs"));
    if (!clGetPlatformIDs) {
        printf("dlsym clGetPlatformIDs FAILED: %s\n", dlerror());
        return 1;
    }
    cl_uint n = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &n);
    printf("clGetPlatformIDs(0,NULL,&n) -> err=%d (CL_PLATFORM_NOT_FOUND_KHR=-1001), num_platforms=%u\n", err, n);
    if (err != 0 || n == 0) {
        printf("RESULT: NO usable OpenCL platform/ICD device on this machine\n");
        return 2;
    }
    printf("RESULT: %u platform(s) available\n", n);
    return 0;
}
