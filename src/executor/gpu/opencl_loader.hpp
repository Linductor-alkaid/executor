#pragma once

#include <string>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef EXECUTOR_ENABLE_OPENCL
#include <CL/cl.h>
#else
// OpenCL类型定义（无OpenCL头文件时）
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_event;

typedef cl_uint cl_bool;
typedef cl_uint cl_device_type;
typedef cl_uint cl_platform_info;
typedef cl_uint cl_device_info;
typedef cl_uint cl_context_properties;
typedef cl_uint cl_command_queue_properties;
typedef cl_uint cl_mem_flags;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_kernel_work_group_info;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_TRUE 1
#define CL_FALSE 0
#endif

namespace executor {
namespace gpu {

// OpenCL函数指针类型
using clGetPlatformIDsFunc = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);
using clGetDeviceIDsFunc = cl_int (*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
using clGetDeviceInfoFunc = cl_int (*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
using clCreateContextFunc = cl_context (*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
using clReleaseContextFunc = cl_int (*)(cl_context);
using clCreateCommandQueueFunc = cl_command_queue (*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
using clReleaseCommandQueueFunc = cl_int (*)(cl_command_queue);
using clCreateBufferFunc = cl_mem (*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using clReleaseMemObjectFunc = cl_int (*)(cl_mem);
using clEnqueueReadBufferFunc = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
using clEnqueueWriteBufferFunc = cl_int (*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
using clEnqueueCopyBufferFunc = cl_int (*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
using clFinishFunc = cl_int (*)(cl_command_queue);
using clFlushFunc = cl_int (*)(cl_command_queue);
using clCreateProgramWithSourceFunc = cl_program (*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
using clBuildProgramFunc = cl_int (*)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
using clReleaseProgramFunc = cl_int (*)(cl_program);
using clCreateKernelFunc = cl_kernel (*)(cl_program, const char*, cl_int*);
using clReleaseKernelFunc = cl_int (*)(cl_kernel);
using clSetKernelArgFunc = cl_int (*)(cl_kernel, cl_uint, size_t, const void*);
using clEnqueueNDRangeKernelFunc = cl_int (*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
using clWaitForEventsFunc = cl_int (*)(cl_uint, const cl_event*);
using clReleaseEventFunc = cl_int (*)(cl_event);

struct OpenCLFunctionPointers {
    clGetPlatformIDsFunc clGetPlatformIDs = nullptr;
    clGetDeviceIDsFunc clGetDeviceIDs = nullptr;
    clGetDeviceInfoFunc clGetDeviceInfo = nullptr;
    clCreateContextFunc clCreateContext = nullptr;
    clReleaseContextFunc clReleaseContext = nullptr;
    clCreateCommandQueueFunc clCreateCommandQueue = nullptr;
    clReleaseCommandQueueFunc clReleaseCommandQueue = nullptr;
    clCreateBufferFunc clCreateBuffer = nullptr;
    clReleaseMemObjectFunc clReleaseMemObject = nullptr;
    clEnqueueReadBufferFunc clEnqueueReadBuffer = nullptr;
    clEnqueueWriteBufferFunc clEnqueueWriteBuffer = nullptr;
    clEnqueueCopyBufferFunc clEnqueueCopyBuffer = nullptr;
    clFinishFunc clFinish = nullptr;
    clFlushFunc clFlush = nullptr;
    clCreateProgramWithSourceFunc clCreateProgramWithSource = nullptr;
    clBuildProgramFunc clBuildProgram = nullptr;
    clReleaseProgramFunc clReleaseProgram = nullptr;
    clCreateKernelFunc clCreateKernel = nullptr;
    clReleaseKernelFunc clReleaseKernel = nullptr;
    clSetKernelArgFunc clSetKernelArg = nullptr;
    clEnqueueNDRangeKernelFunc clEnqueueNDRangeKernel = nullptr;
    clWaitForEventsFunc clWaitForEvents = nullptr;
    clReleaseEventFunc clReleaseEvent = nullptr;

    bool is_complete() const {
        return clGetPlatformIDs && clGetDeviceIDs && clGetDeviceInfo &&
               clCreateContext && clReleaseContext &&
               clCreateCommandQueue && clReleaseCommandQueue &&
               clCreateBuffer && clReleaseMemObject &&
               clEnqueueReadBuffer && clEnqueueWriteBuffer && clEnqueueCopyBuffer &&
               clFinish && clFlush;
    }
};

class OpenCLLoader {
public:
    static OpenCLLoader& instance();
    ~OpenCLLoader();

    OpenCLLoader(const OpenCLLoader&) = delete;
    OpenCLLoader& operator=(const OpenCLLoader&) = delete;

    bool load();
    void unload();
    bool is_available() const;
    const OpenCLFunctionPointers& get_functions() const;
    std::string get_dll_path() const;

private:
    OpenCLLoader();
    std::string search_opencl_dll();
    bool try_load_dll(const std::string& dll_path);
    bool load_functions();
    void* get_function_pointer(const char* function_name);

#ifdef _WIN32
    std::string search_windows_paths();
#else
    std::string search_linux_paths();
#endif

    mutable std::mutex mutex_;
    bool is_loaded_;
    std::string dll_path_;
#ifdef _WIN32
    HMODULE dll_handle_;
#else
    void* dll_handle_;
#endif
    OpenCLFunctionPointers functions_;
};

} // namespace gpu
} // namespace executor
