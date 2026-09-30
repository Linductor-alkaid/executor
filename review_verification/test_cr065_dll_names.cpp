// CR-065: Windows CUDA DLL name construction audit.
// Reproduces VERBATIM the three name-construction sites of
// cuda_loader.cpp and prints every candidate filename, then compares with
// the official NVIDIA cudart DLL names.
//
//   Site 1  cuda_loader.cpp:135-137  (CUDA_PATH\bin, version loop 12..9):
//           "\bin\cudart64_" + std::to_string(version) + ".dll"
//   Site 2  cuda_loader.cpp:184-189  (PATH search, literal table):
//           "cudart64_12.dll", "cudart64_11.dll", "cudart64_10.dll", "cudart64_9.dll"
//   Site 3  cuda_loader.cpp:307-352  (search_windows_paths, version dirs
//           v12.6..v9.0, collapses to major): "cudart64_" + major + ".dll"
//
// Official names (NVIDIA CUDA toolkit, Windows, x64):
//   CUDA 12.x -> cudart64_12.dll
//   CUDA 11.0..11.8 -> cudart64_110.dll .. cudart64_118.dll
//   CUDA 10.0..10.2 -> cudart64_100.dll .. cudart64_102.dll
//   CUDA 9.0..9.2  -> cudart64_90.dll .. cudart64_92.dll
#include <cstdio>
#include <string>
#include <vector>

int main() {
    std::printf("== Site 1: CUDA_PATH loop, version 12..9 (cuda_loader.cpp:135-137) ==\n");
    for (int version = 12; version >= 9; --version) {
        std::string dll_path = std::string("CUDA_PATH") + "\\bin\\cudart64_" +
                               std::to_string(version) + ".dll";
        std::printf("  %s\n", dll_path.c_str());
    }

    std::printf("== Site 2: PATH search literal table (cuda_loader.cpp:184-189) ==\n");
    const char* dll_names[] = {"cudart64_12.dll", "cudart64_11.dll",
                               "cudart64_10.dll", "cudart64_9.dll"};
    for (const char* n : dll_names) std::printf("  %s\n", n);

    std::printf("== Site 3: search_windows_paths (cuda_loader.cpp:307-352) ==\n");
    std::vector<std::string> versions = {
        "v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0",
        "v11.8", "v11.7", "v11.6", "v11.5", "v11.4", "v11.3", "v11.2", "v11.1", "v11.0",
        "v10.2", "v10.1", "v10.0",
        "v9.2", "v9.1", "v9.0"};
    for (const std::string& version : versions) {
        int major_version = 12;
        if (version.length() >= 3 && version[0] == 'v') {
            major_version = std::stoi(version.substr(1));
        }
        // Official per-minor DLL number: decimal concatenation of the major
        // and minor digits (v11.8 -> "11""8" = cudart64_118.dll; v10.1 ->
        // "10""1" = cudart64_101.dll; v9.2 -> "9""2" = cudart64_92.dll).
        // CUDA 12.x is the exception: one name for all minors,
        // cudart64_12.dll.
        const int minor = std::stoi(version.substr(version.find('.') + 1));
        char official[32];
        if (major_version >= 12) {
            std::snprintf(official, sizeof(official), "cudart64_%d.dll", major_version);
        } else {
            std::snprintf(official, sizeof(official), "cudart64_%s%d.dll",
                          version.substr(1, version.find('.') - 1).c_str(), minor);
        }
        std::printf("  dir %-6s -> constructed: cudart64_%d.dll   official: %s%s\n",
                    version.c_str(), major_version, official,
                    (major_version >= 12) ? "  [MATCH]" : "  [MISMATCH]");
    }
    return 0;
}
