# Prefab packaging template

This directory is a template for publishing executor through Android Prefab/AAR.

Expected AAR layout:

```text
executor-android-<version>.aar
└── prefab/modules/executor/
    ├── module.json
    ├── include/executor/...
    └── libs/
        ├── android.arm64-v8a/libexecutor.so
        └── android.x86_64/libexecutor.so
```

Copy this `module.json` unchanged. Do not package tests or examples into the AAR. If the
consumer uses `c++_shared`, also ship `libc++_shared.so` per ABI under `jni/<abi>/`.
See `docs/PACKAGE_ANDROID.md` for the full Gradle and packaging workflow.
