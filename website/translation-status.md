---
title: Translation Status
description: Published English coverage and the synchronization status for Chinese documentation.
sidebar: false
---

# Translation Status

English pages use the same repository examples and commands as Chinese pages; no language-specific code copies are maintained. API names, error codes, configuration fields, paths, and commands remain in English so they are searchable against the source and reference documentation.

## Published English paths

| Chinese route | English route | Status | Fact source |
| --- | --- | --- | --- |
| `/` | `/en/` | Published | Repository README and tutorial example `01` |
| `/zh/getting-started/what-is-executor` | `/en/getting-started/what-is-executor` | Published | Public Facade and website design guidance |
| `/zh/quick-start/build` | `/en/quick-start/build` | Published | `docs/BUILD.md` |
| `/zh/quick-start/first-task` | `/en/quick-start/first-task` | Published | `examples/tutorial/01_first_task.cpp` |
| `/zh/quick-start/task-inputs-and-ownership` | `/en/quick-start/task-inputs-and-ownership` | Published | `examples/tutorial/11_task_inputs.cpp` |
| `/zh/quick-start/return-values-and-errors` | `/en/quick-start/return-values-and-errors` | Published | `std::future` and tutorial example `01` |
| `/zh/quick-start/lifecycle` | `/en/quick-start/lifecycle` | Published | Public lifecycle API and `docs/API.md` |
| `/zh/tutorial/` | `/en/tutorial/` | Published | `examples/tutorial/02`–`05`, `10`, and `comm_robot_pipeline` |
| `/zh/guides/` | `/en/guides/` | Published | Public Facade/communication APIs, tutorials, and production guidance |
| `/zh/realtime-and-communication/` | `/en/realtime-and-communication/` | Published | Realtime and communication public APIs; tutorial examples `07`–`08` |
| `/zh/reliability/` | `/en/reliability/` | Published | Failure/status APIs, monitoring, deployment checks, and tutorial example `06` |
| `/zh/advanced/` | `/en/advanced/` | Published | Public advanced APIs, current source architecture, and benchmark protocol |
| `/zh/gpu/` | `/en/gpu/` | Published | GPU public APIs, backend diagnostics, and tutorial example `09` |
| `/zh/reference/version-and-migration` | `/en/reference/version-and-migration` | Published | `CHANGELOG.md`, `docs/MIGRATION.md` |

## Pending topic groups

| Chinese route group | English status | Synchronization rule |
| --- | --- | --- |
| `/zh/quick-start/` | Complete | All six Chinese quick-start pages have an English counterpart. |
| `/zh/tutorial/` | Complete | All eight Chinese tutorial pages have English counterparts using the same C++ examples and commands. |
| `/zh/guides/` | Complete | All five Chinese scenario and production guides have English counterparts. |
| `/zh/realtime-and-communication/` | Complete | All six Chinese real-time and communication pages have English counterparts. |
| `/zh/reliability/` | Complete | All five Chinese reliability pages have English counterparts. |
| `/zh/advanced/` | Complete | All seven Chinese advanced and internal-behavior pages have English counterparts. |
| `/zh/gpu/` | Complete | All four Chinese GPU pages have English counterparts. |
| `/zh/reference/api` | Linked source | Keep `docs/API.md` as the single complete signature reference. |

## Fallback behavior

The language control opens the exact translated counterpart when it is published. On a Chinese page without an English counterpart, it opens the English home page and labels the destination as a home fallback; no empty English page is published. English pages always link back to their Chinese counterpart.

When a Chinese page adds a public capability, update this table to `Needs translation` or add the published English route in the same change. Release review must check this table together with the API and migration documents.
