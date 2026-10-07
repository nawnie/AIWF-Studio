# Native Windows prototype

**Status:** Working local prototype; recorded October 7, 2026. It is not included in the current public Windows installer or stable release.

AIWF Studio's new direction is one local workstation for image creation, local-model chat, dataset curation, project history, and training-plan preparation. The Windows prototype presents those tasks in a native WinUI 3 app. It uses the local unified engine API and keeps optional model engines in separate processes and environments.

## Two interfaces, one workspace

- **Native Windows app:** the focused Windows workstation interface, with Windows-native controls, local GPU and engine status, and task-based navigation.
- **React web interface:** the existing public Pro app, built with React, TypeScript, and Vite over the FastAPI service. It is the browser-based path toward multi-platform access. A non-Windows deployment is not currently released or verified.

Both interfaces are designed around the same local API contracts, so shared operations can serve the Windows app, browser interface, CLI, and automation without duplicating engine integrations.

## Prototype scope

The local prototype includes Home, Chat, Create, Datasets, Train, Projects, and Settings screens. It can show GPU telemetry, stream local-model chat, generate images through an available image engine, catalog project images, and prepare a dry-run training plan. Training does not start from the native app. Optional engine services and model weights are not bundled with the prototype.

The README screenshots are dated prototype captures with the project selector generalized for privacy. Image generation and training-plan screens illustrate UI state; they do not establish a released installer or a general production deployment.
