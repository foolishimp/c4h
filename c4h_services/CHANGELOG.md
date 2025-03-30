# Change Log

All notable changes to the C4H Services will be documented in this file.

## [0.2.1] - 2023-07-18

### Added
- New Jobs API that provides a structured interface over the existing Workflow API
- `/api/v1/jobs` endpoint for submitting new jobs with workorder, team, and runtime configurations
- `/api/v1/jobs/{jobId}` endpoint for retrieving job status and results
- Bidirectional configuration mapping between Jobs and Workflow formats
- Persistence mechanism for job-to-workflow ID mapping

### Changed
- None (Workflow API remains unchanged and fully compatible)

### Fixed
- None