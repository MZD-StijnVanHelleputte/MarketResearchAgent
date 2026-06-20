using System.Net.Http.Json;
using System.Text.Json;

namespace KomatsuIntel.Frontend.Services;

// ── HTTP client wrapper ─────────────────────────────────────────────────────

public class ApiClient(HttpClient http)
{
    private static readonly JsonSerializerOptions _json = new()
    {
        PropertyNamingPolicy        = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    // POST /chat — start a new intelligence run
    public async Task<ChatResponse?> StartChatAsync(ChatRequest request, CancellationToken ct = default)
    {
        var resp = await http.PostAsJsonAsync("chat", request, _json, ct);
        return await resp.Content.ReadFromJsonAsync<ChatResponse>(_json, ct);
    }

    // POST /runs/{runId}/gates/{gate}/approve
    public async Task<GateDecisionResponse?> ApproveGateAsync(string runId, int gate, object payload, CancellationToken ct = default)
    {
        var resp = await http.PostAsJsonAsync($"runs/{runId}/gates/{gate}/approve", payload, _json, ct);
        return await resp.Content.ReadFromJsonAsync<GateDecisionResponse>(_json, ct);
    }

    // POST /runs/{runId}/gates/{gate}/redirect
    public async Task<GateDecisionResponse?> RedirectGateAsync(string runId, int gate, object payload, CancellationToken ct = default)
    {
        var resp = await http.PostAsJsonAsync($"runs/{runId}/gates/{gate}/redirect", payload, _json, ct);
        return await resp.Content.ReadFromJsonAsync<GateDecisionResponse>(_json, ct);
    }

    // GET /runs/{runId}/report — download PDF bytes
    public Task<byte[]?> DownloadReportAsync(string runId, CancellationToken ct = default)
        => http.GetByteArrayAsync($"runs/{runId}/report", ct);

    // GET /sessions
    public Task<SessionList?> GetSessionsAsync(CancellationToken ct = default)
        => http.GetFromJsonAsync<SessionList>("sessions", _json, ct);

    // DELETE /sessions/{sessionId}
    public Task DeleteSessionAsync(string sessionId, CancellationToken ct = default)
        => http.DeleteAsync($"sessions/{sessionId}", ct);

    // GET /preferences
    public Task<PreferencesResponse?> GetPreferencesAsync(CancellationToken ct = default)
        => http.GetFromJsonAsync<PreferencesResponse>("preferences", _json, ct);

    // PUT /preferences
    public async Task UpdatePreferencesAsync(object payload, CancellationToken ct = default)
        => await http.PutAsJsonAsync("preferences", payload, _json, ct);

    // GET /runs/{runId} — Phase-2 run status polling; replaced by /sessions in Phase 3
    public Task<RunStatus?> GetRunAsync(string runId, CancellationToken ct = default)
        => http.GetFromJsonAsync<RunStatus>($"runs/{runId}", _json, ct);

    // GET /health
    public Task<HealthResponse?> HealthAsync(CancellationToken ct = default)
        => http.GetFromJsonAsync<HealthResponse>("health", _json, ct);

    // GET /metrics — dashboard run/token/cost aggregates
    public Task<DashboardSummary?> GetDashboardAsync(CancellationToken ct = default)
        => http.GetFromJsonAsync<DashboardSummary>("metrics", _json, ct);

    // POST /runs/{runId}/timeout_confirm/continue — user chose to keep the agent running
    public Task ContinueAfterTimeoutAsync(string runId, CancellationToken ct = default)
        => http.PostAsync($"runs/{runId}/timeout_confirm/continue", null, ct);

    // POST /runs/{runId}/timeout_confirm/stop — user chose to stop and get a partial report
    public Task StopAfterTimeoutAsync(string runId, CancellationToken ct = default)
        => http.PostAsync($"runs/{runId}/timeout_confirm/stop", null, ct);

    // POST /runs/{runId}/episodic/save — persist a completed run to episodic memory
    public async Task<EpisodicSaveResponse?> SaveToEpisodicAsync(string runId, CancellationToken ct = default)
    {
        var resp = await http.PostAsJsonAsync($"runs/{runId}/episodic/save", new { }, _json, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<EpisodicSaveResponse>(_json, ct);
    }

    // POST /tests/run/{testId}
    public async Task<TestRunResult?> RunTestAsync(string path, CancellationToken ct = default)
    {
        var resp = await http.PostAsJsonAsync(path, new { }, _json, ct);
        return await resp.Content.ReadFromJsonAsync<TestRunResult>(_json, ct);
    }

    // GET /runs/{runId}/logs — per-step event log
    public async Task<List<StepEvent>> GetRunLogsAsync(string runId, CancellationToken ct = default)
        => await http.GetFromJsonAsync<List<StepEvent>>($"runs/{runId}/logs", _json, ct) ?? [];

    // GET /knowledge — list industry knowledge documents
    public async Task<KnowledgeListResponse?> ListKnowledgeAsync(CancellationToken ct = default)
    {
        var resp = await http.GetAsync("knowledge", ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<KnowledgeListResponse>(_json, ct);
    }

    // POST /knowledge — create a background ingest job for one file (multipart: binary
    // file + domain). Returns immediately with a job_id; the actual convert/chunk/embed/
    // store pipeline runs server-side — poll GetKnowledgeJobAsync for progress.
    public async Task<KnowledgeJobCreated?> CreateKnowledgeJobAsync(string filename, string domain, byte[] content, CancellationToken ct = default)
    {
        using var form = new MultipartFormDataContent();
        using var fileContent = new ByteArrayContent(content);
        form.Add(fileContent, "file", filename);
        form.Add(new StringContent(domain), "domain");

        var resp = await http.PostAsync("knowledge", form, ct);
        if (!resp.IsSuccessStatusCode)
        {
            var body = await resp.Content.ReadAsStringAsync(ct);
            string message;
            try
            {
                var problem = JsonSerializer.Deserialize<JsonElement>(body);
                message = problem.TryGetProperty("detail", out var detail)
                    ? detail.ToString()
                    : body;
            }
            catch (JsonException)
            {
                message = body;
            }
            if (string.IsNullOrWhiteSpace(message))
                message = $"{(int)resp.StatusCode} {resp.StatusCode}";
            throw new HttpRequestException(message, null, resp.StatusCode);
        }
        return await resp.Content.ReadFromJsonAsync<KnowledgeJobCreated>(_json, ct);
    }

    // GET /knowledge/jobs/{jobId} — poll the stage of an in-flight or completed ingest job
    public Task<KnowledgeJobStatus?> GetKnowledgeJobAsync(string jobId, CancellationToken ct = default)
        => http.GetFromJsonAsync<KnowledgeJobStatus>($"knowledge/jobs/{jobId}", _json, ct);

    // DELETE /knowledge/{source}
    public async Task DeleteKnowledgeAsync(string source, CancellationToken ct = default)
    {
        var encoded = Uri.EscapeDataString(source);
        var resp = await http.DeleteAsync($"knowledge/{encoded}", ct);
        resp.EnsureSuccessStatusCode();
    }
}

// ── Request / Response models (mirror backend api/schemas/) ────────────────

public record ChatRequest(string Query, string? SessionId = null);
public record ChatResponse(string RunId, string SessionId, string Status);

public record GateDecisionResponse(string RunId, int Gate, string Decision, string? NextStatus);

// Gate 1 — consolidated plan review
public record PlannedToolCall(
    string Tool,
    JsonElement Params,
    string Domain,
    string Rationale = "");

public record ConsolidatedPlan(
    string PlanId,
    List<string> SourcePlanIds,
    List<string> DomainsActive,
    JsonElement EntityManifest,
    List<PlannedToolCall> PlannedToolCalls,
    string ResearchFindings,
    string Rationale,
    string GapReport,
    double FeasibilityScore,
    double QualityScore);

public record ConsolidatedPlanData(string RunId, ConsolidatedPlan Plan);

// Gate 1 — legacy plan review (kept for backward compatibility)
public record CandidatePlan(
    string PlanId,
    double FeasibilityScore,
    double QualityScore,
    string Rationale,
    string GapReport);

public record PlanReviewData(string RunId, List<CandidatePlan> Plans);

// Gate 2 — gathered-data review (actual data collected per domain)
public record DatasetItem(string? Title, string? Url, string? Snippet);
public record DomainDataset(
    string Tool,
    string Title,
    string Kind,                       // "table" | "list" | "summary"
    List<string>? Columns,
    List<List<string>>? Rows,
    int RowCount,
    string? Summary,
    List<DatasetItem>? Items);
public record DomainData(string Domain, List<DomainDataset> Datasets, List<string> Errors);
public record GatheredDataView(string RunId, List<DomainData> Domains);

// Gate 3 — brief review
public record BriefSubsection(string Title, string Content);
public record BriefSection(string Title, string Content, bool Flagged = false, List<BriefSubsection>? Subsections = null);
public record BriefReviewData(string RunId, List<BriefSection> Sections, string ExecutiveSummary);

// Sessions
public record Session(string SessionId, string CreatedAt, int RunCount);
public record SessionList(List<Session> Sessions, int Total);

// Preferences
public record PreferencesResponse(Dictionary<string, object>? Preferences);

// Health
public record HealthResponse(string Status, string? Version);

// Run status polling (Phase 2)
public record SourceDto(string Domain, string Tool, string Title, string? Url, string? PublishedAt);
public record RunStatus(string Status, string Stage, string? Brief, string? Error, List<SourceDto>? Sources, JsonElement? GateData = null, string? StatusMessage = null, List<string>? ActivityLog = null, string? ExecSummary = null, List<string>? Warnings = null);

// Test runner
public record TestRunResult(bool Passed, string Output);

// Knowledge management
public record KnowledgeDocument(string Source, string Domain, int ChunkCount, string AddedAt);
public record KnowledgeListResponse(List<KnowledgeDocument> Documents, int Total);
public record KnowledgeJobCreated(string JobId);
public record KnowledgeJobStatus(
    string  JobId,
    string  Filename,
    string  Domain,
    string  Stage,
    int?    ChunksTotal,
    int?    ChunksEmbedded,
    int?    ChunksAdded,
    string? Error);

// ── Frontend-only display models (not API-backed) ──────────────────────────

// Archive page
public record ReportSummary(
    string RunId,
    string SessionId,
    string Query,
    string CreatedAt,
    int TokensUsed,
    double CostUsd,
    string Status,
    string[] DomainsActive);

// Dashboard page
public record RunMetric(
    string RunId,
    string Query,
    string StartedAt,
    int DurationSeconds,
    int TotalTokens,
    double CostUsd,
    int ToolCalls,
    string Status);

public record DashboardSummary(
    int TotalRuns,
    long TotalTokens,
    double TotalCostUsd,
    double AvgDurationSeconds,
    List<RunMetric> RecentRuns);

// Episodic memory save
public record EpisodicSaveResponse(string Status, string RunId, string Collection);

// Testing page
public record TestCase(
    string Id,
    string Name,
    string Category,
    string Description,
    string ExpectedBehaviour);

public enum TestStatus { Pending, Running, Pass, Fail }

public record TestResult(string TestId, TestStatus Status, string? Output, DateTime? RunAt);

// Step event log
public record StepEvent(
    int     Id,
    string  Ts,
    string  Level,
    string  Stage,
    string  Domain,
    string  EventType,
    string  Label,
    string? Detail);
