using Microsoft.Extensions.Logging;

namespace Qwen3.Inference;

public sealed record AssistantResult(string Answer, GenerationStats Generation, int PromptTokens,
    int HistoryTurns, bool FactReset, bool RepetitionDetected, bool? ThinkOverride, string SceneName);

public class ChatAssistant : IDisposable
{
    private readonly AssistantConfig _config;
    private readonly IInferenceEngine _engine;
    private readonly ConversationMemory _memory = new();
    private readonly ILogger<ChatAssistant>? _logger;
    private GenerationConfig? _currentConfig;

    public IReadOnlyList<ChatTurn> History => _memory.Turns;

    public ChatAssistant(
        AssistantConfig config,
        IInferenceEngineFactory? engineFactory = null,
        ILogger<ChatAssistant>? logger = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger;
        _engine = (engineFactory ?? new InferenceEngineFactory()).Create(config);
        _logger?.LogInformation("Using inference engine {EngineId}", _engine.EngineId);
    }

    public void ClearMemory()
    {
        _memory.Clear();
        _engine.ResetState();
        _currentConfig = null;
    }

    public void RestoreHistory(IEnumerable<ChatTurn> turns, GenerationConfig config)
    {
        ArgumentNullException.ThrowIfNull(turns);
        ArgumentNullException.ThrowIfNull(config);
        _memory.Replace(turns);
        while (_memory.Count > config.MaxTurns)
        {
            _memory.RemoveOldestTurn();
        }
        _currentConfig = config;
        TrimHistoryToTokenLimit(config.MaxHistoryTokens);
        RebuildCacheFromHistory(config.SystemPrompt ?? _config.SystemPrompt);
    }

    public AssistantResult Ask(
        string question,
        Scene? scene = null,
        bool? forceThinkMode = null,
        Action<string>? onToken = null,
        GenerationConfig? generationConfig = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var cleanedQuestion = TextUtilities.CleanInput(question);
        var detectedScene = scene ?? TextUtilities.DetectScene(cleanedQuestion);
        var effectiveScene = scene ?? detectedScene;
        var config = generationConfig ?? GenerationConfig.SceneConfigs[effectiveScene];
        bool? thinkOverride = null;
        if (forceThinkMode.HasValue && forceThinkMode.Value != config.ThinkMode)
        {
            config = config with { ThinkMode = forceThinkMode.Value };
            thinkOverride = forceThinkMode;
        }

        if (_currentConfig is not null &&
            (_currentConfig.MultiTurn != config.MultiTurn ||
             _currentConfig.ThinkMode != config.ThinkMode ||
             _currentConfig.SystemPrompt != config.SystemPrompt))
        {
            _logger?.LogInformation("Generation mode changed; clearing conversation state");
            ClearMemory();
        }
        _currentConfig = config;

        if (cleanedQuestion.Length == 0)
        {
            return new AssistantResult(string.Empty, new GenerationStats(0, 0, 0, 0, false, false),
                0, _memory.Count, false, false, thinkOverride, generationConfig is null
                    ? GenerationConfig.GetSceneName(effectiveScene)
                    : "custom");
        }

        var factReset = false;
        if (config.ResetOnFactQuestion && TextUtilities.IsFactQuestion(cleanedQuestion) && _memory.Count > 0)
        {
            _logger?.LogInformation("Fact question detected; clearing conversation history");
            ClearMemory();
            factReset = true;
        }

        var systemPrompt = config.SystemPrompt ?? _config.SystemPrompt;
        string prompt;
        if (config.MultiTurn)
        {
            RebuildCacheFromHistory(systemPrompt);
            prompt = _engine.FormatUserPrompt(cleanedQuestion, config.ThinkMode);
        }
        else
        {
            _engine.ResetState();
            prompt = _engine.FormatSystemPrompt(systemPrompt) +
                     _engine.FormatUserPrompt(cleanedQuestion, config.ThinkMode);
        }
        var promptTokenCount = _engine.CountTokens(prompt);
        if (_engine.HistoryLength + promptTokenCount > _config.MaxSequenceLength)
        {
            throw new InvalidOperationException(
                $"Prompt and KV cache exceed the configured maximum sequence length ({_config.MaxSequenceLength}).");
        }

        var (answer, generationStats) = _engine.Generate(prompt, config, onToken, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        var repetitionDetected = false;
        if (config.EnableRepetitionCheck)
        {
            var cleanAnswer = TextUtilities.ExtractAnswerForHistory(answer);
            if (cleanAnswer.Length > 0 &&
                TextUtilities.IsRepetition(cleanAnswer, _memory.RecentAnswers(), config.RepetitionThreshold))
            {
                answer = "（抱歉，我没有生成新的回答，请换个说法再问一次。）";
                repetitionDetected = true;
                _logger?.LogWarning("A repeated answer was detected and discarded");
            }
        }

        if (config.MultiTurn)
        {
            var historyAnswer = config.StripThinkingInHistory
                ? TextUtilities.ExtractAnswerForHistory(answer)
                : answer;
            if (config.SanitizeHistory && historyAnswer.Length > 0)
            {
                historyAnswer = TextUtilities.SanitizeRoleplay(historyAnswer);
            }
            _memory.AddTurn(cleanedQuestion, historyAnswer, config.MaxTurns, config.MaxHistoryChars);
            TruncateHistory(config.MaxHistoryTokens);
        }

        return new AssistantResult(answer, generationStats, promptTokenCount, _memory.Count,
            factReset, repetitionDetected, thinkOverride, generationConfig is null
                ? GenerationConfig.GetSceneName(effectiveScene)
                : "custom");
    }

    public void Dispose() => _engine.Dispose();

    private void RebuildCacheFromHistory(string systemPrompt)
    {
        _engine.ResetState();
        _engine.Prefill(_engine.FormatSystemPrompt(systemPrompt));
        var historyPrompt = _engine.FormatHistoryPrompt(_memory.Turns);
        if (historyPrompt.Length > 0)
        {
            _engine.Prefill(historyPrompt);
        }
    }

    private void TruncateHistory(int maxHistoryTokens)
    {
        TrimHistoryToTokenLimit(maxHistoryTokens);
        RebuildCacheFromHistory(_currentConfig?.SystemPrompt ?? _config.SystemPrompt);
    }

    private void TrimHistoryToTokenLimit(int maxHistoryTokens)
    {
        while (_memory.Count > 0 && _engine.CountTokens(
                   _engine.FormatHistoryPrompt(_memory.Turns)) > maxHistoryTokens)
        {
            _memory.RemoveOldestTurn();
        }
    }
}
