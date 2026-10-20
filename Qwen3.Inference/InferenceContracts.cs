using Microsoft.Extensions.Logging;

namespace Qwen3.Inference;

public sealed record ChatTurn(string User, string Assistant);

public interface IInferenceEngine : IDisposable
{
    string EngineId { get; }
    long HistoryLength { get; }
    int CountTokens(string text);
    void ResetState();
    void Prefill(string text);
    string FormatSystemPrompt(string systemPrompt);
    string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns);
    string FormatUserPrompt(string question, bool thinkMode);
    (string Text, GenerationStats Stats) Generate(
        string prompt,
        GenerationConfig config,
        Action<string>? onToken = null);
}

public interface IInferenceEngineFactory
{
    IInferenceEngine Create(AssistantConfig config);
}

public sealed class InferenceEngineFactory : IInferenceEngineFactory
{
    private readonly Dictionary<string, Func<AssistantConfig, IInferenceEngine>> _registrations =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly ILoggerFactory? _loggerFactory;

    public InferenceEngineFactory(ILoggerFactory? loggerFactory = null)
    {
        _loggerFactory = loggerFactory;
        Register("qwen3-onnx", config => new OnnxInferenceEngine(
            config, _loggerFactory?.CreateLogger<OnnxInferenceEngine>()));
    }

    public void Register(string engineId, Func<AssistantConfig, IInferenceEngine> factory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(engineId);
        ArgumentNullException.ThrowIfNull(factory);
        if (!_registrations.TryAdd(engineId, factory))
        {
            throw new InvalidOperationException($"An inference engine named '{engineId}' is already registered.");
        }
    }

    public IInferenceEngine Create(AssistantConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (!_registrations.TryGetValue(config.EngineId, out var factory))
        {
            throw new NotSupportedException(
                $"Inference engine '{config.EngineId}' is not registered. Registered engines: {string.Join(", ", _registrations.Keys)}.");
        }

        return factory(config);
    }
}
