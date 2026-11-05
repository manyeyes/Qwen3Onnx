using Microsoft.Extensions.Logging;

namespace Onar.Inference;

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
        Action<string>? onToken = null,
        CancellationToken cancellationToken = default);
}

public interface IConversationPromptFormatter
{
    string FormatConversationPrompt(
        string systemPrompt,
        IReadOnlyList<ChatTurn> history,
        string question,
        bool thinkMode);
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
        Register("qwen3-onnx-genai", config => new OnnxInferenceGenaiEngine(
            config, _loggerFactory?.CreateLogger<OnnxInferenceGenaiEngine>()));
        Register("llama-onnx-genai", config => new OnnxInferenceLlamaGenaiEngine(
            config, _loggerFactory?.CreateLogger<OnnxInferenceLlamaGenaiEngine>()));
        Register("phi-onnx-genai", config => new OnnxInferencePhiGenaiEngine(
            config, _loggerFactory?.CreateLogger<OnnxInferencePhiGenaiEngine>()));
        Register("deepseek-onnx-genai", config => new OnnxInferenceDeepSeekGenaiEngine(
            config, _loggerFactory?.CreateLogger<OnnxInferenceDeepSeekGenaiEngine>()));
        Register("gemma-onnx-genai", config => new OnnxInferenceGemmaGenaiEngine(
            config, _loggerFactory?.CreateLogger<OnnxInferenceGemmaGenaiEngine>()));
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
