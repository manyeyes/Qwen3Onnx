using Microsoft.Extensions.Logging;

namespace Onar.Inference;

public sealed class OnnxInferenceGemmaGenaiEngine : OnnxInferenceChatTemplateGenaiEngine
{
    private static readonly string[] StopTokens = ["<end_of_turn>"];

    public OnnxInferenceGemmaGenaiEngine(
        AssistantConfig config,
        ILogger<OnnxInferenceGemmaGenaiEngine>? logger = null)
        : base(config, "gemma-onnx-genai", logger, StopTokens)
    {
    }

    protected override string NormalizeDecodedChunk(string text) =>
        text.Replace('▁', ' ');
}
