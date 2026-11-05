using Microsoft.Extensions.Logging;

namespace Qwen3.Inference;

public sealed class OnnxInferencePhiGenaiEngine : OnnxInferenceChatTemplateGenaiEngine
{
    public OnnxInferencePhiGenaiEngine(
        AssistantConfig config,
        ILogger<OnnxInferencePhiGenaiEngine>? logger = null)
        : base(config, "phi-onnx-genai", logger)
    {
    }
}
