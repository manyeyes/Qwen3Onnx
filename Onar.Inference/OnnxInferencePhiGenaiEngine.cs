using Microsoft.Extensions.Logging;

namespace Onar.Inference;

public sealed class OnnxInferencePhiGenaiEngine : OnnxInferenceChatTemplateGenaiEngine
{
    public OnnxInferencePhiGenaiEngine(
        AssistantConfig config,
        ILogger<OnnxInferencePhiGenaiEngine>? logger = null)
        : base(config, "phi-onnx-genai", logger)
    {
    }
}
