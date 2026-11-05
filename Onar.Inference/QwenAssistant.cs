using Microsoft.Extensions.Logging;

namespace Onar.Inference;

public sealed class QwenAssistant : ChatAssistant
{
    public QwenAssistant(
        AssistantConfig config,
        IInferenceEngineFactory? engineFactory = null,
        ILogger<ChatAssistant>? logger = null)
        : base(config, engineFactory, logger)
    {
    }
}
