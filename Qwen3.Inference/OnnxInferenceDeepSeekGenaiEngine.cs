using Microsoft.Extensions.Logging;

namespace Qwen3.Inference;

public sealed class OnnxInferenceDeepSeekGenaiEngine : OnnxInferenceChatTemplateGenaiEngine
{
    public OnnxInferenceDeepSeekGenaiEngine(
        AssistantConfig config,
        ILogger<OnnxInferenceDeepSeekGenaiEngine>? logger = null)
        : base(config, "deepseek-onnx-genai", logger)
    {
    }

    /// <summary>
    /// 展示文本的起始前缀。
    /// 思考模式：补上 `<think>\n`，让返回文本包含完整的 `<think>…</think>` 结构，UI 可据此展示思考块。
    /// 非思考模式：返回空，模型会直接生成答案，展示文本无需前缀。
    /// </summary>
    protected override string GetGenerationPrefix(GenerationConfig config) =>
        config.ThinkMode ? "<think>\n" : string.Empty;

    /// <summary>
    /// prompt 层的思考模式控制。
    /// 模板末尾固定写入 `<think>\n`；非思考模式下在 prompt 末尾追加 `</think>\n\n` 闭合它，
    /// 模型看到空思考块后会跳过推理，直接生成答案。
    /// 这是 DeepSeek-R1 官方 chat_template 在 enable_thinking=False 时的行为。
    /// </summary>
    protected override string ApplyThinkingMode(string prompt, bool thinkMode) =>
        thinkMode ? prompt : prompt + "</think>\n\n";

    /// <summary>
    /// 不再通过语言指令追加约束，交由 prompt 预填充做结构性控制。
    /// </summary>
    protected override string AdaptUserMessage(string content, bool thinkMode)
    {
        _ = thinkMode;
        return content;
    }

    /// <summary>
    /// 后备清理：非思考模式下，若模型偶发不遵从预填充格式、仍输出思考内容，
    /// 截掉第一个 `</think>` 之前的部分，只保留最终答案。
    /// </summary>
    protected override string NormalizeGeneratedText(string text, GenerationConfig config)
    {
        if (config.ThinkMode)
        {
            return text;
        }

        var closeIndex = text.IndexOf("</think>", StringComparison.Ordinal);
        if (closeIndex < 0)
        {
            return text;
        }

        return text[(closeIndex + "</think>".Length)..].TrimStart();
    }
}