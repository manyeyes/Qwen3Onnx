using Microsoft.Extensions.Logging;

namespace Onar.Inference;

public sealed class OnnxInferenceDeepSeekGenaiEngine : OnnxInferenceChatTemplateGenaiEngine
{
    public OnnxInferenceDeepSeekGenaiEngine(
        AssistantConfig config,
        ILogger<OnnxInferenceDeepSeekGenaiEngine>? logger = null)
        : base(config, "deepseek-onnx-genai", logger)
    {
    }

    /// <summary>
    /// չʾ�ı�����ʼǰ׺��
    /// ˼��ģʽ������ `<think>\n`���÷����ı����������� `<think>��</think>` �ṹ��UI �ɾݴ�չʾ˼���顣
    /// ��˼��ģʽ�����ؿգ�ģ�ͻ�ֱ�����ɴ𰸣�չʾ�ı�����ǰ׺��
    /// </summary>
    protected override string GetGenerationPrefix(GenerationConfig config) =>
        config.ThinkMode ? "<think>\n" : string.Empty;

    /// <summary>
    /// prompt ���˼��ģʽ���ơ�
    /// ģ��ĩβ�̶�д�� `<think>\n`����˼��ģʽ���� prompt ĩβ׷�� `</think>\n\n` �պ�����
    /// ģ�Ϳ�����˼���������������ֱ�����ɴ𰸡�
    /// ���� DeepSeek-R1 �ٷ� chat_template �� enable_thinking=False ʱ����Ϊ��
    /// </summary>
    protected override string ApplyThinkingMode(string prompt, bool thinkMode) =>
        thinkMode ? prompt : prompt + "</think>\n\n";

    /// <summary>
    /// ����ͨ������ָ��׷��Լ�������� prompt Ԥ������ṹ�Կ��ơ�
    /// </summary>
    protected override string AdaptUserMessage(string content, bool thinkMode)
    {
        _ = thinkMode;
        return content;
    }

    /// <summary>
    /// ���������˼��ģʽ�£���ģ��ż�������Ԥ����ʽ�������˼�����ݣ�
    /// �ص��һ�� `</think>` ֮ǰ�Ĳ��֣�ֻ�������մ𰸡�
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