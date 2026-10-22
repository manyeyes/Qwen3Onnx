using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Qwen3.Inference;

namespace Qwen3.Desktop.Services;

public sealed class ConversationTitleService(ILogger logger)
{
    public string CreateFallback(string firstMessage)
    {
        var normalized = Regex.Replace(firstMessage, @"\s+", " ").Trim();
        if (normalized.Length == 0)
        {
            return "附件对话";
        }
        return normalized.Length <= 32 ? normalized : normalized[..32].TrimEnd() + "…";
    }

    public string Generate(
        ChatAssistant assistant,
        string firstQuestion,
        string firstAnswer,
        string currentModelSystemPrompt,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var config = new GenerationConfig
            {
                ThinkMode = false,
                ShowThinking = false,
                ThinkingBudget = 0,
                MultiTurn = false,
                MaxTurns = 0,
                MaxHistoryChars = 0,
                MaxHistoryTokens = 0,
                MaxNewTokens = 32,
                Stream = false,
                Temperature = 0.1,
                TopP = 1,
                TopK = 1,
                RepetitionPenalty = 1,
                DoSample = false,
                SystemPrompt = "请根据用户问题和回答生成一个简洁的中文对话标题。仅输出标题，不加引号，不超过20个汉字。",
                StripThinkingInHistory = true,
                SanitizeHistory = true,
                ResetOnFactQuestion = false,
                EnableRepetitionCheck = false,
                EnableLoopDetection = false
            };
            var titleInput = $"用户问题：{firstQuestion}\n回答摘要：{firstAnswer}";
            var result = assistant.Ask(
                titleInput,
                Scene.Precise,
                generationConfig: config,
                cancellationToken: cancellationToken);
            var title = Regex.Replace(
                    result.Answer, @"<think[^>]*>.*?</think\s*>", string.Empty,
                    RegexOptions.Singleline | RegexOptions.CultureInvariant)
                .Replace("<think>", string.Empty, StringComparison.Ordinal)
                .Replace("</think>", string.Empty, StringComparison.Ordinal)
                .Trim().Trim('"', '“', '”', '\'', '。', '.', ':', '：');
            if (title.Length is < 2 or > 40)
            {
                return CreateFallback(firstQuestion);
            }
            return title;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception)
        {
            logger.LogWarning(exception, "Conversation title generation failed; using an extractive title");
            return CreateFallback(firstQuestion);
        }
        finally
        {
            assistant.RestoreHistory([], GenerationConfig.SceneConfigs[Scene.Chitchat] with
            {
                SystemPrompt = currentModelSystemPrompt
            });
        }
    }
}
