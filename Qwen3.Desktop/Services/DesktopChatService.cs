using Microsoft.Extensions.Logging;
using Qwen3.Desktop.Models;
using Qwen3.Inference;

namespace Qwen3.Desktop.Services;

public sealed class DesktopChatService : IDisposable
{
    private readonly SqliteDatabase _database;
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<DesktopChatService> _logger;
    private readonly ConversationTitleService _titleService;
    private ChatAssistant? _assistant;
    private string? _loadedModelId;

    public DesktopChatService(SqliteDatabase database, ILoggerFactory loggerFactory)
    {
        _database = database;
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<DesktopChatService>();
        _titleService = new ConversationTitleService(_logger);
    }

    public async Task<IReadOnlyList<StoredMessage>> GetMessagesAsync(
        string conversationId, string? ownerUserId, CancellationToken cancellationToken = default)
    {
        var conversation = await _database.GetConversationAsync(
            conversationId, ownerUserId, cancellationToken);
        if (conversation is null)
        {
            throw new UnauthorizedAccessException("Conversation does not exist or is not accessible to this user.");
        }
        return await _database.GetMessagesAsync(conversationId, ownerUserId, cancellationToken);
    }

    public void ClearConversationState() => _assistant?.ClearMemory();

    public async Task<ChatSendResult> SendAsync(
        string? conversationId,
        string? ownerUserId,
        ModelDefinition model,
        string userText,
        IReadOnlyList<string> attachmentNames,
        IReadOnlyList<string> attachmentContents,
        GenerationConfig generationConfig,
        Action<string>? onToken,
        CancellationToken cancellationToken = default,
        string? promptTextOverride = null)
    {
        if (string.IsNullOrWhiteSpace(userText) && attachmentContents.Count == 0 &&
            string.IsNullOrWhiteSpace(promptTextOverride))
        {
            throw new ArgumentException("Enter a message or attach a text file.");
        }
        if (promptTextOverride is null && attachmentNames.Count != attachmentContents.Count)
        {
            throw new ArgumentException("Attachment names and contents do not match.");
        }
        if (!ModelDownloadService.IsInstalled(model))
        {
            throw new InvalidOperationException($"Model '{model.Title}' is not installed or is incomplete.");
        }

        var id = conversationId;
        var firstTurn = id is null;
        var title = firstTurn ? _titleService.CreateFallback(userText) : string.Empty;
        if (id is not null)
        {
            var existing = await _database.GetConversationAsync(id, ownerUserId, cancellationToken);
            if (existing is null)
            {
                throw new UnauthorizedAccessException("Conversation does not exist or is not accessible to this user.");
            }
            if (!string.Equals(existing.ModelId, model.Id, StringComparison.Ordinal))
            {
                throw new InvalidOperationException("The selected model does not match this conversation's model.");
            }
            title = existing.Title;
            var existingMessages = await _database.GetMessagesAsync(id, ownerUserId, cancellationToken);
            firstTurn = existingMessages.Count == 0;
        }

        cancellationToken.ThrowIfCancellationRequested();
        var modelConfig = new AssistantConfig
        {
            EngineId = "qwen3-onnx",
            ModelPath = model.LocalPath,
            EmbeddingModelName = model.EmbeddingFileName,
            DecoderModelName = model.DecoderFileName,
            SystemPrompt = generationConfig.SystemPrompt ?? "You are a helpful assistant."
        };
        await Task.Run(() => EnsureAssistant(model.Id, modelConfig), cancellationToken);
        var assistant = _assistant
            ?? throw new InvalidOperationException("The inference assistant was not initialized.");

        var existingHistory = id is null
            ? Array.Empty<StoredMessage>()
            : await _database.GetMessagesAsync(id, ownerUserId, cancellationToken);
        var turns = ToTurns(existingHistory);
        _logger.LogDebug(
            "Restoring desktop conversation history. ConversationId={ConversationId}, StoredMessages={StoredMessageCount}, Turns={TurnCount}, MaxTurns={MaxTurns}, ResetOnFactQuestion={ResetOnFactQuestion}",
            id, existingHistory.Count, turns.Count, generationConfig.MaxTurns,
            generationConfig.ResetOnFactQuestion);

        var promptText = promptTextOverride ?? BuildPrompt(userText, attachmentNames, attachmentContents);
        var result = await Task.Run(
            () =>
            {
                assistant.RestoreHistory(turns, generationConfig);
                return assistant.Ask(
                    promptText,
                    generationConfig: generationConfig,
                    onToken: onToken,
                    cancellationToken: cancellationToken);
            },
            cancellationToken);
        _logger.LogDebug(
            "Desktop inference completed. ConversationId={ConversationId}, PromptTokens={PromptTokens}, RetainedHistoryTurns={HistoryTurns}, GenerationTokens={GenerationTokens}, GenerationSeconds={GenerationSeconds:F3}, TokensPerSecond={TokensPerSecond:F2}",
            id, result.PromptTokens, result.HistoryTurns, result.Generation.TotalTokens,
            result.Generation.TotalTimeSeconds, result.Generation.TokensPerSecond);

        cancellationToken.ThrowIfCancellationRequested();
        if (firstTurn)
        {
            var titleResult = await Task.Run(
                () => _titleService.Generate(
                    assistant, userText, result.Answer, modelConfig.SystemPrompt, cancellationToken),
                cancellationToken);
            title = titleResult;
        }

        cancellationToken.ThrowIfCancellationRequested();
        id = await _database.SaveTurnAsync(
            id,
            ownerUserId,
            model.Id,
            title,
            firstTurn,
            promptText,
            attachmentNames,
            result.PromptTokens,
            result.Answer,
            result.Generation.TotalTokens,
            cancellationToken);
        if (firstTurn)
        {
            var savedHistory = ToTurns(await _database.GetMessagesAsync(
                id, ownerUserId, cancellationToken));
            await Task.Run(
                () => assistant.RestoreHistory(savedHistory, generationConfig),
                cancellationToken);
        }

        var conversation = await _database.GetConversationAsync(id, ownerUserId, cancellationToken)
            ?? throw new InvalidOperationException("Conversation disappeared after saving the response.");
        return new ChatSendResult(conversation, result, promptText);
    }

    public async Task<ChatSendResult> RegenerateLastReplyAsync(
        string conversationId,
        string? ownerUserId,
        ModelDefinition model,
        GenerationConfig generationConfig,
        Action<string>? onToken,
        CancellationToken cancellationToken = default)
    {
        var conversation = await _database.GetConversationAsync(
            conversationId, ownerUserId, cancellationToken)
            ?? throw new UnauthorizedAccessException("Conversation does not exist or is not accessible to this user.");
        if (!string.Equals(conversation.ModelId, model.Id, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("The selected model does not match this conversation's model.");
        }
        if (!ModelDownloadService.IsInstalled(model))
        {
            throw new InvalidOperationException($"Model '{model.Title}' is not installed or is incomplete.");
        }

        var messages = await _database.GetMessagesAsync(conversationId, ownerUserId, cancellationToken);
        if (messages.Count < 2 || messages[^1].Role != "assistant" || messages[^2].Role != "user")
        {
            throw new InvalidOperationException("The conversation has no complete final reply to regenerate.");
        }

        var userMessage = messages[^2];
        var oldAssistantMessage = messages[^1];
        var modelConfig = new AssistantConfig
        {
            EngineId = "qwen3-onnx",
            ModelPath = model.LocalPath,
            EmbeddingModelName = model.EmbeddingFileName,
            DecoderModelName = model.DecoderFileName,
            SystemPrompt = generationConfig.SystemPrompt ?? "You are a helpful assistant."
        };
        await Task.Run(() => EnsureAssistant(model.Id, modelConfig), cancellationToken);
        var assistant = _assistant
            ?? throw new InvalidOperationException("The inference assistant was not initialized.");
        var turns = ToTurns(messages.Take(messages.Count - 2).ToArray());
        var result = await Task.Run(
            () =>
            {
                assistant.RestoreHistory(turns, generationConfig);
                return assistant.Ask(
                    userMessage.Content,
                    generationConfig: generationConfig,
                    onToken: onToken,
                    cancellationToken: cancellationToken);
            },
            cancellationToken);

        await _database.ReplaceAssistantReplyAsync(
            conversationId, ownerUserId, oldAssistantMessage.Id, result.Answer,
            result.PromptTokens, result.Generation.TotalTokens, cancellationToken);
        var updatedConversation = await _database.GetConversationAsync(
            conversationId, ownerUserId, cancellationToken)
            ?? throw new InvalidOperationException("Conversation disappeared after regenerating the response.");
        return new ChatSendResult(updatedConversation, result, userMessage.Content);
    }

    public void Dispose()
    {
        _assistant?.Dispose();
        _assistant = null;
        _loadedModelId = null;
    }

    private void EnsureAssistant(string modelId, AssistantConfig config)
    {
        if (_assistant is not null && string.Equals(_loadedModelId, modelId, StringComparison.Ordinal))
        {
            return;
        }

        _assistant?.Dispose();
        _assistant = new ChatAssistant(
            config, new InferenceEngineFactory(_loggerFactory),
            _loggerFactory.CreateLogger<ChatAssistant>());
        _loadedModelId = modelId;
    }

    private static IReadOnlyList<ChatTurn> ToTurns(IReadOnlyList<StoredMessage> messages)
    {
        var turns = new List<ChatTurn>();
        string? pendingUser = null;
        foreach (var message in messages)
        {
            if (message.Role == "user")
            {
                if (pendingUser is not null)
                {
                    turns.Add(new ChatTurn(pendingUser, string.Empty));
                }
                pendingUser = message.Content;
            }
            else if (message.Role == "assistant" && pendingUser is not null)
            {
                turns.Add(new ChatTurn(pendingUser, message.Content));
                pendingUser = null;
            }
        }
        if (pendingUser is not null)
        {
            turns.Add(new ChatTurn(pendingUser, string.Empty));
        }
        return turns;
    }

    private static string BuildPrompt(
        string userText, IReadOnlyList<string> attachmentNames, IReadOnlyList<string> attachmentContents)
    {
        if (attachmentContents.Count == 0)
        {
            return userText.Trim();
        }
        var prompt = new System.Text.StringBuilder(userText.Trim());
        if (prompt.Length > 0)
        {
            prompt.AppendLine().AppendLine();
        }
        prompt.AppendLine("请参考以下文本附件内容：");
        for (var index = 0; index < attachmentContents.Count; index++)
        {
            prompt.AppendLine().Append("【附件：").Append(attachmentNames[index]).AppendLine("】");
            prompt.AppendLine(attachmentContents[index]);
        }
        return prompt.ToString();
    }
}

public sealed record ChatSendResult(
    ConversationSummary Conversation,
    AssistantResult Assistant,
    string PromptText);
