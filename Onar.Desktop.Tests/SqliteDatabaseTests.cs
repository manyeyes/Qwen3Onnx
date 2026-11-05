using Microsoft.Data.Sqlite;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Onar.Desktop.Models;
using Onar.Desktop.Services;
using Onar.Desktop.ViewModels;
using Onar.Inference;
using System.Net;
using System.Security.Cryptography;

namespace Onar.Desktop.Tests;

[TestClass]
public sealed class SqliteDatabaseTests
{
    private string _temporaryDirectory = null!;
    private SqliteDatabase _database = null!;
    private CancellationToken _cancellationToken;

    [TestInitialize]
    public async Task InitializeAsync()
    {
        _temporaryDirectory = Path.Combine(
            Path.GetTempPath(), $"qwen3-desktop-tests-{Guid.NewGuid():N}");
        _database = new SqliteDatabase(_temporaryDirectory);
        _cancellationToken = CancellationToken.None;
        var bootstrap = await _database.InitializeAsync(_cancellationToken);
        Assert.IsFalse(string.IsNullOrWhiteSpace(bootstrap.GeneratedAdminPassword));

        var admin = await _database.AuthenticateAsync(
            "admin", bootstrap.GeneratedAdminPassword!, _cancellationToken);
        Assert.IsNotNull(admin);
        Assert.AreEqual("admin", admin.Role);
        var subsequentBootstrap = await _database.InitializeAsync(_cancellationToken);
        Assert.IsNull(subsequentBootstrap.GeneratedAdminPassword);
        Assert.IsNotNull(await _database.AuthenticateAsync(
            "admin", bootstrap.GeneratedAdminPassword!, _cancellationToken));
    }

    [TestCleanup]
    public void Cleanup()
    {
        SqliteConnection.ClearAllPools();
        if (Directory.Exists(_temporaryDirectory))
        {
            Directory.Delete(_temporaryDirectory, recursive: true);
        }
    }

    [TestMethod]
    public async Task ConversationsAndMessagesAreIsolatedByOwner()
    {
        var userId = await _database.CreateUserAsync(
            "private-user", "secure-password-123", _cancellationToken);
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        var guestConversation = await SaveTurnAsync(null, model.Id, "guest");
        var privateConversation = await SaveTurnAsync(userId, model.Id, "private");

        var guestHistory = await _database.GetConversationsAsync(null, _cancellationToken);
        var privateHistory = await _database.GetConversationsAsync(userId, _cancellationToken);
        Assert.AreEqual(1, guestHistory.Count);
        Assert.AreEqual(guestConversation, guestHistory[0].Id);
        Assert.AreEqual(1, privateHistory.Count);
        Assert.AreEqual(privateConversation, privateHistory[0].Id);
        var user = (await _database.GetUsersAsync(_cancellationToken))
            .Single(account => account.Id == userId);
        Assert.AreEqual(7L, user.TokensUsed);
        Assert.IsNull(await _database.GetConversationAsync(privateConversation, null, _cancellationToken));
        Assert.IsNull(await _database.GetConversationAsync(guestConversation, userId, _cancellationToken));

        await Assert.ThrowsExceptionAsync<UnauthorizedAccessException>(() =>
            _database.SaveTurnAsync(
                guestConversation, userId, model.Id, "Unauthorized", false,
                "intrusion attempt", [], 1, "should fail", 1, _cancellationToken));
    }

    [TestMethod]
    public async Task GuestCanRenamePinAndDeleteConversation()
    {
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        var conversationId = await SaveTurnAsync(null, model.Id, "guest-manage");

        await _database.RenameConversationAsync(
                conversationId, null, "Guest renamed", _cancellationToken);
        await _database.SetConversationPinnedAsync(
                conversationId, null, true, _cancellationToken);
        var pinned = (await _database.GetConversationsAsync(null, _cancellationToken)).Single();
        Assert.AreEqual("Guest renamed", pinned.Title);
        Assert.IsTrue(pinned.IsPinned);

        await _database.DeleteConversationAsync(conversationId, null, _cancellationToken);
        Assert.AreEqual(0, (await _database.GetConversationsAsync(null, _cancellationToken)).Count);
    }

    [TestMethod]
    public async Task SavingNextTurnWithDifferentModelKeepsConversationIdentity()
    {
        var models = await _database.GetModelsAsync(_cancellationToken);
        var conversationId = await _database.SaveTurnAsync(
                null, null, models[0].Id, "Keep this chat", true,
                "first question", [], 2, "first answer", 3, _cancellationToken);

        var continuedId = await _database.SaveTurnAsync(
                conversationId, null, models[1].Id, "Keep this chat", false,
                "second question", [], 2, "second answer", 3, _cancellationToken);

        Assert.AreEqual(conversationId, continuedId);
        var updated = await _database.GetConversationAsync(conversationId, null, _cancellationToken);
        Assert.IsNotNull(updated);
        Assert.AreEqual(models[1].Id, updated.ModelId);
        Assert.AreEqual(4, (await _database.GetMessagesAsync(
                conversationId, null, _cancellationToken)).Count);
    }

    [TestMethod]
    public async Task UsersCanManageTheirOwnProfilePasswordAndAccount()
    {
        const string initialPassword = "initial-password-123";
        const string updatedPassword = "updated-password-456";
        var userId = await _database.CreateUserAsync(
            "profile-user", initialPassword, _cancellationToken);
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        await SaveTurnAsync(userId, model.Id, "profile");

        await _database.UpdateUserProfileAsync(
            userId,
            "Profile User",
            "https://example.com/avatar.png",
            "1990-01-02",
            "https://example.com/blog",
            "A short profile.",
            _cancellationToken);
        var profile = await _database.GetUserProfileAsync(userId, _cancellationToken);
        Assert.AreEqual("Profile User", profile.Nickname);
        Assert.AreEqual("1990-01-02", profile.Birthday);
        Assert.AreEqual(7L, profile.TokensUsed);

        var insights = await _database.GetOwnUserInsightsAsync(userId, _cancellationToken);
        Assert.AreEqual(1, insights.ConversationCount);
        Assert.AreEqual(2, insights.MessageCount);
        Assert.AreEqual(7L, insights.TokensUsed);

        await _database.ChangeOwnPasswordAsync(
            userId, initialPassword, updatedPassword, _cancellationToken);
        Assert.IsNull(await _database.AuthenticateAsync(
            "profile-user", initialPassword, _cancellationToken));
        Assert.IsNotNull(await _database.AuthenticateAsync(
            "profile-user", updatedPassword, _cancellationToken));

        await _database.DeleteOwnAccountAsync(userId, updatedPassword, _cancellationToken);
        Assert.IsNull(await _database.AuthenticateAsync(
            "profile-user", updatedPassword, _cancellationToken));
        await Assert.ThrowsExceptionAsync<UnauthorizedAccessException>(() =>
            _database.GetUserProfileAsync(userId, _cancellationToken));
    }

    [TestMethod]
    public async Task UserInsightsCountKeywordsAcrossQuestionsAndAnswers()
    {
        var userId = await _database.CreateUserAsync(
            "keyword-user", "secure-password-123", _cancellationToken);
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        await _database.SaveTurnAsync(
            null, userId, model.Id, "关键词统计", true,
            "人工智能", [], 3,
            "人工智能，量子计算。", 6, _cancellationToken);

        var insights = await _database.GetOwnUserInsightsAsync(userId, _cancellationToken);
        Assert.AreEqual(2, insights.Keywords.Single(keyword => keyword.Term == "人工智能").Count);
        Assert.AreEqual(1, insights.Keywords.Single(keyword => keyword.Term == "量子").Count);
    }

    [TestMethod]
    public void StreamingContentSeparatesThinkingAndAnswer()
    {
        var parsed = MessageViewModel.ParseStreamingContent(
            "<think>先分析问题</think>\n最终答案");

        Assert.AreEqual("先分析问题", parsed.Thinking);
        Assert.AreEqual("最终答案", parsed.Answer);
    }

    [TestMethod]
    public void ThinkContinuationWithoutOpeningTagIsNotMistakenForThinking()
    {
        var parsed = MessageViewModel.ParseStreamingContent(
            "先分析问题\n</think>\n最终答案");
        var message = new MessageViewModel
        {
            Role = "assistant",
            Content = "先分析问题\n</think>\n最终答案"
        };

        Assert.AreEqual(string.Empty, parsed.Thinking);
        Assert.AreEqual("先分析问题\n</think>\n最终答案", parsed.Answer);
        Assert.AreEqual(string.Empty, message.ThinkingContent);
        Assert.IsFalse(message.HasThinking);
    }

    [TestMethod]
    public void StreamingContentParserHandlesThinkMarkersSplitAcrossChunks()
    {
        var parser = new MessageViewModel.StreamingContentParser();
        parser.Append("<th");
        parser.Append("ink>分析");
        parser.Append("中</think>答");
        var result = parser.Append("案");

        Assert.AreEqual("分析中", result.Thinking);
        Assert.AreEqual("答案", result.Answer);
    }

    [TestMethod]
    public void ThinkTagStringsInsideAnswerRemainOrdinaryText()
    {
        const string output = "Use <think> and </think> as literal template tokens in this example.";
        var parsed = MessageViewModel.ParseStreamingContent(output);
        var parser = new MessageViewModel.StreamingContentParser();
        parser.Append("Use <think> and ");
        var streamed = parser.Append("</think> as literal template tokens in this example.");
        var message = new MessageViewModel
        {
            Role = "assistant",
            Content = output
        };

        Assert.AreEqual(string.Empty, parsed.Thinking);
        Assert.AreEqual(output, parsed.Answer);
        Assert.AreEqual(string.Empty, streamed.Thinking);
        Assert.AreEqual(output, streamed.Answer);
        Assert.IsFalse(message.HasThinking);
        Assert.AreEqual(output, message.MarkdownContent);
    }

    [TestMethod]
    public void StreamingSnapshotsDoNotReplaceMarkdownContent()
    {
        var message = new MessageViewModel { IsPending = true };
        var markdownNotifications = 0;
        message.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(MessageViewModel.MarkdownContent))
            {
                markdownNotifications++;
            }
        };

        message.ApplyStreamingSnapshot("分析", "回答");

        Assert.AreEqual(string.Empty, message.Content);
        Assert.AreEqual("回答", message.StreamingAnswer);
        Assert.AreEqual(0, markdownNotifications);

        message.Content = "<think>分析</think>回答";
        message.IsPending = false;

        Assert.AreEqual(1, markdownNotifications);
        Assert.AreEqual("回答", message.MarkdownContent);
    }

    [TestMethod]
    public async Task GuestCanRegisterAndOpenUserCenter()
    {
        using var services = new DesktopServices(_temporaryDirectory);
        var shell = new MainWindowViewModel(services, new BootstrapResult(null));
        await shell.Initialization;
        shell.LoginUsername = "registered-user";
        shell.RegisterPassword = "registered-password-123";
        shell.RegisterConfirmPassword = shell.RegisterPassword;
        shell.IsRegisterMode = true;

        await shell.RegisterCommand.ExecuteAsync(null);

        Assert.IsNotNull(shell.CurrentUser);
        Assert.AreEqual("registered-user", shell.CurrentUser.Username);
        Assert.AreEqual("user", shell.CurrentUser.Role);
        shell.Navigate("Profile");
        await shell.Profile.RefreshAsync(_cancellationToken);
        Assert.AreSame(shell.Profile, shell.CurrentViewModel);
        Assert.AreEqual("registered-user", shell.Profile.Profile?.Username);
    }

    [TestMethod]
    public async Task AboutPageCanBeOpenedFromBottomMenu()
    {
        using var services = new DesktopServices(_temporaryDirectory);
        var shell = new MainWindowViewModel(services, new BootstrapResult(null));
        await shell.Initialization;
        shell.ToggleBottomMenuCommand.Execute(null);

        Assert.IsTrue(shell.IsBottomMenuOpen);
        shell.OpenAboutPageCommand.Execute(null);

        Assert.AreSame(shell.About, shell.CurrentViewModel);
        Assert.IsFalse(shell.IsBottomMenuOpen);
        Assert.IsFalse(string.IsNullOrWhiteSpace(shell.About.Version));
        Assert.IsFalse(string.IsNullOrWhiteSpace(shell.About.RuntimeVersion));
    }

    [TestMethod]
    public async Task TurnAndUsageAreCommittedTogether()
    {
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        var before = await _database.GetConversationsAsync(null, _cancellationToken);

        await Assert.ThrowsExceptionAsync<SqliteException>(() =>
            _database.SaveTurnAsync(
                null, null, "missing-model", "Invalid", true,
                "question", [], 2, "answer", 3, _cancellationToken));

        var after = await _database.GetConversationsAsync(null, _cancellationToken);
        Assert.AreEqual(before.Count, after.Count, "A failed transaction must not leave an empty conversation.");

        var conversationId = await SaveTurnAsync(null, model.Id, "atomic");
        var messages = await _database.GetMessagesAsync(conversationId, null, _cancellationToken);
        Assert.AreEqual(2, messages.Count);
        Assert.AreEqual("user", messages[0].Role);
        Assert.AreEqual("assistant", messages[1].Role);
        Assert.AreEqual("atomic.txt", messages[0].Attachments.Single());
    }

    [TestMethod]
    public async Task InterruptedTurnIsPersistedAndMarkedInHistory()
    {
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        var conversationId = await _database.SaveTurnAsync(
            null, null, model.Id, "Interrupted turn", true,
            "question in progress", [], 4, "partial answer", 2,
            _cancellationToken, wasInterrupted: true,
            generationTimeSeconds: 1.25, stopReason: "用户停止生成");

        var messages = await _database.GetMessagesAsync(
            conversationId, null, _cancellationToken);
        Assert.AreEqual(2, messages.Count);
        Assert.IsFalse(messages[0].WasInterrupted);
        Assert.IsTrue(messages[1].WasInterrupted);
        Assert.AreEqual("partial answer", messages[1].Content);
        Assert.AreEqual(1.25, messages[1].GenerationTimeSeconds, 0.001);
        Assert.AreEqual("用户停止生成", messages[1].StopReason);
    }

    [TestMethod]
    public async Task MessagesCanBeDeletedOnlyByTheirConversationOwner()
    {
        var userId = await _database.CreateUserAsync(
            "delete-message-user", "secure-password-123", _cancellationToken);
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        var conversationId = await SaveTurnAsync(userId, model.Id, "delete");
        var messages = await _database.GetMessagesAsync(conversationId, userId, _cancellationToken);

        await Assert.ThrowsExceptionAsync<UnauthorizedAccessException>(() =>
            _database.DeleteMessageAsync(
                conversationId, messages[0].Id, null, _cancellationToken));
        await _database.DeleteMessageAsync(
            conversationId, messages[0].Id, userId, _cancellationToken);

        var remaining = await _database.GetMessagesAsync(conversationId, userId, _cancellationToken);
        Assert.AreEqual(1, remaining.Count);
        Assert.AreEqual(messages[1].Id, remaining[0].Id);
    }

    [TestMethod]
    public async Task RegeneratedAssistantReplyReplacesOnlyLatestReply()
    {
        var userId = await _database.CreateUserAsync(
            "regenerate-user", "secure-password-123", _cancellationToken);
        var model = (await _database.GetModelsAsync(_cancellationToken))[0];
        var conversationId = await SaveTurnAsync(userId, model.Id, "regenerate");
        var messages = await _database.GetMessagesAsync(conversationId, userId, _cancellationToken);
        var assistantMessage = messages.Last();

        await _database.ReplaceAssistantReplyAsync(
            conversationId, userId, assistantMessage.Id, "replacement answer", 5, 8,
            _cancellationToken, generationTimeSeconds: 2.5, stopReason: "达到最大生成长度");

        var updated = await _database.GetMessagesAsync(conversationId, userId, _cancellationToken);
        Assert.AreEqual(2, updated.Count);
        Assert.AreEqual(assistantMessage.Id, updated[1].Id);
        Assert.AreEqual("replacement answer", updated[1].Content);
        Assert.AreEqual(8, updated[1].TokenCount);
        Assert.AreEqual(2.5, updated[1].GenerationTimeSeconds, 0.001);
        Assert.AreEqual("达到最大生成长度", updated[1].StopReason);
        await Assert.ThrowsExceptionAsync<InvalidOperationException>(() =>
            _database.ReplaceAssistantReplyAsync(
                conversationId, null, assistantMessage.Id, "unauthorized", 1, 1,
                _cancellationToken));
    }

    [TestMethod]
    public void MessageViewModelSeparatesThinkingPreviewAndMarkdownAnswer()
    {
        var message = new MessageViewModel
        {
            Role = "assistant",
            Content = "<think>step one\nstep two</think>\n### Answer\n\n**42**"
        };

        Assert.AreEqual("step one\nstep two", message.ThinkingContent);
        Assert.AreEqual("### Answer\n\n**42**", message.MarkdownContent);
        Assert.IsTrue(message.HasThinking);
        Assert.AreEqual(112, message.ThinkingMaxHeight);
        message.IsThinkingExpanded = true;
        Assert.AreEqual(240, message.ThinkingMaxHeight);
    }

    [TestMethod]
    public void MessageViewModelKeepsThinkingVisibleAfterGenerationCompletes()
    {
        var message = new MessageViewModel
        {
            Role = "assistant",
            IsPending = true
        };
        var changedProperties = new HashSet<string?>();
        message.PropertyChanged += (_, args) => changedProperties.Add(args.PropertyName);
        message.ApplyStreamingSnapshot("reasoning", "answer");
        message.Content = "<think>reasoning</think>answer";

        message.IsPending = false;

        Assert.AreEqual("reasoning", message.ThinkingContent);
        Assert.IsTrue(message.HasThinking);
        Assert.AreEqual("answer", message.MarkdownContent);
        Assert.IsTrue(changedProperties.Contains(nameof(MessageViewModel.ThinkingContent)));
        Assert.IsTrue(changedProperties.Contains(nameof(MessageViewModel.HasThinking)));
        Assert.IsTrue(changedProperties.Contains(nameof(MessageViewModel.MarkdownContent)));
    }

    [TestMethod]
    public void ChatAssistantRestoresPreviousTurnsWhenFactQuestionResetIsDisabled()
    {
        var factory = new TestInferenceEngineFactory();
        using var assistant = new ChatAssistant(
            new AssistantConfig { EngineId = "test" }, factory);
        var config = new GenerationConfig
        {
            ThinkMode = false,
            ThinkingBudget = 0,
            MultiTurn = true,
            MaxTurns = 3,
            MaxHistoryChars = 400,
            MaxHistoryTokens = 1024,
            ResetOnFactQuestion = false,
            EnableRepetitionCheck = false,
            EnableLoopDetection = false
        };

        assistant.Ask("我最喜欢蓝色", generationConfig: config);
        var followUp = assistant.Ask("这是什么", generationConfig: config);

        Assert.AreEqual(2, followUp.HistoryTurns);
        StringAssert.Contains(factory.Engine.LastPrefill, "我最喜欢蓝色");
        StringAssert.Contains(factory.Engine.LastPrefill, "test answer");
    }

    [TestMethod]
    public void ChatAssistantDoesNotAddTurnWhenGenerationIsCancelled()
    {
        var factory = new TestInferenceEngineFactory();
        using var assistant = new ChatAssistant(
            new AssistantConfig { EngineId = "test" }, factory);
        var config = new GenerationConfig
        {
            ThinkMode = false,
            ThinkingBudget = 0,
            MultiTurn = true,
            MaxTurns = 3,
            MaxHistoryChars = 400,
            MaxHistoryTokens = 1024,
            ResetOnFactQuestion = false,
            EnableRepetitionCheck = false,
            EnableLoopDetection = false
        };
        assistant.Ask("first question", generationConfig: config);
        using var cancellation = new CancellationTokenSource();

        Assert.ThrowsException<OperationCanceledException>(() => assistant.Ask(
            "cancelled question",
            onToken: _ => cancellation.Cancel(),
            generationConfig: config,
            cancellationToken: cancellation.Token));

        Assert.AreEqual(1, assistant.History.Count);
        Assert.AreEqual("first question", assistant.History[0].User);
    }

    [TestMethod]
    public void ChatAssistantKeepsInterruptedPartialAnswerInHistory()
    {
        var factory = new TestInferenceEngineFactory();
        factory.Engine.ReturnInterruptedGeneration = true;
        using var assistant = new ChatAssistant(
            new AssistantConfig { EngineId = "test" }, factory);
        var config = new GenerationConfig
        {
            ThinkMode = false,
            ThinkingBudget = 0,
            MultiTurn = true,
            MaxTurns = 3,
            MaxHistoryChars = 400,
            MaxHistoryTokens = 1024,
            ResetOnFactQuestion = false,
            EnableRepetitionCheck = false,
            EnableLoopDetection = false
        };

        var result = assistant.Ask("interrupted question", generationConfig: config);

        Assert.IsTrue(result.Generation.WasCancelled);
        Assert.AreEqual(1, assistant.History.Count);
        Assert.AreEqual("test answer", assistant.History[0].Assistant);
    }

    [TestMethod]
    public async Task RefreshingModelsDoesNotReplaceLoadedConversationWithNewChat()
    {
        using var services = new DesktopServices(_temporaryDirectory);
        await services.Database.InitializeAsync(_cancellationToken);
        var models = await services.Database.GetModelsAsync(_cancellationToken);
        var standardModels = models.Where(item => !item.RequiresGenAi).Take(2).ToArray();
        Assert.AreEqual(2, standardModels.Length);
        var model = standardModels[0];
        var alternativeModel = standardModels[1];
        foreach (var installedModel in new[] { model, alternativeModel })
        {
            Directory.CreateDirectory(installedModel.LocalPath);
            await File.WriteAllTextAsync(
                Path.Combine(installedModel.LocalPath, "config.json"), "{}", _cancellationToken);
            await File.WriteAllTextAsync(
                Path.Combine(installedModel.LocalPath, "tokenizer.json"), "{}", _cancellationToken);
            await File.WriteAllBytesAsync(
                Path.Combine(installedModel.LocalPath, installedModel.EmbeddingFileName), [1]);
            await File.WriteAllBytesAsync(
                Path.Combine(installedModel.LocalPath, installedModel.DecoderFileName), [2]);
            foreach (var file in installedModel.HttpFileLinks)
            {
                var filePath = Path.Combine(installedModel.LocalPath, file.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
                await File.WriteAllBytesAsync(filePath, [3]);
            }
        }

        var conversationId = await services.Database.SaveTurnAsync(
            null, null, model.Id, "Saved conversation", true,
            "previous question", [], 2, "previous answer", 3, _cancellationToken);
        var conversation = await services.Database.GetConversationAsync(
            conversationId, null, _cancellationToken);
        Assert.IsNotNull(conversation);

        var shell = new MainWindowViewModel(services, new BootstrapResult(null));
        await shell.Initialization;
        await shell.Chat.LoadConversationAsync(conversation, null, _cancellationToken);
        await shell.Chat.RefreshModelsAsync(_cancellationToken);
        shell.Chat.SelectedModel = shell.Chat.InstalledModels.Single(
            item => item.Id == alternativeModel.Id);

        Assert.AreEqual(conversationId, shell.Chat.ConversationId);
        Assert.AreEqual(2, shell.Chat.Messages.Count);
        Assert.IsFalse(shell.Chat.HasNoMessages);
        StringAssert.Contains(shell.Chat.StatusText, "继续保存在当前对话");
    }

    [TestMethod]
    public async Task LockedAccountCannotAuthenticate()
    {
        var userId = await _database.CreateUserAsync(
            "locked-user", "secure-password-123", _cancellationToken);
        Assert.IsNotNull(await _database.AuthenticateAsync(
            "locked-user", "secure-password-123", _cancellationToken));

        await _database.SetUserLockedAsync(userId, true, _cancellationToken);

        Assert.IsNull(await _database.AuthenticateAsync(
            "locked-user", "secure-password-123", _cancellationToken));
    }

    [TestMethod]
    public async Task LoginSessionPersistsAcrossRestartsAndExpiresOrClears()
    {
        var userId = await _database.CreateUserAsync(
            "remembered-user", "secure-password-123", _cancellationToken);
        var expiry = DateTimeOffset.UtcNow.AddDays(30);

        await _database.SaveLoginSessionAsync(userId, expiry, _cancellationToken);
        var restartedDatabase = new SqliteDatabase(_temporaryDirectory);
        var restored = await restartedDatabase.RestoreLoginSessionAsync(_cancellationToken);
        Assert.IsNotNull(restored);
        Assert.AreEqual(userId, restored.Id);
        Assert.AreEqual("remembered-user", restored.Username);

        await _database.SaveLoginSessionAsync(
            userId, DateTimeOffset.UtcNow.AddMinutes(-1), _cancellationToken);
        Assert.IsNull(await _database.RestoreLoginSessionAsync(_cancellationToken));

        await _database.SaveLoginSessionAsync(userId, expiry, _cancellationToken);
        await _database.SetUserLockedAsync(userId, true, _cancellationToken);
        Assert.IsNull(await _database.RestoreLoginSessionAsync(_cancellationToken));

        await _database.SetUserLockedAsync(userId, false, _cancellationToken);
        await _database.SaveLoginSessionAsync(userId, expiry, _cancellationToken);
        await _database.ClearLoginSessionAsync(_cancellationToken);
        Assert.IsNull(await _database.RestoreLoginSessionAsync(_cancellationToken));
    }

    [TestMethod]
    public async Task BuiltInModelsHaveCompleteModelScopeHttpFallbacks()
    {
        var models = await _database.GetModelsAsync(_cancellationToken);
        Assert.IsTrue(models.Count >= 8);

        foreach (var model in models)
        {
            if (model.RequiresGenAi)
            {
                var genAiFiles = model.EngineId switch
                {
                    "phi-onnx-genai" => new HashSet<string>(
                        ["added_tokens.json", "chat_template.jinja", "genai_config.json",
                            "merges.txt", "model.onnx", "model.onnx.data", "special_tokens_map.json",
                            "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                        StringComparer.OrdinalIgnoreCase),
                    "deepseek-onnx-genai" => new HashSet<string>(
                        ["chat_template.jinja", "genai_config.json", "model.onnx",
                            "model.onnx.data", "special_tokens_map.json", "tokenizer.json",
                            "tokenizer_config.json"],
                        StringComparer.OrdinalIgnoreCase),
                    _ => new HashSet<string>(
                        ["chat_template.jinja", "genai_config.json", "model.onnx",
                            "model.onnx.data", "tokenizer.json", "tokenizer_config.json"],
                        StringComparer.OrdinalIgnoreCase)
                };
                Assert.IsTrue(
                    model.EngineId is "qwen3-onnx-genai" or "llama-onnx-genai" or
                        "phi-onnx-genai" or "deepseek-onnx-genai");
                Assert.IsTrue(genAiFiles.SetEquals(model.HttpFileLinks.Select(file => file.Path)));
                foreach (var file in model.HttpFileLinks)
                {
                    Assert.AreEqual(
                        $"{model.RepositoryUrl}/resolve/master/{file.Path}?view=false",
                        file.Url);
                }
                continue;
            }

            var expectedFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "config.json", "tokenizer.json", "embedding.int8.onnx",
                model.EnglishName.Contains("w4a32", StringComparison.OrdinalIgnoreCase)
                    ? "decoder.int4.onnx"
                    : "decoder.int8.onnx",
                "vocab.json",
                model.EnglishName.Contains("w4a32", StringComparison.OrdinalIgnoreCase)
                    ? "decoder.int4.onnx.data"
                    : "decoder.int8.onnx.data"
            };

            Assert.IsTrue(expectedFiles.SetEquals(model.HttpFileLinks.Select(file => file.Path)));
            foreach (var file in model.HttpFileLinks)
            {
                Assert.AreEqual(
                    $"{model.RepositoryUrl}/resolve/master/{file.Path}?view=false",
                    file.Url);
            }
        }
    }

    [TestMethod]
    public async Task ExistingLlamaHttpFallbacksAreUpdatedToTheRepositoryOwner()
    {
        var llama = (await _database.GetModelsAsync(_cancellationToken))
            .Single(model => model.IsLlamaGenAi &&
                model.EnglishName.Contains("1B", StringComparison.OrdinalIgnoreCase));
        var staleLinks = llama.HttpFileLinks.Select(link => link with
        {
            Url = link.Url.Replace("onnx-community", "manyeyes", StringComparison.Ordinal)
        }).ToArray();

        await using (var connection = new SqliteConnection($"Data Source={_database.DatabasePath}"))
        {
            await connection.OpenAsync(_cancellationToken);
            await using var command = connection.CreateCommand();
            command.CommandText =
                "UPDATE models SET direct_downloads_json = $links WHERE id = $id;";
            command.Parameters.AddWithValue("$links", System.Text.Json.JsonSerializer.Serialize(staleLinks));
            command.Parameters.AddWithValue("$id", llama.Id);
            await command.ExecuteNonQueryAsync(_cancellationToken);
        }

        await _database.InitializeAsync(_cancellationToken);
        var refreshed = (await _database.GetModelsAsync(_cancellationToken))
            .Single(model => model.Id == llama.Id);

        Assert.IsTrue(refreshed.HttpFileLinks.All(file =>
            file.Url.StartsWith(
                $"{refreshed.RepositoryUrl}/resolve/master/",
                StringComparison.Ordinal)));
    }

    [TestMethod]
    public async Task ExistingModelCatalogGetsInferenceEngineColumnMigration()
    {
    var legacyDirectory = Path.Combine(_temporaryDirectory, "legacy-engine-db");
    Directory.CreateDirectory(legacyDirectory);
    var legacyPath = Path.Combine(legacyDirectory, "qwen3-desktop.sqlite3");
    await using (var connection = new SqliteConnection($"Data Source={legacyPath}"))
    {
        await connection.OpenAsync(_cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            CREATE TABLE models (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                english_name TEXT NOT NULL,
                description TEXT NOT NULL,
                repository_url TEXT NOT NULL,
                local_path TEXT NOT NULL,
                is_installed INTEGER NOT NULL DEFAULT 0,
                embedding_file_name TEXT NOT NULL DEFAULT 'embedding.int8.onnx',
                decoder_file_name TEXT NOT NULL DEFAULT 'decoder.int8.onnx',
                required_files_json TEXT NOT NULL DEFAULT '[]',
                direct_downloads_json TEXT NOT NULL DEFAULT '[]',
                created_utc TEXT NOT NULL
            );
            """;
        await command.ExecuteNonQueryAsync(_cancellationToken);
    }

    var legacyDatabase = new SqliteDatabase(legacyDirectory);
    await legacyDatabase.InitializeAsync(_cancellationToken);
    var upgradedModels = await legacyDatabase.GetModelsAsync(_cancellationToken);
    Assert.IsTrue(upgradedModels.Count >= 8);
    Assert.IsTrue(upgradedModels.Any(model =>
        model.EngineId == "qwen3-onnx-genai" && model.RequiresGenAi));
    }

    [TestMethod]
    public void InstalledW4A32ModelRequiresItsExternalDecoderWeights()
    {
        var modelPath = Path.Combine(_temporaryDirectory, "w4a32-installation");
        Directory.CreateDirectory(modelPath);
        foreach (var fileName in new[]
                 {
                     "config.json", "tokenizer.json", "embedding.int8.onnx", "decoder.int4.onnx"
                 })
        {
            File.WriteAllText(Path.Combine(modelPath, fileName), "model");
        }
        var model = new ModelDefinition(
            "w4a32", "Qwen3 0.6B W4A32", "Qwen3-0.6B-w4a32-onnx", string.Empty,
            string.Empty, modelPath, false, "embedding.int8.onnx", "decoder.int4.onnx", [],
            [new ModelFileLink("decoder.int4.onnx.data", "https://example.com/decoder.int4.onnx.data")]);

        Assert.IsFalse(ModelDownloadService.IsInstalled(model));

        File.WriteAllText(Path.Combine(modelPath, "decoder.int4.onnx.data"), "weights");
        Assert.IsTrue(ModelDownloadService.IsInstalled(model));
    }

    [TestMethod]
    public async Task ModelCatalogSupportsDirectLinksAndConversationRenameIsOwnerScoped()
    {
        var userId = await _database.CreateUserAsync(
            "model-user", "secure-password-123", _cancellationToken);
        var directLinks = new[]
        {
            new ModelFileLink("config.json", "https://cdn.example/model/config.json"),
            new ModelFileLink("tokenizer.json", "https://cdn.example/model/tokenizer.json"),
            new ModelFileLink("embedding.int8.onnx", "https://cdn.example/model/embedding.onnx"),
            new ModelFileLink("decoder.int8.onnx", "https://cdn.example/model/decoder.onnx")
        };
        await _database.RegisterModelAsync(
            "Direct Model", "DirectModel", "Direct download test", string.Empty,
            directLinks, _cancellationToken);
        var model = (await _database.GetModelsAsync(_cancellationToken))
            .Single(item => item.EnglishName == "DirectModel");
        Assert.AreEqual(4, model.HttpFileLinks.Count);
        Assert.AreEqual(string.Empty, model.RepositoryUrl);
        await _database.UpdateModelAsync(
            model.Id, "Edited Direct Model", "DirectModel", "Edited description",
            string.Empty, directLinks, _cancellationToken);
        var editedModel = (await _database.GetModelsAsync(_cancellationToken))
            .Single(item => item.Id == model.Id);
        Assert.AreEqual("Edited Direct Model", editedModel.Title);
        Assert.AreEqual("Edited description", editedModel.Description);

        var conversationId = await SaveTurnAsync(userId, editedModel.Id, "renamable");
        await Assert.ThrowsExceptionAsync<InvalidOperationException>(() =>
            _database.UpdateModelAsync(
                editedModel.Id, editedModel.Title, "RenamedModel", editedModel.Description,
                editedModel.RepositoryUrl, editedModel.HttpFileLinks, _cancellationToken));
        await _database.RenameConversationAsync(
            conversationId, userId, "Updated title", _cancellationToken);
        Assert.AreEqual("Updated title",
            (await _database.GetConversationAsync(conversationId, userId, _cancellationToken))?.Title);
        await Assert.ThrowsExceptionAsync<UnauthorizedAccessException>(() =>
            _database.RenameConversationAsync(
                conversationId, null, "Unauthorized title", _cancellationToken));
    }

    [TestMethod]
    public async Task DownloaderUsesPerFileHttpFallbackAndValidatesHashes()
    {
        var payloads = new Dictionary<string, byte[]>(StringComparer.Ordinal)
        {
            ["https://cdn.example/config.json"] = """{"model_type":"qwen3"}"""u8.ToArray(),
            ["https://cdn.example/tokenizer.json"] = """{"version":"1"}"""u8.ToArray(),
            ["https://cdn.example/embedding.onnx"] = [1, 2, 3, 4],
            ["https://cdn.example/decoder.onnx"] = [5, 6, 7, 8, 9]
        };
        var links = payloads.Select(pair =>
        {
            var fileName = pair.Key.Split('/').Last();
            var path = fileName switch
            {
                "embedding.onnx" => "embedding.int8.onnx",
                "decoder.onnx" => "decoder.int8.onnx",
                _ => fileName
            };
            var hash = Convert.ToHexString(SHA256.HashData(pair.Value));
            return new ModelFileLink(path, pair.Key, pair.Value.Length, hash);
        }).ToArray();
        using var httpClient = new HttpClient(new InMemoryDownloadHandler(payloads));
        var downloader = new ModelDownloadService(httpClient);
        var destination = Path.Combine(_temporaryDirectory, "downloaded-model");
        var model = new ModelDefinition(
            "direct-http", "Direct HTTP", "DirectHttp", "Test direct file fallback",
            string.Empty, destination, false, "embedding.int8.onnx", "decoder.int8.onnx", [], links);

        var downloaded = await downloader.DownloadAsync(model, cancellationToken: _cancellationToken);

        Assert.IsTrue(downloaded.Model.IsInstalled);
        Assert.IsNull(downloaded.CleanupWarning);
        Assert.IsTrue(ModelDownloadService.IsInstalled(downloaded.Model));
        Assert.AreEqual(4, downloaded.Model.RequiredFiles.Count);
        Assert.IsTrue(File.Exists(Path.Combine(destination, "decoder.int8.onnx")));
    }

    [TestMethod]
    public async Task DownloaderInstallsGenAiModelUsingItsDistinctFileManifest()
    {
        var fileNames = new[]
        {
            "chat_template.jinja",
            "genai_config.json",
            "model.onnx",
            "model.onnx.data",
            "tokenizer.json",
            "tokenizer_config.json"
        };
        var payloads = fileNames.ToDictionary(
            fileName => $"https://cdn.example/genai/{fileName}",
            fileName => System.Text.Encoding.UTF8.GetBytes(fileName),
            StringComparer.Ordinal);
        var links = fileNames.Select(fileName =>
            new ModelFileLink(fileName, $"https://cdn.example/genai/{fileName}")).ToArray();
        using var httpClient = new HttpClient(new InMemoryDownloadHandler(payloads));
        var downloader = new ModelDownloadService(httpClient);
        var destination = Path.Combine(_temporaryDirectory, "downloaded-genai-model");
        var model = new ModelDefinition(
            "genai-http", "GenAI", "GenAI", "Test GenAI file fallback",
            string.Empty, destination, false, string.Empty, string.Empty, [], links, "qwen3-onnx-genai");

        var downloaded = await downloader.DownloadAsync(model, cancellationToken: _cancellationToken);

        Assert.IsTrue(downloaded.Model.IsInstalled);
        Assert.IsTrue(downloaded.Model.RequiresGenAi);
        Assert.AreEqual(string.Empty, downloaded.Model.EmbeddingFileName);
        Assert.AreEqual(string.Empty, downloaded.Model.DecoderFileName);
        Assert.IsTrue(new HashSet<string>(fileNames, StringComparer.OrdinalIgnoreCase)
            .SetEquals(downloaded.Model.RequiredFiles));
        Assert.IsTrue(ModelDownloadService.IsInstalled(downloaded.Model));
    }

    [TestMethod]
    public async Task DownloaderInstallsLlamaWithJinjaTemplate()
    {
        var files = new Dictionary<string, byte[]>(StringComparer.Ordinal)
        {
            ["genai_config.json"] = "{}"u8.ToArray(),
            ["model.onnx"] = [1],
            ["model.onnx.data"] = [2],
            ["tokenizer.json"] = "{}"u8.ToArray(),
            ["chat_template.jinja"] = System.Text.Encoding.UTF8.GetBytes(
                "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"),
            ["tokenizer_config.json"] =
                """{"backend":"tokenizers","bos_token":"<|begin_of_text|>","clean_up_tokenization_spaces":true,"eos_token":"<|end_of_text|>","is_local":true,"model_input_names":["input_ids","attention_mask"],"model_max_length":131072,"tokenizer_class":"TokenizersBackend"}"""u8.ToArray()
        };
        var links = files.Select(pair =>
            new ModelFileLink(pair.Key, $"https://cdn.example/llama/{pair.Key}")).ToArray();
        var payloads = links.ToDictionary(
            link => link.Url, link => files[link.Path], StringComparer.Ordinal);
        using var httpClient = new HttpClient(new InMemoryDownloadHandler(payloads));
        var downloader = new ModelDownloadService(httpClient);
        var destination = Path.Combine(_temporaryDirectory, "downloaded-llama-model");
        var model = new ModelDefinition(
            "llama-http", "Llama", "Llama", "Test Llama GenAI file fallback",
            string.Empty, destination, false, string.Empty, string.Empty, [], links, "llama-onnx-genai");

        var downloaded = await downloader.DownloadAsync(model, cancellationToken: _cancellationToken);

        Assert.IsTrue(downloaded.Model.IsInstalled);
        Assert.IsTrue(downloaded.Model.IsLlamaGenAi);
        Assert.IsTrue(downloaded.Model.RequiredFiles.Contains("chat_template.jinja"));
        Assert.IsTrue(ModelDownloadService.IsInstalled(downloaded.Model));
    }

    [TestMethod]
    public async Task DownloaderInstallsPhiAndDeepSeekGenAiManifests()
    {
        var manifests = new Dictionary<string, string[]>(StringComparer.Ordinal)
        {
            ["phi-onnx-genai"] =
            [
                "added_tokens.json", "chat_template.jinja", "genai_config.json", "merges.txt",
                "model.onnx", "model.onnx.data", "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json"
            ],
            ["deepseek-onnx-genai"] =
            [
                "chat_template.jinja", "genai_config.json", "model.onnx", "model.onnx.data",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"
            ]
        };

        foreach (var (engineId, fileNames) in manifests)
        {
            var payloads = fileNames.ToDictionary(
                fileName => $"https://cdn.example/{engineId}/{fileName}",
                fileName => System.Text.Encoding.UTF8.GetBytes(fileName),
                StringComparer.Ordinal);
            var links = fileNames.Select(fileName =>
                new ModelFileLink(fileName, $"https://cdn.example/{engineId}/{fileName}")).ToArray();
            using var httpClient = new HttpClient(new InMemoryDownloadHandler(payloads));
            var downloader = new ModelDownloadService(httpClient);
            var destination = Path.Combine(_temporaryDirectory, engineId);
            var model = new ModelDefinition(
                engineId, engineId, engineId, $"Test {engineId} download",
                string.Empty, destination, false, string.Empty, string.Empty, [], links, engineId);

            var downloaded = await downloader.DownloadAsync(model, cancellationToken: _cancellationToken);

            Assert.IsTrue(downloaded.Model.IsInstalled);
            Assert.IsTrue(downloaded.Model.RequiresGenAi);
            Assert.IsTrue(ModelDownloadService.IsInstalled(downloaded.Model));
            Assert.IsTrue(new HashSet<string>(fileNames, StringComparer.OrdinalIgnoreCase)
                .SetEquals(downloaded.Model.RequiredFiles));
        }
    }

    [TestMethod]
    public async Task DownloaderPreservesPartialFilesWhenHttpFallbackFails()
    {
        var config = """{"model_type":"qwen3"}"""u8.ToArray();
        var links = new[]
        {
            new ModelFileLink("config.json", "https://cdn.example/config.json", config.Length),
            new ModelFileLink("tokenizer.json", "https://cdn.example/missing-tokenizer.json"),
            new ModelFileLink("embedding.int8.onnx", "https://cdn.example/embedding.onnx"),
            new ModelFileLink("decoder.int8.onnx", "https://cdn.example/decoder.onnx")
        };
        using var httpClient = new HttpClient(new InMemoryDownloadHandler(
            new Dictionary<string, byte[]>(StringComparer.Ordinal)
            {
                ["https://cdn.example/config.json"] = config
            }));
        var downloader = new ModelDownloadService(httpClient);
        var destination = Path.Combine(_temporaryDirectory, "incomplete-model");
        var model = new ModelDefinition(
            "partial-http", "Partial HTTP", "PartialHttp", "Partial download test",
            string.Empty, destination, false, "embedding.int8.onnx", "decoder.int8.onnx", [], links);

        var exception = await Assert.ThrowsExceptionAsync<IOException>(() =>
            downloader.DownloadAsync(model, cancellationToken: _cancellationToken));

        StringAssert.Contains(exception.Message, ".partial");
        Assert.IsFalse(Directory.Exists(destination));
        var stagingDirectories = Directory.GetDirectories(_temporaryDirectory, "*.partial");
        Assert.AreEqual(1, stagingDirectories.Length);
        Assert.IsTrue(File.Exists(Path.Combine(stagingDirectories[0], "config.json")));
    }

    [TestMethod]
    public async Task ExistingModelCatalogGetsDirectDownloadColumnMigration()
    {
        var legacyDirectory = Path.Combine(_temporaryDirectory, "legacy-db");
        Directory.CreateDirectory(legacyDirectory);
        var legacyPath = Path.Combine(legacyDirectory, "qwen3-desktop.sqlite3");
        await using (var connection = new SqliteConnection($"Data Source={legacyPath}"))
        {
            await connection.OpenAsync(_cancellationToken);
            await using var command = connection.CreateCommand();
            command.CommandText = """
                CREATE TABLE models (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    english_name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    description TEXT NOT NULL,
                    repository_url TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    is_installed INTEGER NOT NULL DEFAULT 0,
                    embedding_file_name TEXT NOT NULL DEFAULT 'embedding.int8.onnx',
                    decoder_file_name TEXT NOT NULL DEFAULT 'decoder.int8.onnx',
                    required_files_json TEXT NOT NULL DEFAULT '[]',
                    created_utc TEXT NOT NULL
                );
                """;
            await command.ExecuteNonQueryAsync(_cancellationToken);
        }

        var legacyDatabase = new SqliteDatabase(legacyDirectory);
        await legacyDatabase.InitializeAsync(_cancellationToken);
        var models = await legacyDatabase.GetModelsAsync(_cancellationToken);
        Assert.IsTrue(models.Count >= 8);
        Assert.IsTrue(models.All(model =>
            model.HttpFileLinks.Count >= (model.RequiresGenAi && model.IsLlamaGenAi ? 5 : 6)));
    }

    [TestMethod]
    public async Task ExistingMessageTableGetsGenerationMetadataMigrations()
    {
        var legacyDirectory = Path.Combine(_temporaryDirectory, "legacy-message-db");
        Directory.CreateDirectory(legacyDirectory);
        var legacyPath = Path.Combine(legacyDirectory, "qwen3-desktop.sqlite3");
        await using (var connection = new SqliteConnection($"Data Source={legacyPath}"))
        {
            await connection.OpenAsync(_cancellationToken);
            await using var command = connection.CreateCommand();
            command.CommandText = """
                CREATE TABLE messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_utc TEXT NOT NULL
                );
                """;
            await command.ExecuteNonQueryAsync(_cancellationToken);
        }

        var legacyDatabase = new SqliteDatabase(legacyDirectory);
        await legacyDatabase.InitializeAsync(_cancellationToken);
        await using var migratedConnection = new SqliteConnection($"Data Source={legacyPath}");
        await migratedConnection.OpenAsync(_cancellationToken);
        await using var columns = migratedConnection.CreateCommand();
        columns.CommandText = "PRAGMA table_info(messages);";
        await using var reader = await columns.ExecuteReaderAsync(_cancellationToken);
        var hasInterruptedColumn = false;
        var hasGenerationTimeColumn = false;
        var hasStopReasonColumn = false;
        while (await reader.ReadAsync(_cancellationToken))
        {
            hasInterruptedColumn |= reader.GetString(1) == "was_interrupted";
            hasGenerationTimeColumn |= reader.GetString(1) == "generation_time_seconds";
            hasStopReasonColumn |= reader.GetString(1) == "stop_reason";
        }
        Assert.IsTrue(hasInterruptedColumn);
        Assert.IsTrue(hasGenerationTimeColumn);
        Assert.IsTrue(hasStopReasonColumn);
    }

    [TestMethod]
    public async Task ExistingConversationTableGetsPinnedStateMigration()
    {
        var legacyDirectory = Path.Combine(_temporaryDirectory, "legacy-conversation-db");
        Directory.CreateDirectory(legacyDirectory);
        var legacyPath = Path.Combine(legacyDirectory, "qwen3-desktop.sqlite3");
        await using (var connection = new SqliteConnection($"Data Source={legacyPath}"))
        {
            await connection.OpenAsync(_cancellationToken);
            await using var command = connection.CreateCommand();
            command.CommandText = """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    owner_user_id TEXT NULL,
                    model_id TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                );
                """;
            await command.ExecuteNonQueryAsync(_cancellationToken);
        }

        var legacyDatabase = new SqliteDatabase(legacyDirectory);
        await legacyDatabase.InitializeAsync(_cancellationToken);
        await using var migratedConnection = new SqliteConnection($"Data Source={legacyPath}");
        await migratedConnection.OpenAsync(_cancellationToken);
        await using var columns = migratedConnection.CreateCommand();
        columns.CommandText = "PRAGMA table_info(conversations);";
        await using var reader = await columns.ExecuteReaderAsync(_cancellationToken);
        var hasPinnedColumn = false;
        while (await reader.ReadAsync(_cancellationToken))
        {
            hasPinnedColumn |= reader.GetString(1) == "is_pinned";
        }
        Assert.IsTrue(hasPinnedColumn);
    }

    private async Task<string> SaveTurnAsync(string? ownerId, string modelId, string content) =>
        await _database.SaveTurnAsync(
            null, ownerId, modelId, $"{content} title", true,
            $"{content} question", ["atomic.txt"], 4,
            $"{content} answer", 3, _cancellationToken);

    private sealed class InMemoryDownloadHandler(IReadOnlyDictionary<string, byte[]> files)
        : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.RequestUri is null || !files.TryGetValue(request.RequestUri.ToString(), out var content))
            {
                throw new HttpRequestException($"Unexpected test download URL: {request.RequestUri}");
            }
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(content)
            });
        }
    }

    private sealed class TestInferenceEngineFactory : IInferenceEngineFactory
    {
        public TestInferenceEngine Engine { get; } = new();

        public IInferenceEngine Create(AssistantConfig config) => Engine;
    }

    private sealed class TestInferenceEngine : IInferenceEngine
    {
        public string EngineId => "test";
        public long HistoryLength { get; private set; }
        public string LastPrefill { get; private set; } = string.Empty;
        public bool ReturnInterruptedGeneration { get; set; }

        public int CountTokens(string text) => text.Length;

        public void ResetState()
        {
            HistoryLength = 0;
            LastPrefill = string.Empty;
        }

        public void Prefill(string text)
        {
            LastPrefill += text;
            HistoryLength += CountTokens(text);
        }

        public string FormatSystemPrompt(string systemPrompt) => $"<system>{systemPrompt}</system>";

        public string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns) =>
            string.Join("", turns.Select(turn => $"<user>{turn.User}</user><assistant>{turn.Assistant}</assistant>"));

        public string FormatUserPrompt(string question, bool thinkMode) =>
            $"<user>{question}</user>{(thinkMode ? "<think>" : "<think></think>")}";

        public (string Text, GenerationStats Stats) Generate(
            string prompt,
            GenerationConfig config,
            Action<string>? onToken = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            onToken?.Invoke("test answer");
            if (ReturnInterruptedGeneration)
            {
                return ("test answer", new GenerationStats(2, 1, 2, 0.5, false, false, true));
            }
            cancellationToken.ThrowIfCancellationRequested();
            return ("test answer", new GenerationStats(2, 1, 2, 0.5, false, false));
        }

        public void Dispose()
        {
        }
    }
}
