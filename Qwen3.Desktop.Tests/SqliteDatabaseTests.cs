using Microsoft.Data.Sqlite;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;
using Qwen3.Desktop.ViewModels;
using Qwen3.Inference;
using System.Net;
using System.Security.Cryptography;

namespace Qwen3.Desktop.Tests;

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
            _cancellationToken);

        var updated = await _database.GetMessagesAsync(conversationId, userId, _cancellationToken);
        Assert.AreEqual(2, updated.Count);
        Assert.AreEqual(assistantMessage.Id, updated[1].Id);
        Assert.AreEqual("replacement answer", updated[1].Content);
        Assert.AreEqual(8, updated[1].TokenCount);
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
    public async Task RefreshingModelsDoesNotReplaceLoadedConversationWithNewChat()
    {
        using var services = new DesktopServices(_temporaryDirectory);
        await services.Database.InitializeAsync(_cancellationToken);
        var model = (await services.Database.GetModelsAsync(_cancellationToken))[0];
        Directory.CreateDirectory(model.LocalPath);
        await File.WriteAllTextAsync(Path.Combine(model.LocalPath, "config.json"), "{}", _cancellationToken);
        await File.WriteAllTextAsync(Path.Combine(model.LocalPath, "tokenizer.json"), "{}", _cancellationToken);
        await File.WriteAllBytesAsync(Path.Combine(model.LocalPath, model.EmbeddingFileName), [1]);
        await File.WriteAllBytesAsync(Path.Combine(model.LocalPath, model.DecoderFileName), [2]);

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

        Assert.AreEqual(conversationId, shell.Chat.ConversationId);
        Assert.AreEqual(2, shell.Chat.Messages.Count);
        Assert.IsFalse(shell.Chat.HasNoMessages);
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
        Assert.AreEqual(4, models.Count);
        Assert.IsTrue(models.All(model => model.HttpFileLinks.Count == 0));
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
            string prompt, GenerationConfig config, Action<string>? onToken = null)
        {
            onToken?.Invoke("test answer");
            return ("test answer", new GenerationStats(2, 1, 2, 0.5, false, false));
        }

        public void Dispose()
        {
        }
    }
}
