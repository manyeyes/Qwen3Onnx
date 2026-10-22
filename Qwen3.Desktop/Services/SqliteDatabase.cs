using System.Security.Cryptography;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Data.Sqlite;
using Qwen3.Desktop.Models;

namespace Qwen3.Desktop.Services;

public sealed class SqliteDatabase
{
    private static readonly HashSet<string> InsightStopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "the", "and", "for", "with", "that", "this", "from", "have", "what", "when", "where",
        "how", "why", "which", "would", "could", "should", "about", "your", "you", "are",
        "是", "的", "了", "我", "你", "他", "她", "它", "这", "那", "什么", "怎么", "如何",
        "可以", "请问", "一个", "一下", "以及", "进行", "帮助", "帮我", "是否"
    };

    private static readonly (string Name, string[] Terms)[] InsightDomains =
    [
        ("软件与技术", ["代码", "编程", "软件", "程序", "python", "csharp", "java", "api", "database", "模型", "算法", "技术"]),
        ("科学与学习", ["数学", "物理", "化学", "生物", "科学", "学习", "研究", "历史", "教育", "知识"]),
        ("写作与创作", ["写作", "文章", "故事", "小说", "创作", "文案", "翻译", "诗歌", "设计", "创意"]),
        ("工作与生活", ["工作", "生活", "计划", "健康", "旅行", "沟通", "建议", "管理", "职业"])
    ];

    private static readonly (string Title, string Name, string Description, string Url)[] BuiltInModels =
    [
        ("Qwen3 0.6B INT8", "Qwen3-0.6B-int8-onnx", "Qwen3 0.6B ONNX model, INT8 quantized.",
            "https://modelscope.cn/models/manyeyes/Qwen3-0.6B-int8-onnx"),
        ("Qwen3 0.6B W4A32", "Qwen3-0.6B-w4a32-onnx", "Qwen3 0.6B ONNX model, W4A32 quantized.",
            "https://modelscope.cn/models/manyeyes/Qwen3-0.6B-w4a32-onnx"),
        ("Qwen3 1.7B INT8", "Qwen3-1.7B-int8-onnx", "Qwen3 1.7B ONNX model, INT8 quantized.",
            "https://modelscope.cn/models/manyeyes/Qwen3-1.7B-int8-onnx"),
        ("Qwen3 1.7B W4A32", "Qwen3-1.7B-w4a32-onnx", "Qwen3 1.7B ONNX model, W4A32 quantized.",
            "https://modelscope.cn/models/manyeyes/Qwen3-1.7B-w4a32-onnx")
    ];

    private readonly string _connectionString;
    private readonly string _modelsDirectory;

    public string DatabasePath { get; }
    public string ModelsDirectory => _modelsDirectory;

    public SqliteDatabase(string applicationDataDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(applicationDataDirectory);
        Directory.CreateDirectory(applicationDataDirectory);
        DatabasePath = Path.Combine(applicationDataDirectory, "qwen3-desktop.sqlite3");
        _modelsDirectory = Path.Combine(applicationDataDirectory, "models");
        Directory.CreateDirectory(_modelsDirectory);
        _connectionString = new SqliteConnectionStringBuilder
        {
            DataSource = DatabasePath,
            Mode = SqliteOpenMode.ReadWriteCreate,
            Cache = SqliteCacheMode.Shared,
            Pooling = true
        }.ToString();
    }

    public async Task<BootstrapResult> InitializeAsync(CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using (var command = connection.CreateCommand())
        {
            command.CommandText = """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS app_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
                    is_locked INTEGER NOT NULL DEFAULT 0,
                    created_utc TEXT NOT NULL,
                    nickname TEXT NOT NULL DEFAULT '',
                    avatar_url TEXT NOT NULL DEFAULT '',
                    birthday TEXT NOT NULL DEFAULT '',
                    blog_url TEXT NOT NULL DEFAULT '',
                    bio TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS models (
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
                    direct_downloads_json TEXT NOT NULL DEFAULT '[]',
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    owner_user_id TEXT NULL REFERENCES users(id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL REFERENCES models(id),
                    created_utc TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_conversations_owner_updated
                    ON conversations(owner_user_id, updated_utc DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    attachments_json TEXT NOT NULL DEFAULT '[]',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    was_interrupted INTEGER NOT NULL DEFAULT 0,
                    created_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_messages_conversation_created
                    ON messages(conversation_id, created_utc);
                CREATE TABLE IF NOT EXISTS usage_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NULL REFERENCES users(id) ON DELETE SET NULL,
                    conversation_id TEXT NULL REFERENCES conversations(id) ON DELETE SET NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_usage_user ON usage_events(user_id);
                INSERT OR IGNORE INTO app_metadata(key, value) VALUES ('schema_version', '1');
                """;
            await command.ExecuteNonQueryAsync(cancellationToken);
        }

        await EnsureModelDownloadColumnAsync(connection, cancellationToken);
        await EnsureUserProfileColumnsAsync(connection, cancellationToken);
        await EnsureInterruptedMessageColumnAsync(connection, cancellationToken);
        await using (var versionCommand = connection.CreateCommand())
        {
            versionCommand.CommandText =
                "UPDATE app_metadata SET value = '4' WHERE key = 'schema_version';";
            await versionCommand.ExecuteNonQueryAsync(cancellationToken);
        }
        await SeedModelsAsync(connection, cancellationToken);
        var adminCount = await ScalarLongAsync(
            connection, "SELECT COUNT(*) FROM users WHERE role = 'admin';", cancellationToken);
        if (adminCount > 0)
        {
            return new BootstrapResult(null);
        }

        var password = CreateTemporaryPassword();
        var (hash, salt) = HashPassword(password);
        await using (var command = connection.CreateCommand())
        {
            command.CommandText = """
                INSERT INTO users(id, username, password_hash, password_salt, role, is_locked, created_utc)
                VALUES ($id, 'admin', $hash, $salt, 'admin', 0, $created);
                """;
            command.Parameters.AddWithValue("$id", Guid.NewGuid().ToString("N"));
            command.Parameters.AddWithValue("$hash", hash);
            command.Parameters.AddWithValue("$salt", salt);
            command.Parameters.AddWithValue("$created", DateTimeOffset.UtcNow.ToString("O"));
            await command.ExecuteNonQueryAsync(cancellationToken);
        }

        return new BootstrapResult(password);
    }

    public async Task<IReadOnlyList<ModelDefinition>> GetModelsAsync(CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT id, title, english_name, description, repository_url, local_path, is_installed,
                   embedding_file_name, decoder_file_name, required_files_json, direct_downloads_json
            FROM models ORDER BY title COLLATE NOCASE;
            """;
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var results = new List<ModelDefinition>();
        while (await reader.ReadAsync(cancellationToken))
        {
            results.Add(new ModelDefinition(
                reader.GetString(0), reader.GetString(1), reader.GetString(2), reader.GetString(3),
                reader.GetString(4), reader.GetString(5), reader.GetInt64(6) != 0,
                reader.GetString(7), reader.GetString(8),
                System.Text.Json.JsonSerializer.Deserialize<string[]>(reader.GetString(9)) ?? [],
                System.Text.Json.JsonSerializer.Deserialize<ModelFileLink[]>(reader.GetString(10)) ?? []));
        }
        return results;
    }

    public async Task RegisterModelAsync(
        string title, string englishName, string description, string repositoryUrl,
        IReadOnlyList<ModelFileLink> httpFileLinks,
        CancellationToken cancellationToken = default)
    {
        ValidateModelFields(title, englishName, repositoryUrl, httpFileLinks);
        var safeName = SanitizeModelDirectoryName(englishName);
        var localPath = Path.Combine(_modelsDirectory, safeName);
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO models(id, title, english_name, description, repository_url, local_path,
                is_installed, embedding_file_name, decoder_file_name, required_files_json,
                direct_downloads_json, created_utc)
            VALUES($id, $title, $name, $description, $url, $path, 0,
                'embedding.int8.onnx', 'decoder.int8.onnx', '[]', $files, $created);
            """;
        command.Parameters.AddWithValue("$id", Guid.NewGuid().ToString("N"));
        command.Parameters.AddWithValue("$title", title.Trim());
        command.Parameters.AddWithValue("$name", safeName);
        command.Parameters.AddWithValue("$description", description.Trim());
        command.Parameters.AddWithValue("$url", repositoryUrl.Trim());
        command.Parameters.AddWithValue("$files", System.Text.Json.JsonSerializer.Serialize(httpFileLinks));
        command.Parameters.AddWithValue("$path", localPath);
        command.Parameters.AddWithValue("$created", DateTimeOffset.UtcNow.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task UpdateModelAsync(
        string modelId,
        string title,
        string englishName,
        string description,
        string repositoryUrl,
        IReadOnlyList<ModelFileLink> httpFileLinks,
        CancellationToken cancellationToken = default)
    {
        ValidateModelFields(title, englishName, repositoryUrl, httpFileLinks);
        var safeName = SanitizeModelDirectoryName(englishName);
        await using var connection = await OpenAsync(cancellationToken);
        await using var transaction = await connection.BeginTransactionAsync(cancellationToken);
        await using var currentCommand = connection.CreateCommand();
        currentCommand.Transaction = (SqliteTransaction)transaction;
        currentCommand.CommandText = """
            SELECT english_name, local_path, is_installed,
                EXISTS(SELECT 1 FROM conversations WHERE model_id = models.id)
            FROM models WHERE id = $id;
            """;
        currentCommand.Parameters.AddWithValue("$id", modelId);
        await using var reader = await currentCommand.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken))
        {
            throw new InvalidOperationException("Model catalog entry no longer exists.");
        }
        var oldName = reader.GetString(0);
        var oldPath = reader.GetString(1);
        var isInstalled = reader.GetInt64(2) != 0;
        var hasConversations = reader.GetInt64(3) != 0;
        await reader.DisposeAsync();
        if (!string.Equals(oldName, safeName, StringComparison.OrdinalIgnoreCase) &&
            (isInstalled || hasConversations || Directory.Exists(oldPath)))
        {
            throw new InvalidOperationException(
                "The English name cannot change after the model is installed or referenced by a conversation.");
        }

        await using var update = connection.CreateCommand();
        update.Transaction = (SqliteTransaction)transaction;
        update.CommandText = """
            UPDATE models SET title = $title, english_name = $name, description = $description,
                repository_url = $url, direct_downloads_json = $files,
                local_path = $path WHERE id = $id;
            """;
        update.Parameters.AddWithValue("$title", title.Trim());
        update.Parameters.AddWithValue("$name", safeName);
        update.Parameters.AddWithValue("$description", description.Trim());
        update.Parameters.AddWithValue("$url", repositoryUrl.Trim());
        update.Parameters.AddWithValue("$files", System.Text.Json.JsonSerializer.Serialize(httpFileLinks));
        update.Parameters.AddWithValue("$path", Path.Combine(_modelsDirectory, safeName));
        update.Parameters.AddWithValue("$id", modelId);
        if (await update.ExecuteNonQueryAsync(cancellationToken) != 1)
        {
            throw new InvalidOperationException("Model catalog entry no longer exists.");
        }
        await transaction.CommitAsync(cancellationToken);
    }

    public async Task SetModelInstalledAsync(
        string modelId, string embeddingFileName, string decoderFileName,
        IReadOnlyList<string> requiredFiles, bool installed,
        CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            UPDATE models SET is_installed = $installed, embedding_file_name = $embedding,
                decoder_file_name = $decoder, required_files_json = $files WHERE id = $id;
            """;
        command.Parameters.AddWithValue("$installed", installed ? 1 : 0);
        command.Parameters.AddWithValue("$embedding", embeddingFileName);
        command.Parameters.AddWithValue("$decoder", decoderFileName);
        command.Parameters.AddWithValue("$files", System.Text.Json.JsonSerializer.Serialize(requiredFiles));
        command.Parameters.AddWithValue("$id", modelId);
        if (await command.ExecuteNonQueryAsync(cancellationToken) != 1)
        {
            throw new InvalidOperationException("Model catalog entry no longer exists.");
        }
    }

    public async Task DeleteModelAsync(string modelId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = "DELETE FROM models WHERE id = $id;";
        command.Parameters.AddWithValue("$id", modelId);
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task<bool> HasConversationsForModelAsync(
        string modelId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT EXISTS(SELECT 1 FROM conversations WHERE model_id = $id);";
        command.Parameters.AddWithValue("$id", modelId);
        return Convert.ToInt64(await command.ExecuteScalarAsync(cancellationToken)) != 0;
    }

    public async Task<string> SaveTurnAsync(
        string? conversationId,
        string? ownerUserId,
        string modelId,
        string title,
        bool updateTitle,
        string userContent,
        IReadOnlyList<string> attachments,
        int promptTokens,
        string assistantContent,
        int completionTokens,
        CancellationToken cancellationToken = default,
        bool wasInterrupted = false)
    {
        var id = conversationId ?? Guid.NewGuid().ToString("N");
        var now = DateTimeOffset.UtcNow.ToString("O");
        await using var connection = await OpenAsync(cancellationToken);
        await using var transaction = await connection.BeginTransactionAsync(cancellationToken);

        if (conversationId is null)
        {
            await using var create = connection.CreateCommand();
            create.Transaction = (SqliteTransaction)transaction;
            create.CommandText = """
                INSERT INTO conversations(id, title, owner_user_id, model_id, created_utc, updated_utc)
                VALUES($id, $title, $owner, $model, $created, $updated);
                """;
            create.Parameters.AddWithValue("$id", id);
            create.Parameters.AddWithValue("$title", title);
            create.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
            create.Parameters.AddWithValue("$model", modelId);
            create.Parameters.AddWithValue("$created", now);
            create.Parameters.AddWithValue("$updated", now);
            await create.ExecuteNonQueryAsync(cancellationToken);
        }
        else
        {
            await using var validate = connection.CreateCommand();
            validate.Transaction = (SqliteTransaction)transaction;
            validate.CommandText = """
                SELECT EXISTS(
                    SELECT 1 FROM conversations
                    WHERE id = $id AND model_id = $model AND
                        (($owner IS NULL AND owner_user_id IS NULL) OR owner_user_id = $owner)
                );
                """;
            validate.Parameters.AddWithValue("$id", id);
            validate.Parameters.AddWithValue("$model", modelId);
            validate.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
            if (Convert.ToInt64(await validate.ExecuteScalarAsync(cancellationToken)) == 0)
            {
                throw new UnauthorizedAccessException("Conversation does not exist or is not accessible to this user.");
            }
        }

        await using (var message = connection.CreateCommand())
        {
            message.Transaction = (SqliteTransaction)transaction;
            message.CommandText = """
                INSERT INTO messages(id, conversation_id, role, content, attachments_json, token_count,
                    was_interrupted, created_utc)
                VALUES($id, $conversation, $role, $content, $attachments, $tokens, $interrupted, $created);
                """;
            message.Parameters.Add("$id", SqliteType.Text);
            message.Parameters.AddWithValue("$conversation", id);
            message.Parameters.Add("$role", SqliteType.Text);
            message.Parameters.Add("$content", SqliteType.Text);
            message.Parameters.Add("$attachments", SqliteType.Text);
            message.Parameters.Add("$tokens", SqliteType.Integer);
            message.Parameters.Add("$interrupted", SqliteType.Integer);
            message.Parameters.AddWithValue("$created", now);

            message.Parameters["$id"].Value = Guid.NewGuid().ToString("N");
            message.Parameters["$role"].Value = "user";
            message.Parameters["$content"].Value = userContent;
            message.Parameters["$attachments"].Value = System.Text.Json.JsonSerializer.Serialize(attachments);
            message.Parameters["$tokens"].Value = Math.Max(0, promptTokens);
            message.Parameters["$interrupted"].Value = 0;
            await message.ExecuteNonQueryAsync(cancellationToken);

            message.Parameters["$id"].Value = Guid.NewGuid().ToString("N");
            message.Parameters["$role"].Value = "assistant";
            message.Parameters["$content"].Value = assistantContent;
            message.Parameters["$attachments"].Value = "[]";
            message.Parameters["$tokens"].Value = Math.Max(0, completionTokens);
            message.Parameters["$interrupted"].Value = wasInterrupted ? 1 : 0;
            await message.ExecuteNonQueryAsync(cancellationToken);
        }

        await using (var update = connection.CreateCommand())
        {
            update.Transaction = (SqliteTransaction)transaction;
            update.CommandText = """
                UPDATE conversations SET title = CASE WHEN $set_title = 1 THEN $title ELSE title END,
                    updated_utc = $updated WHERE id = $id;
                """;
            update.Parameters.AddWithValue("$updated", now);
            update.Parameters.AddWithValue("$id", id);
            update.Parameters.AddWithValue("$title", title);
            update.Parameters.AddWithValue("$set_title", updateTitle ? 1 : 0);
            await update.ExecuteNonQueryAsync(cancellationToken);
        }

        await using (var usage = connection.CreateCommand())
        {
            usage.Transaction = (SqliteTransaction)transaction;
            usage.CommandText = """
                INSERT INTO usage_events(id, user_id, conversation_id, prompt_tokens, completion_tokens, created_utc)
                VALUES($id, $user, $conversation, $prompt, $completion, $created);
                """;
            usage.Parameters.AddWithValue("$id", Guid.NewGuid().ToString("N"));
            usage.Parameters.AddWithValue("$user", (object?)ownerUserId ?? DBNull.Value);
            usage.Parameters.AddWithValue("$conversation", id);
            usage.Parameters.AddWithValue("$prompt", Math.Max(0, promptTokens));
            usage.Parameters.AddWithValue("$completion", Math.Max(0, completionTokens));
            usage.Parameters.AddWithValue("$created", now);
            await usage.ExecuteNonQueryAsync(cancellationToken);
        }

        await transaction.CommitAsync(cancellationToken);
        return id;
    }

    public async Task ReplaceAssistantReplyAsync(
        string conversationId,
        string? ownerUserId,
        string assistantMessageId,
        string assistantContent,
        int promptTokens,
        int completionTokens,
        CancellationToken cancellationToken = default,
        bool wasInterrupted = false)
    {
        var now = DateTimeOffset.UtcNow.ToString("O");
        await using var connection = await OpenAsync(cancellationToken);
        await using var transaction = await connection.BeginTransactionAsync(cancellationToken);

        await using (var update = connection.CreateCommand())
        {
            update.Transaction = (SqliteTransaction)transaction;
            update.CommandText = """
                UPDATE messages SET content = $content, token_count = $tokens,
                    was_interrupted = $interrupted, created_utc = $updated
                WHERE id = $message AND conversation_id = $conversation AND role = 'assistant'
                    AND rowid = (
                        SELECT MAX(rowid) FROM messages
                        WHERE conversation_id = $conversation
                    )
                    AND EXISTS (
                        SELECT 1 FROM conversations
                        WHERE id = $conversation AND
                            (($owner IS NULL AND owner_user_id IS NULL) OR owner_user_id = $owner)
                    );
                """;
            update.Parameters.AddWithValue("$content", assistantContent);
            update.Parameters.AddWithValue("$tokens", Math.Max(0, completionTokens));
            update.Parameters.AddWithValue("$interrupted", wasInterrupted ? 1 : 0);
            update.Parameters.AddWithValue("$updated", now);
            update.Parameters.AddWithValue("$message", assistantMessageId);
            update.Parameters.AddWithValue("$conversation", conversationId);
            update.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
            if (await update.ExecuteNonQueryAsync(cancellationToken) != 1)
            {
                throw new InvalidOperationException(
                    "The latest assistant reply changed before regeneration could be saved.");
            }
        }

        await using (var updateConversation = connection.CreateCommand())
        {
            updateConversation.Transaction = (SqliteTransaction)transaction;
            updateConversation.CommandText = """
                UPDATE conversations SET updated_utc = $updated WHERE id = $conversation;
                """;
            updateConversation.Parameters.AddWithValue("$updated", now);
            updateConversation.Parameters.AddWithValue("$conversation", conversationId);
            await updateConversation.ExecuteNonQueryAsync(cancellationToken);
        }

        await using (var usage = connection.CreateCommand())
        {
            usage.Transaction = (SqliteTransaction)transaction;
            usage.CommandText = """
                INSERT INTO usage_events(id, user_id, conversation_id, prompt_tokens, completion_tokens, created_utc)
                VALUES($id, $user, $conversation, $prompt, $completion, $created);
                """;
            usage.Parameters.AddWithValue("$id", Guid.NewGuid().ToString("N"));
            usage.Parameters.AddWithValue("$user", (object?)ownerUserId ?? DBNull.Value);
            usage.Parameters.AddWithValue("$conversation", conversationId);
            usage.Parameters.AddWithValue("$prompt", Math.Max(0, promptTokens));
            usage.Parameters.AddWithValue("$completion", Math.Max(0, completionTokens));
            usage.Parameters.AddWithValue("$created", now);
            await usage.ExecuteNonQueryAsync(cancellationToken);
        }

        await transaction.CommitAsync(cancellationToken);
    }

    public async Task<ConversationSummary?> GetConversationAsync(
        string id, string? ownerUserId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT id, title, owner_user_id, model_id, created_utc, updated_utc
            FROM conversations WHERE id = $id AND
                (($owner IS NULL AND owner_user_id IS NULL) OR owner_user_id = $owner);
            """;
        command.Parameters.AddWithValue("$id", id);
        command.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        return await reader.ReadAsync(cancellationToken) ? ReadConversation(reader) : null;
    }

    public async Task<IReadOnlyList<ConversationSummary>> GetConversationsAsync(
        string? ownerUserId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT id, title, owner_user_id, model_id, created_utc, updated_utc
            FROM conversations
            WHERE ($owner IS NULL AND owner_user_id IS NULL) OR owner_user_id = $owner
            ORDER BY updated_utc DESC;
            """;
        command.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var results = new List<ConversationSummary>();
        while (await reader.ReadAsync(cancellationToken))
        {
            results.Add(ReadConversation(reader));
        }
        return results;
    }

    public async Task<IReadOnlyList<StoredMessage>> GetMessagesAsync(
        string conversationId, string? ownerUserId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT m.id, m.conversation_id, m.role, m.content, m.attachments_json, m.token_count,
                m.created_utc, m.was_interrupted
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE c.id = $conversation AND
                (($owner IS NULL AND c.owner_user_id IS NULL) OR c.owner_user_id = $owner)
            ORDER BY m.created_utc, m.rowid;
            """;
        command.Parameters.AddWithValue("$conversation", conversationId);
        command.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var results = new List<StoredMessage>();
        while (await reader.ReadAsync(cancellationToken))
        {
            results.Add(new StoredMessage(
                reader.GetString(0), reader.GetString(1), reader.GetString(2), reader.GetString(3),
                System.Text.Json.JsonSerializer.Deserialize<string[]>(reader.GetString(4)) ?? [],
                reader.GetInt32(5), DateTimeOffset.Parse(reader.GetString(6)), reader.GetInt64(7) != 0));
        }
        return results;
    }

    public async Task<UserAccount?> AuthenticateAsync(
        string username, string password, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT u.id, u.username, u.password_hash, u.password_salt, u.role, u.is_locked, u.created_utc,
                COALESCE((SELECT SUM(prompt_tokens + completion_tokens) FROM usage_events WHERE user_id = u.id), 0)
            FROM users u WHERE u.username = $username COLLATE NOCASE;
            """;
        command.Parameters.AddWithValue("$username", username.Trim());
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken) || reader.GetInt64(5) != 0)
        {
            return null;
        }
        var (candidateHash, _) = HashPassword(password, Convert.FromBase64String(reader.GetString(3)));
        var storedHash = Convert.FromBase64String(reader.GetString(2));
        if (!CryptographicOperations.FixedTimeEquals(Convert.FromBase64String(candidateHash), storedHash))
        {
            return null;
        }
        return new UserAccount(reader.GetString(0), reader.GetString(1), reader.GetString(4),
            false, DateTimeOffset.Parse(reader.GetString(6)), reader.GetInt64(7));
    }

    public async Task<IReadOnlyList<UserAccount>> GetUsersAsync(CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT u.id, u.username, u.role, u.is_locked, u.created_utc,
                COALESCE(SUM(x.prompt_tokens + x.completion_tokens), 0)
            FROM users u LEFT JOIN usage_events x ON x.user_id = u.id
            GROUP BY u.id ORDER BY u.created_utc;
            """;
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var users = new List<UserAccount>();
        while (await reader.ReadAsync(cancellationToken))
        {
            users.Add(new UserAccount(reader.GetString(0), reader.GetString(1), reader.GetString(2),
                reader.GetInt64(3) != 0, DateTimeOffset.Parse(reader.GetString(4)), reader.GetInt64(5)));
        }
        return users;
    }

    public async Task<string> CreateUserAsync(
        string username, string password, CancellationToken cancellationToken = default)
    {
        ValidateCredentials(username, password);
        var (hash, salt) = HashPassword(password);
        var id = Guid.NewGuid().ToString("N");
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            INSERT INTO users(id, username, password_hash, password_salt, role, created_utc)
            VALUES($id, $username, $hash, $salt, 'user', $created);
            """;
        command.Parameters.AddWithValue("$id", id);
        command.Parameters.AddWithValue("$username", username.Trim());
        command.Parameters.AddWithValue("$hash", hash);
        command.Parameters.AddWithValue("$salt", salt);
        command.Parameters.AddWithValue("$created", DateTimeOffset.UtcNow.ToString("O"));
        await command.ExecuteNonQueryAsync(cancellationToken);
        return id;
    }

    public async Task<UserProfile> GetUserProfileAsync(
        string userId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT u.id, u.username, u.nickname, u.avatar_url, u.birthday, u.blog_url, u.bio,
                u.role, u.created_utc,
                COALESCE((SELECT SUM(prompt_tokens + completion_tokens)
                    FROM usage_events WHERE user_id = u.id), 0)
            FROM users u WHERE u.id = $id;
            """;
        command.Parameters.AddWithValue("$id", userId);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken))
        {
            throw new UnauthorizedAccessException("The current account no longer exists.");
        }
        return new UserProfile(
            reader.GetString(0), reader.GetString(1), reader.GetString(2), reader.GetString(3),
            reader.GetString(4), reader.GetString(5), reader.GetString(6), reader.GetString(7),
            DateTimeOffset.Parse(reader.GetString(8)), reader.GetInt64(9));
    }

    public async Task UpdateUserProfileAsync(
        string userId,
        string nickname,
        string avatarUrl,
        string birthday,
        string blogUrl,
        string bio,
        CancellationToken cancellationToken = default)
    {
        nickname = nickname.Trim();
        avatarUrl = avatarUrl.Trim();
        birthday = birthday.Trim();
        blogUrl = blogUrl.Trim();
        bio = bio.Trim();
        if (nickname.Length is < 1 or > 60)
        {
            throw new ArgumentException("昵称长度需为 1-60 个字符。", nameof(nickname));
        }
        if (avatarUrl.Length > 2048 || (!string.IsNullOrEmpty(avatarUrl) && !IsWebUrl(avatarUrl)))
        {
            throw new ArgumentException("头像链接必须是有效的 HTTP/HTTPS 地址，且不超过 2048 个字符。", nameof(avatarUrl));
        }
        if (!string.IsNullOrEmpty(birthday) &&
            !DateOnly.TryParseExact(birthday, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out _))
        {
            throw new ArgumentException("生日请使用 yyyy-MM-dd 格式。", nameof(birthday));
        }
        if (blogUrl.Length > 2048 || (!string.IsNullOrEmpty(blogUrl) && !IsWebUrl(blogUrl)))
        {
            throw new ArgumentException("博客链接必须是有效的 HTTP/HTTPS 地址，且不超过 2048 个字符。", nameof(blogUrl));
        }
        if (bio.Length > 1000)
        {
            throw new ArgumentException("个人简介不能超过 1000 个字符。", nameof(bio));
        }

        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            UPDATE users SET nickname = $nickname, avatar_url = $avatar, birthday = $birthday,
                blog_url = $blog, bio = $bio WHERE id = $id;
            """;
        command.Parameters.AddWithValue("$nickname", nickname);
        command.Parameters.AddWithValue("$avatar", avatarUrl);
        command.Parameters.AddWithValue("$birthday", birthday);
        command.Parameters.AddWithValue("$blog", blogUrl);
        command.Parameters.AddWithValue("$bio", bio);
        command.Parameters.AddWithValue("$id", userId);
        if (await command.ExecuteNonQueryAsync(cancellationToken) != 1)
        {
            throw new UnauthorizedAccessException("The current account no longer exists.");
        }
    }

    public async Task ChangeOwnPasswordAsync(
        string userId,
        string currentPassword,
        string newPassword,
        CancellationToken cancellationToken = default)
    {
        ValidatePassword(newPassword);
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            SELECT password_hash, password_salt FROM users WHERE id = $id;
            """;
        command.Parameters.AddWithValue("$id", userId);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken))
        {
            throw new UnauthorizedAccessException("The current account no longer exists.");
        }
        var hash = reader.GetString(0);
        var salt = reader.GetString(1);
        await reader.DisposeAsync();
        if (!VerifyPassword(currentPassword, hash, salt))
        {
            throw new UnauthorizedAccessException("当前密码不正确。");
        }
        var (newHash, newSalt) = HashPassword(newPassword);
        await using var update = connection.CreateCommand();
        update.CommandText = "UPDATE users SET password_hash = $hash, password_salt = $salt WHERE id = $id;";
        update.Parameters.AddWithValue("$hash", newHash);
        update.Parameters.AddWithValue("$salt", newSalt);
        update.Parameters.AddWithValue("$id", userId);
        if (await update.ExecuteNonQueryAsync(cancellationToken) != 1)
        {
            throw new UnauthorizedAccessException("The current account no longer exists.");
        }
    }

    public async Task<UserInsights> GetOwnUserInsightsAsync(
        string userId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var summary = connection.CreateCommand();
        summary.CommandText = """
            SELECT
                (SELECT COUNT(*) FROM conversations WHERE owner_user_id = $id),
                (SELECT COUNT(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id
                    WHERE c.owner_user_id = $id),
                COALESCE((SELECT SUM(prompt_tokens + completion_tokens)
                    FROM usage_events WHERE user_id = $id), 0);
            """;
        summary.Parameters.AddWithValue("$id", userId);
        await using var reader = await summary.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken))
        {
            throw new InvalidOperationException("Could not load user profile statistics.");
        }
        var conversationCount = reader.GetInt32(0);
        var messageCount = reader.GetInt32(1);
        var tokensUsed = reader.GetInt64(2);
        await reader.DisposeAsync();

        await using var messagesCommand = connection.CreateCommand();
        messagesCommand.CommandText = """
            SELECT m.content FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE c.owner_user_id = $id AND m.role = 'user'
            ORDER BY m.created_utc DESC;
            """;
        messagesCommand.Parameters.AddWithValue("$id", userId);
        await using var messagesReader = await messagesCommand.ExecuteReaderAsync(cancellationToken);
        var text = new StringBuilder();
        while (await messagesReader.ReadAsync(cancellationToken))
        {
            text.AppendLine(messagesReader.GetString(0));
        }

        var terms = Regex.Matches(text.ToString(), @"[\p{IsCJKUnifiedIdeographs}]{2,}|[A-Za-z][A-Za-z0-9+#.-]{2,}")
            .Select(match => match.Value.Trim('.', '-', '+', '#'))
            .Where(term => term.Length > 1 && !InsightStopWords.Contains(term))
            .GroupBy(term => term, StringComparer.OrdinalIgnoreCase)
            .Select(group => (Term: group.Key, Count: group.Count()))
            .OrderByDescending(item => item.Count)
            .ThenBy(item => item.Term, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var domains = InsightDomains.Select(domain =>
        {
            var count = terms.Where(term => domain.Terms.Contains(
                    term.Term, StringComparer.OrdinalIgnoreCase))
                .Sum(term => term.Count);
            return (domain.Name, Count: count);
        }).Where(domain => domain.Count > 0)
            .OrderByDescending(domain => domain.Count)
            .ToArray();
        var domainTotal = Math.Max(1, domains.Sum(domain => domain.Count));

        return new UserInsights(
            conversationCount,
            messageCount,
            tokensUsed,
            terms.Take(12).Select(term => term.Term).ToArray(),
            domains.Select(domain => new UserDomainInsight(
                domain.Name, domain.Count, domain.Count * 100.0 / domainTotal)).ToArray());
    }

    public async Task DeleteOwnAccountAsync(
        string userId, string password, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var transaction = await connection.BeginTransactionAsync(cancellationToken);
        await using var lookup = connection.CreateCommand();
        lookup.Transaction = (SqliteTransaction)transaction;
        lookup.CommandText = "SELECT password_hash, password_salt, role FROM users WHERE id = $id;";
        lookup.Parameters.AddWithValue("$id", userId);
        await using var reader = await lookup.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken))
        {
            throw new UnauthorizedAccessException("The current account no longer exists.");
        }
        var hash = reader.GetString(0);
        var salt = reader.GetString(1);
        var role = reader.GetString(2);
        await reader.DisposeAsync();
        if (string.Equals(role, "admin", StringComparison.Ordinal))
        {
            throw new InvalidOperationException("管理员账户不能自主注销，以免失去本地管理入口。");
        }
        if (!VerifyPassword(password, hash, salt))
        {
            throw new UnauthorizedAccessException("密码不正确，无法注销账户。");
        }
        await using var delete = connection.CreateCommand();
        delete.Transaction = (SqliteTransaction)transaction;
        delete.CommandText = "DELETE FROM users WHERE id = $id AND role = 'user';";
        delete.Parameters.AddWithValue("$id", userId);
        if (await delete.ExecuteNonQueryAsync(cancellationToken) != 1)
        {
            throw new InvalidOperationException("账户未能注销。");
        }
        await transaction.CommitAsync(cancellationToken);
    }

    public async Task SetUserLockedAsync(string userId, bool locked, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = "UPDATE users SET is_locked = $locked WHERE id = $id AND role <> 'admin';";
        command.Parameters.AddWithValue("$locked", locked ? 1 : 0);
        command.Parameters.AddWithValue("$id", userId);
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task DeleteConversationAsync(
        string conversationId, string? ownerUserId, CancellationToken cancellationToken = default)
    {
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            DELETE FROM conversations WHERE id = $id AND
                (($owner IS NULL AND owner_user_id IS NULL) OR owner_user_id = $owner);
            """;
        command.Parameters.AddWithValue("$id", conversationId);
        command.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task RenameConversationAsync(
        string conversationId,
        string? ownerUserId,
        string title,
        CancellationToken cancellationToken = default)
    {
        var normalizedTitle = title.Trim();
        if (normalizedTitle.Length is < 1 or > 120)
        {
            throw new ArgumentException("Conversation title must contain 1-120 characters.", nameof(title));
        }
        await using var connection = await OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            UPDATE conversations SET title = $title, updated_utc = $updated
            WHERE id = $id AND
                (($owner IS NULL AND owner_user_id IS NULL) OR owner_user_id = $owner);
            """;
        command.Parameters.AddWithValue("$title", normalizedTitle);
        command.Parameters.AddWithValue("$updated", DateTimeOffset.UtcNow.ToString("O"));
        command.Parameters.AddWithValue("$id", conversationId);
        command.Parameters.AddWithValue("$owner", (object?)ownerUserId ?? DBNull.Value);
        if (await command.ExecuteNonQueryAsync(cancellationToken) != 1)
        {
            throw new UnauthorizedAccessException("Conversation does not exist or is not accessible to this user.");
        }
    }

    public static (string Hash, string Salt) HashPassword(string password, byte[]? salt = null)
    {
        salt ??= RandomNumberGenerator.GetBytes(16);
        var hash = Rfc2898DeriveBytes.Pbkdf2(password, salt, 600_000, HashAlgorithmName.SHA256, 32);
        return (Convert.ToBase64String(hash), Convert.ToBase64String(salt));
    }

    public static string CreateTemporaryPassword()
    {
        var bytes = RandomNumberGenerator.GetBytes(24);
        return Convert.ToBase64String(bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_');
    }

    private async Task SeedModelsAsync(SqliteConnection connection, CancellationToken cancellationToken)
    {
        foreach (var model in BuiltInModels)
        {
            var directoryName = SanitizeModelDirectoryName(model.Name);
            var path = Path.Combine(_modelsDirectory, directoryName);
            var embedded = model.Name.Contains("w4a32", StringComparison.OrdinalIgnoreCase)
                ? "embedding.w4a32.onnx" : "embedding.int8.onnx";
            var decoder = model.Name.Contains("w4a32", StringComparison.OrdinalIgnoreCase)
                ? "decoder.w4a32.onnx" : "decoder.int8.onnx";
            await using var command = connection.CreateCommand();
            command.CommandText = """
                INSERT OR IGNORE INTO models(id, title, english_name, description, repository_url, local_path,
                    is_installed, embedding_file_name, decoder_file_name, required_files_json, created_utc)
                VALUES($id, $title, $name, $description, $url, $path, 0, $embedding, $decoder, '[]', $created);
                """;
            command.Parameters.AddWithValue("$id", directoryName.ToLowerInvariant());
            command.Parameters.AddWithValue("$title", model.Title);
            command.Parameters.AddWithValue("$name", directoryName);
            command.Parameters.AddWithValue("$description", model.Description);
            command.Parameters.AddWithValue("$url", model.Url);
            command.Parameters.AddWithValue("$path", path);
            command.Parameters.AddWithValue("$embedding", embedded);
            command.Parameters.AddWithValue("$decoder", decoder);
            command.Parameters.AddWithValue("$created", DateTimeOffset.UtcNow.ToString("O"));
            await command.ExecuteNonQueryAsync(cancellationToken);
        }
    }

    private static async Task EnsureModelDownloadColumnAsync(
        SqliteConnection connection, CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.CommandText = "PRAGMA table_info(models);";
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var hasDirectDownloads = false;
        while (await reader.ReadAsync(cancellationToken))
        {
            if (string.Equals(reader.GetString(1), "direct_downloads_json", StringComparison.OrdinalIgnoreCase))
            {
                hasDirectDownloads = true;
                break;
            }
        }
        await reader.DisposeAsync();
        if (!hasDirectDownloads)
        {
            await using var migration = connection.CreateCommand();
            migration.CommandText =
                "ALTER TABLE models ADD COLUMN direct_downloads_json TEXT NOT NULL DEFAULT '[]';";
            await migration.ExecuteNonQueryAsync(cancellationToken);
        }
    }

    private static async Task EnsureUserProfileColumnsAsync(
        SqliteConnection connection, CancellationToken cancellationToken)
    {
        var existing = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        await using (var command = connection.CreateCommand())
        {
            command.CommandText = "PRAGMA table_info(users);";
            await using var reader = await command.ExecuteReaderAsync(cancellationToken);
            while (await reader.ReadAsync(cancellationToken))
            {
                existing.Add(reader.GetString(1));
            }
        }

        var migrations = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["nickname"] = "ALTER TABLE users ADD COLUMN nickname TEXT NOT NULL DEFAULT '';",
            ["avatar_url"] = "ALTER TABLE users ADD COLUMN avatar_url TEXT NOT NULL DEFAULT '';",
            ["birthday"] = "ALTER TABLE users ADD COLUMN birthday TEXT NOT NULL DEFAULT '';",
            ["blog_url"] = "ALTER TABLE users ADD COLUMN blog_url TEXT NOT NULL DEFAULT '';",
            ["bio"] = "ALTER TABLE users ADD COLUMN bio TEXT NOT NULL DEFAULT '';"
        };
        foreach (var (column, sql) in migrations)
        {
            if (existing.Contains(column))
            {
                continue;
            }
            await using var migration = connection.CreateCommand();
            migration.CommandText = sql;
            await migration.ExecuteNonQueryAsync(cancellationToken);
        }
    }

    private static async Task EnsureInterruptedMessageColumnAsync(
        SqliteConnection connection, CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.CommandText = "PRAGMA table_info(messages);";
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var exists = false;
        while (await reader.ReadAsync(cancellationToken))
        {
            if (string.Equals(reader.GetString(1), "was_interrupted", StringComparison.OrdinalIgnoreCase))
            {
                exists = true;
                break;
            }
        }
        await reader.DisposeAsync();
        if (exists)
        {
            return;
        }

        await using var migration = connection.CreateCommand();
        migration.CommandText =
            "ALTER TABLE messages ADD COLUMN was_interrupted INTEGER NOT NULL DEFAULT 0;";
        await migration.ExecuteNonQueryAsync(cancellationToken);
    }

    private async Task<SqliteConnection> OpenAsync(CancellationToken cancellationToken)
    {
        var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(cancellationToken);
        await using var command = connection.CreateCommand();
        command.CommandText = "PRAGMA foreign_keys = ON;";
        await command.ExecuteNonQueryAsync(cancellationToken);
        return connection;
    }

    private static async Task<long> ScalarLongAsync(
        SqliteConnection connection, string sql, CancellationToken cancellationToken)
    {
        await using var command = connection.CreateCommand();
        command.CommandText = sql;
        return (long)(await command.ExecuteScalarAsync(cancellationToken) ?? 0L);
    }

    private static ConversationSummary ReadConversation(SqliteDataReader reader) =>
        new(reader.GetString(0), reader.GetString(1), reader.IsDBNull(2) ? null : reader.GetString(2),
            reader.GetString(3), DateTimeOffset.Parse(reader.GetString(4)), DateTimeOffset.Parse(reader.GetString(5)));

    private static string SanitizeModelDirectoryName(string value)
    {
        var name = new string(value.Trim().Select(character =>
            char.IsLetterOrDigit(character) || character is '-' or '_' or '.' ? character : '-').ToArray());
        if (string.IsNullOrWhiteSpace(name) || name is "." or "..")
        {
            throw new ArgumentException("Model English name must contain letters or digits.", nameof(value));
        }
        return name;
    }

    private static bool VerifyPassword(string password, string encodedHash, string encodedSalt)
    {
        var (candidateHash, _) = HashPassword(password, Convert.FromBase64String(encodedSalt));
        return CryptographicOperations.FixedTimeEquals(
            Convert.FromBase64String(candidateHash), Convert.FromBase64String(encodedHash));
    }

    private static bool IsWebUrl(string value) =>
        Uri.TryCreate(value, UriKind.Absolute, out var uri) &&
        uri.Scheme is "http" or "https" &&
        !string.IsNullOrWhiteSpace(uri.Host);

    private static void ValidateModelFields(
        string title,
        string englishName,
        string repositoryUrl,
        IReadOnlyList<ModelFileLink> httpFileLinks)
    {
        if (string.IsNullOrWhiteSpace(title) || title.Length > 100)
        {
            throw new ArgumentException("Model title is required and must be at most 100 characters.");
        }
        if (string.IsNullOrWhiteSpace(englishName) || englishName.Length > 100)
        {
            throw new ArgumentException("Model English name is required and must be at most 100 characters.");
        }
        if (!string.IsNullOrWhiteSpace(repositoryUrl) &&
            (!Uri.TryCreate(repositoryUrl, UriKind.Absolute, out var uri) ||
             uri.Scheme is not ("https" or "http")))
        {
            throw new ArgumentException("Git repository URL must be an HTTP or HTTPS URL.");
        }
        if (string.IsNullOrWhiteSpace(repositoryUrl) && httpFileLinks.Count == 0)
        {
            throw new ArgumentException("Provide a Git repository URL or direct HTTP file links.");
        }
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var file in httpFileLinks)
        {
            if (string.IsNullOrWhiteSpace(file.Path) || Path.IsPathRooted(file.Path) ||
                file.Path.Split('/', '\\').Any(part => part is "" or "." or "..") ||
                file.Path.Contains('\0') || file.Size < 0 ||
                !Uri.TryCreate(file.Url, UriKind.Absolute, out var fileUri) ||
                fileUri.Scheme is not ("https" or "http") ||
                !seenPaths.Add(file.Path.Replace('\\', '/')) ||
                (file.Sha256 is not null &&
                 (file.Sha256.Length != 64 || !file.Sha256.All(Uri.IsHexDigit))))
            {
                throw new ArgumentException($"Invalid direct download file entry: {file.Path}");
            }
        }
    }

    private static void ValidateCredentials(string username, string password)
    {
        if (string.IsNullOrWhiteSpace(username) || username.Length is < 3 or > 64 ||
            username.Any(character => !char.IsLetterOrDigit(character) && character is not ('_' or '-' or '.')))
        {
            throw new ArgumentException("Username must be 3-64 characters and contain only letters, digits, '.', '_' or '-'.");
        }
        ValidatePassword(password);
    }

    private static void ValidatePassword(string password)
    {
        if (string.IsNullOrWhiteSpace(password) || password.Length < 12)
        {
            throw new ArgumentException("Password must contain at least 12 characters.");
        }
    }
}
