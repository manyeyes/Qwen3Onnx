using System.Text;
using System.Diagnostics;
using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;
using Qwen3.Inference;

namespace Qwen3.Desktop.ViewModels;

public partial class ChatPageViewModel : ObservableObject
{
    private const int MaximumAttachmentCount = 5;
    private const long MaximumAttachmentSize = 1024 * 1024;
    private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".xml", ".yaml", ".yml",
        ".log", ".ini", ".toml", ".cs", ".py", ".js", ".ts", ".html", ".css", ".sql"
    };

    private readonly DesktopServices _services;
    private readonly MainWindowViewModel _shell;
    private readonly ILogger<ChatPageViewModel> _logger;
    private readonly List<(string Name, string Content)> _attachments = [];
    private bool _syncing;
    private CancellationTokenSource? _generationCancellation;

    [ObservableProperty]
    private string inputText = string.Empty;

    [ObservableProperty]
    private ObservableCollection<MessageViewModel> messages = [];

    [ObservableProperty]
    private ObservableCollection<ModelDefinition> installedModels = [];

    [ObservableProperty]
    private ModelDefinition? selectedModel;

    [ObservableProperty]
    private ObservableCollection<string> attachmentNames = [];

    [ObservableProperty]
    private string attachmentHint = string.Empty;

    [ObservableProperty]
    private string? conversationId;

    [ObservableProperty]
    private string conversationTitle = "新对话";

    [ObservableProperty]
    private bool isThinking = true;

    [ObservableProperty]
    private bool isAdvancedOpen;

    [ObservableProperty]
    private bool isBusy;

    [ObservableProperty]
    private string statusText = "选择本地模型后开始对话。";

    [ObservableProperty]
    private int maxNewTokens = 256;

    [ObservableProperty]
    private double temperature = 0.6;

    [ObservableProperty]
    private double topP = 0.95;

    [ObservableProperty]
    private int topK = 20;

    [ObservableProperty]
    private double repetitionPenalty = 1.05;

    [ObservableProperty]
    private int thinkingBudget = 192;

    [ObservableProperty]
    private int maxHistoryTurns = 3;

    [ObservableProperty]
    private bool doSample = true;

    [ObservableProperty]
    private bool streamOutput = true;

    public Action? ConversationsChanged { get; set; }
    public bool CanSend => !IsBusy && (SelectedModel?.IsInstalled ?? false);
    public bool CanStop => IsBusy;
    public bool CanShowSend => !IsBusy;
    public bool CanChangeConversation => !IsBusy;
    public bool HasNoMessages => Messages.Count == 0;
    public bool HasAttachmentHint => !string.IsNullOrEmpty(AttachmentHint);
    public Func<string, Task>? CopyTextAsync { get; set; }

    partial void OnMessagesChanged(ObservableCollection<MessageViewModel> value) =>
        OnPropertyChanged(nameof(HasNoMessages));

    partial void OnAttachmentHintChanged(string value) =>
        OnPropertyChanged(nameof(HasAttachmentHint));

    public ChatPageViewModel(DesktopServices services, MainWindowViewModel shell)
    {
        _services = services;
        _shell = shell;
        _logger = services.LoggerFactory.CreateLogger<ChatPageViewModel>();
    }

    public async Task RefreshModelsAsync(CancellationToken cancellationToken = default)
    {
        var previousId = SelectedModel?.Id;
        var models = await _services.Database.GetModelsAsync(cancellationToken);
        _syncing = true;
        try
        {
            InstalledModels = new ObservableCollection<ModelDefinition>(
                models.Where(model => ModelDownloadService.IsInstalled(model))
                    .Select(model => model with { IsInstalled = true }));
            SelectedModel = InstalledModels.FirstOrDefault(model => model.Id == previousId)
                ?? InstalledModels.FirstOrDefault();
        }
        finally
        {
            _syncing = false;
        }
        OnPropertyChanged(nameof(CanSend));
        if (InstalledModels.Count == 0)
        {
            StatusText = "尚未安装模型。请打开“模型管理”下载并安装模型。";
        }
        else if (StatusText.StartsWith("尚未安装模型", StringComparison.Ordinal))
        {
            StatusText = "本地模型已就绪。";
        }
    }

    public async Task LoadConversationAsync(
        ConversationSummary conversation,
        string? ownerUserId,
        CancellationToken cancellationToken = default)
    {
        if (IsBusy || !string.Equals(ownerUserId, _shell.CurrentUser?.Id, StringComparison.Ordinal))
        {
            throw new UnauthorizedAccessException("The conversation can no longer be loaded for this identity.");
        }
        var accessible = await _services.Database.GetConversationAsync(
            conversation.Id, ownerUserId, cancellationToken);
        if (accessible is null)
        {
            throw new UnauthorizedAccessException("此对话不属于当前访问范围。");
        }

        var models = await _services.Database.GetModelsAsync(cancellationToken);
        var selectedModel = models.FirstOrDefault(model => model.Id == conversation.ModelId &&
            ModelDownloadService.IsInstalled(model));
        var storedMessages = await _services.Chat.GetMessagesAsync(
            conversation.Id, ownerUserId, cancellationToken);
        if (IsBusy || !string.Equals(ownerUserId, _shell.CurrentUser?.Id, StringComparison.Ordinal))
        {
            throw new UnauthorizedAccessException("The active identity changed while loading this conversation.");
        }

        _syncing = true;
        try
        {
            SelectedModel = selectedModel is null ? null : selectedModel with { IsInstalled = true };
            ConversationId = conversation.Id;
            ConversationTitle = conversation.Title;
            Messages = new ObservableCollection<MessageViewModel>(storedMessages.Select(message =>
                new MessageViewModel
                {
                    Role = message.Role,
                    Content = message.Role == "user" && message.Attachments.Count > 0
                        ? GetStoredUserDisplayContent(message.Content)
                        : message.Content,
                    PromptContent = message.Role == "user" ? message.Content : string.Empty,
                    AttachmentNames = new ObservableCollection<string>(message.Attachments),
                    WasInterrupted = message.WasInterrupted
                }));
            UpdateLatestAssistantAction();
            _attachments.Clear();
            AttachmentNames.Clear();
            AttachmentHint = string.Empty;
            StatusText = SelectedModel is null
                ? "此对话使用的模型尚未安装，请先在模型管理中下载。"
                : $"已加载对话：{conversation.Title}";
        }
        finally
        {
            _syncing = false;
        }
        OnPropertyChanged(nameof(CanSend));
    }

    public void StartNewConversation()
    {
        if (IsBusy)
        {
            return;
        }
        ConversationId = null;
        ConversationTitle = "新对话";
        Messages.Clear();
        OnPropertyChanged(nameof(HasNoMessages));
        UpdateLatestAssistantAction();
        _attachments.Clear();
        AttachmentNames.Clear();
        AttachmentHint = string.Empty;
        InputText = string.Empty;
        StatusText = "新对话。";
        _ = RefreshModelsAsync();
    }

    public async Task ChangeIdentityAsync(CancellationToken cancellationToken = default)
    {
        _syncing = true;
        try
        {
            ConversationId = null;
            ConversationTitle = "新对话";
            Messages.Clear();
            OnPropertyChanged(nameof(HasNoMessages));
            UpdateLatestAssistantAction();
            _attachments.Clear();
            AttachmentNames.Clear();
            AttachmentHint = string.Empty;
            InputText = string.Empty;
            await RefreshModelsAsync(cancellationToken);
            StatusText = _shell.CurrentUser is null
                ? "访客模式：此处创建的对话对本机访客公开。"
                : $"已登录为 {_shell.CurrentUser.Username}，新对话仅此账户可见。";
        }
        finally
        {
            _syncing = false;
        }
    }

    [RelayCommand]
    private void ToggleAdvanced() => IsAdvancedOpen = !IsAdvancedOpen;

    [RelayCommand]
    private void CloseAdvanced() => IsAdvancedOpen = false;

    [RelayCommand]
    private void RemoveAttachment(string name)
    {
        _attachments.RemoveAll(item => item.Name == name);
        AttachmentNames.Remove(name);
        AttachmentHint = string.Empty;
    }

    public async Task AddAttachmentsAsync(
        IEnumerable<string> paths, CancellationToken cancellationToken = default)
    {
        AttachmentHint = string.Empty;
        foreach (var path in paths)
        {
            if (_attachments.Count >= MaximumAttachmentCount)
            {
                AttachmentHint = $"每条消息最多添加 {MaximumAttachmentCount} 个文本附件。";
                break;
            }
            var extension = Path.GetExtension(path);
            if (!TextExtensions.Contains(extension))
            {
                AttachmentHint = $"不支持的文本附件格式：{extension}";
                continue;
            }
            var info = new FileInfo(path);
            if (info.Length > MaximumAttachmentSize)
            {
                AttachmentHint = $"附件超过 1 MiB 限制：{info.Name}";
                continue;
            }
            if (_attachments.Any(item => item.Name.Equals(info.Name, StringComparison.OrdinalIgnoreCase)))
            {
                continue;
            }
            var content = await File.ReadAllTextAsync(path, Encoding.UTF8, cancellationToken);
            if (content.Contains('\0'))
            {
                AttachmentHint = $"附件看起来不是文本文件：{info.Name}";
                continue;
            }
            _attachments.Add((info.Name, content));
            AttachmentNames.Add(info.Name);
        }
    }

    public async Task PickAttachmentsAsync(CancellationToken cancellationToken = default)
    {
        if (SelectTextFilesAsync is null)
        {
            AttachmentHint = "无法打开文件选择器。";
            return;
        }
        var paths = await SelectTextFilesAsync();
        await AddAttachmentsAsync(paths, cancellationToken);
    }

    public Func<Task<IReadOnlyList<string>>>? SelectTextFilesAsync { get; set; }

    [RelayCommand]
    private Task AddAttachmentsAsync() => PickAttachmentsAsync();

    [RelayCommand]
    private Task SendFromKeyboardAsync() => SendAsync(CancellationToken.None);

    [RelayCommand]
    private async Task SendAsync(CancellationToken cancellationToken)
    {
        if (IsBusy || SelectedModel is null || !SelectedModel.IsInstalled)
        {
            StatusText = "请先在模型管理中安装并选择一个模型。";
            return;
        }
        if (string.IsNullOrWhiteSpace(InputText) && _attachments.Count == 0)
        {
            StatusText = "请输入消息或添加文本附件。";
            return;
        }

        await SendMessageAsync(InputText.Trim(), _attachments.ToArray(), null, clearComposer: true,
            cancellationToken);
    }

    [RelayCommand]
    private async Task ResendMessageAsync(MessageViewModel? message, CancellationToken cancellationToken)
    {
        if (message is null || !message.IsUser || IsBusy)
        {
            return;
        }
        await SendMessageAsync(message.Content, [], message.PromptContent, clearComposer: false,
            cancellationToken, message.AttachmentNames);
    }

    [RelayCommand]
    private async Task RegenerateReplyAsync(MessageViewModel? message, CancellationToken cancellationToken)
    {
        if (message is null || !message.CanRegenerate || IsBusy ||
            ConversationId is null || SelectedModel is null)
        {
            return;
        }

        var oldContent = message.Content;
        using var generationCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _generationCancellation = generationCancellation;
        IsBusy = true;
        message.Content = string.Empty;
        message.IsPending = true;
        StatusText = "正在重新生成回复…";
        var elapsed = Stopwatch.StartNew();
        using var streamUpdater = new StreamingMessageUpdater(
            message, exception => _logger.LogError(exception, "Parsing streamed reply failed"));

        try
        {
            var result = await _services.Chat.RegenerateLastReplyAsync(
                ConversationId,
                _shell.CurrentUser?.Id,
                SelectedModel,
                BuildGenerationConfig(),
                streamUpdater.Append,
                generationCancellation.Token);
            elapsed.Stop();
            await streamUpdater.CompleteAsync(result.Assistant.Answer);
            message.WasInterrupted = result.Assistant.Generation.WasCancelled;
            StatusText = result.Assistant.Generation.WasCancelled
                ? "已停止生成，部分回复已保存。"
                : FormatGenerationStatus(result.Assistant, elapsed.Elapsed);
            UpdateLatestAssistantAction();
            ConversationsChanged?.Invoke();
        }
        catch (Exception exception)
        {
            if (exception is OperationCanceledException && generationCancellation.IsCancellationRequested)
            {
                StatusText = "已停止生成，原回复已保留。";
            }
            else
            {
                _logger.LogError(exception, "Desktop chat reply regeneration failed");
                StatusText = $"重新生成失败，原回复已保留：{exception.Message}";
            }
            streamUpdater.Stop();
            message.Content = oldContent;
            message.IsPending = false;
        }
        finally
        {
            if (ReferenceEquals(_generationCancellation, generationCancellation))
            {
                _generationCancellation = null;
            }
            IsBusy = false;
            OnPropertyChanged(nameof(CanSend));
        }
    }

    [RelayCommand]
    private async Task CopyMessageAsync(MessageViewModel? message)
    {
        if (message is null || CopyTextAsync is null)
        {
            StatusText = "当前窗口未提供剪贴板服务。";
            return;
        }
        try
        {
            await CopyTextAsync(message.MarkdownContent);
            StatusText = "消息内容已复制。";
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Copying chat message to clipboard failed");
            StatusText = $"复制失败：{exception.Message}";
        }
    }

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void StopGeneration() => _generationCancellation?.Cancel();

    private async Task SendMessageAsync(
        string text,
        IReadOnlyList<(string Name, string Content)> attachments,
        string? promptTextOverride,
        bool clearComposer,
        CancellationToken cancellationToken,
        IReadOnlyList<string>? attachmentNamesOverride = null)
    {
        if (IsBusy || SelectedModel is null || !SelectedModel.IsInstalled)
        {
            StatusText = "请先在模型管理中安装并选择一个模型。";
            return;
        }
        if (string.IsNullOrWhiteSpace(text) && attachments.Count == 0 &&
            string.IsNullOrWhiteSpace(promptTextOverride))
        {
            StatusText = "请输入消息或添加文本附件。";
            return;
        }

        var model = SelectedModel;
        var ownerId = _shell.CurrentUser?.Id;
        var attachmentNames = attachmentNamesOverride is null
            ? attachments.Select(item => item.Name).ToArray()
            : attachmentNamesOverride.ToArray();
        var config = BuildGenerationConfig();
        using var generationCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _generationCancellation = generationCancellation;
        if (clearComposer)
        {
            InputText = string.Empty;
            _attachments.Clear();
            AttachmentNames.Clear();
            AttachmentHint = string.Empty;
        }
        IsBusy = true;
        var userMessage = new MessageViewModel
        {
            Role = "user",
            Content = text,
            PromptContent = promptTextOverride ?? string.Empty,
            AttachmentNames = new ObservableCollection<string>(attachmentNames)
        };
        Messages.Add(userMessage);
        OnPropertyChanged(nameof(HasNoMessages));
        var assistantMessage = new MessageViewModel { Role = "assistant", IsPending = true };
        Messages.Add(assistantMessage);
        UpdateLatestAssistantAction();
        StatusText = "正在加载模型并生成回答…";
        var elapsed = Stopwatch.StartNew();
        using var streamUpdater = new StreamingMessageUpdater(
            assistantMessage, exception => _logger.LogError(exception, "Parsing streamed reply failed"));

        try
        {
            var result = await _services.Chat.SendAsync(
                ConversationId,
                ownerId,
                model,
                text,
                attachmentNames,
                attachments.Select(item => item.Content).ToArray(),
                config,
                streamUpdater.Append,
                generationCancellation.Token,
                promptTextOverride);

            elapsed.Stop();
            ConversationId = result.Conversation.Id;
            ConversationTitle = result.Conversation.Title;
            userMessage.PromptContent = result.PromptText;
            await streamUpdater.CompleteAsync(result.Assistant.Answer);
            assistantMessage.WasInterrupted = result.Assistant.Generation.WasCancelled;
            StatusText = result.Assistant.Generation.WasCancelled
                ? "已停止生成，本轮输入与部分回复已保存，可继续对话。"
                : FormatGenerationStatus(result.Assistant, elapsed.Elapsed);
            UpdateLatestAssistantAction();
            ConversationsChanged?.Invoke();
        }
        catch (Exception exception)
        {
            var canceled = exception is OperationCanceledException &&
                generationCancellation.IsCancellationRequested;
            if (canceled)
            {
                StatusText = "已停止生成，本轮对话未保存。";
            }
            else
            {
                _logger.LogError(exception, "Desktop chat generation or persistence failed");
                StatusText = $"本次请求失败：{exception.Message}";
            }
            streamUpdater.Stop();
            Messages.Remove(userMessage);
            Messages.Remove(assistantMessage);
            if (clearComposer)
            {
                InputText = text;
                foreach (var attachment in attachments)
                {
                    _attachments.Add(attachment);
                    AttachmentNames.Add(attachment.Name);
                }
            }
            UpdateLatestAssistantAction();
        }
        finally
        {
            if (ReferenceEquals(_generationCancellation, generationCancellation))
            {
                _generationCancellation = null;
            }
            IsBusy = false;
            OnPropertyChanged(nameof(CanSend));
        }
    }

    private static string FormatGenerationStatus(AssistantResult result, TimeSpan elapsed) =>
        $"生成 {result.Generation.TotalTokens} tokens · {result.Generation.TokensPerSecond:F1} tokens/s" +
        $" · 上下文 {Math.Max(0, result.HistoryTurns - 1)} 轮 · 本轮 {elapsed.TotalSeconds:F1}s";

    private void UpdateLatestAssistantAction()
    {
        foreach (var message in Messages)
        {
            message.CanRegenerate = false;
        }
        var latestMessage = Messages.LastOrDefault();
        if (latestMessage is { IsAssistant: true, IsPending: false })
        {
            latestMessage.CanRegenerate = true;
        }
    }

    partial void OnSelectedModelChanged(ModelDefinition? value)
    {
        OnPropertyChanged(nameof(CanSend));
        if (!_syncing && ConversationId is not null && value is not null)
        {
            StatusText = "模型已切换，后续消息会继续保存在当前对话中。";
        }
    }

    partial void OnIsBusyChanged(bool value)
    {
        OnPropertyChanged(nameof(CanSend));
        OnPropertyChanged(nameof(CanShowSend));
        OnPropertyChanged(nameof(CanStop));
        OnPropertyChanged(nameof(CanChangeConversation));
        _shell.OnChatBusyChanged();
        StopGenerationCommand.NotifyCanExecuteChanged();
    }

    private GenerationConfig BuildGenerationConfig() =>
        GenerationConfig.SceneConfigs[Scene.Chitchat] with
        {
            ThinkMode = IsThinking,
            ShowThinking = false,
            ThinkingBudget = ThinkingBudget,
            MultiTurn = true,
            MaxTurns = MaxHistoryTurns,
            ResetOnFactQuestion = false,
            MaxNewTokens = MaxNewTokens,
            Temperature = Temperature,
            TopP = TopP,
            TopK = TopK,
            RepetitionPenalty = RepetitionPenalty,
            DoSample = DoSample,
            Stream = StreamOutput,
            SystemPrompt = "You are a helpful assistant."
        };

    private static string GetStoredUserDisplayContent(string prompt)
    {
        const string attachmentMarker = "\n\n请参考以下文本附件内容：";
        var markerIndex = prompt.IndexOf(attachmentMarker, StringComparison.Ordinal);
        return markerIndex < 0 ? prompt : prompt[..markerIndex];
    }

    private sealed class StreamingMessageUpdater : IDisposable
    {
        private readonly MessageViewModel _message;
        private readonly Action<Exception> _onParseError;
        private readonly object _gate = new();
        private readonly StringBuilder _pendingContent = new();
        private readonly MessageViewModel.StreamingContentParser _parser = new();
        private readonly DispatcherTimer _timer;
        private bool _flushScheduled;
        private bool _stopped;

        public StreamingMessageUpdater(MessageViewModel message, Action<Exception> onParseError)
        {
            _message = message;
            _onParseError = onParseError;
            _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
            _timer.Tick += OnTimerTick;
            _timer.Start();
        }

        public void Append(string token)
        {
            lock (_gate)
            {
                if (!_stopped)
                {
                    _pendingContent.Append(token);
                }
            }
        }

        public async Task CompleteAsync(string finalContent)
        {
            Stop();
            var parsed = await Task.Run(
                () => MessageViewModel.ParseStreamingContent(finalContent));
            await Dispatcher.UIThread.InvokeAsync(
                () =>
                {
                    _message.ApplyStreamingSnapshot(parsed.Thinking, parsed.Answer);
                    _message.Content = finalContent;
                    _message.IsPending = false;
                });
        }

        public void Stop()
        {
            lock (_gate)
            {
                _stopped = true;
            }
            _timer.Stop();
            _timer.Tick -= OnTimerTick;
        }

        public void Dispose() => Stop();

        private void OnTimerTick(object? sender, EventArgs eventArgs)
        {
            string snapshot;
            lock (_gate)
            {
                if (_stopped || _flushScheduled || _pendingContent.Length == 0)
                {
                    return;
                }
                _flushScheduled = true;
                snapshot = _pendingContent.ToString();
                _pendingContent.Clear();
            }

            _ = ParseSnapshotAsync(snapshot);
        }

        private async Task ParseSnapshotAsync(string snapshot)
        {
            (string Thinking, string Answer) parsed;
            try
            {
                parsed = await Task.Run(() => _parser.Append(snapshot));
            }
            catch (Exception exception)
            {
                lock (_gate)
                {
                    _flushScheduled = false;
                }
                _onParseError(exception);
                return;
            }

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                var applySnapshot = false;
                lock (_gate)
                {
                    applySnapshot = !_stopped;
                    _flushScheduled = false;
                }
                if (applySnapshot)
                {
                    _message.ApplyStreamingSnapshot(parsed.Thinking, parsed.Answer);
                }
            });
        }
    }
}