using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Text.RegularExpressions;

namespace Qwen3.Desktop.ViewModels;

public partial class MessageViewModel : ObservableObject
{
    [ObservableProperty]
    private string role = "assistant";

    [ObservableProperty]
    private string content = string.Empty;

    [ObservableProperty]
    private string attachmentsText = string.Empty;

    [ObservableProperty]
    private ObservableCollection<string> attachmentNames = [];

    [ObservableProperty]
    private bool isPending;

    [ObservableProperty]
    private bool isThinkingExpanded;

    [ObservableProperty]
    private string promptContent = string.Empty;

    [ObservableProperty]
    private bool canRegenerate;

    [ObservableProperty]
    private bool wasInterrupted;

    [ObservableProperty]
    private bool isTextSelectionMode;

    public string InterruptionLabel => "已停止 · 部分回复已保存";
    public int BubbleColumn => IsUser ? 1 : 0;
    public int BubbleColumnSpan => IsUser ? 1 : 3;
    public double BubbleMaxWidth => IsUser ? 880 : double.PositiveInfinity;
    public Avalonia.Layout.HorizontalAlignment BubbleAlignment =>
        IsUser ? Avalonia.Layout.HorizontalAlignment.Right : Avalonia.Layout.HorizontalAlignment.Stretch;
    public Avalonia.Thickness BubbleMargin => IsUser
        ? new Avalonia.Thickness(64, 0, 8, 0)
        : new Avalonia.Thickness(8, 0, 8, 0);
    public string SelectionModeTooltip => IsTextSelectionMode ? "返回 Markdown 显示" : "切换为可选择文本";

    public string MarkdownContent
    {
        get => ExtractAnswer(Content);
    }

    public string StreamingAnswer => ExtractAnswer(Content);

    partial void OnAttachmentNamesChanged(ObservableCollection<string> value) =>
        AttachmentsText = string.Join(", ", value);

    public string ThinkingContent
    {
        get
        {
            var normalized = NormalizeThinkTags(Content);
            var start = normalized.IndexOf("<think>", StringComparison.Ordinal);
            if (start < 0)
            {
                return string.Empty;
            }
            start += "<think>".Length;
            var end = normalized.IndexOf("</think>", start, StringComparison.Ordinal);
            return (end < 0 ? normalized[start..] : normalized[start..end]).Trim();
        }
    }

    public bool HasThinking => ThinkingContent.Length > 0;
    public bool HasAttachments => !string.IsNullOrWhiteSpace(AttachmentsText);
    public bool IsAssistant => !IsUser;
    public bool IsNotPending => !IsPending;
    public bool IsMarkdownVisible => IsNotPending && IsAssistant && !IsTextSelectionMode;
    public string DisplayRole => IsUser ? "用户" : "助手";
    public string SelectionModeGlyph => IsTextSelectionMode ? "¶" : "▤";

    public bool IsUser => Role == "user";

    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void ToggleThinking() => IsThinkingExpanded = !IsThinkingExpanded;

    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void ToggleTextSelectionMode() => IsTextSelectionMode = !IsTextSelectionMode;

    partial void OnIsTextSelectionModeChanged(bool value) =>
        OnTextSelectionModeChanged();

    private void OnTextSelectionModeChanged()
    {
        OnPropertyChanged(nameof(SelectionModeTooltip));
        OnPropertyChanged(nameof(SelectionModeGlyph));
        OnPropertyChanged(nameof(IsMarkdownVisible));
    }

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsUser));
        OnPropertyChanged(nameof(IsAssistant));
        OnPropertyChanged(nameof(DisplayRole));
        OnPropertyChanged(nameof(BubbleColumn));
        OnPropertyChanged(nameof(BubbleAlignment));
        OnPropertyChanged(nameof(BubbleMargin));
        OnPropertyChanged(nameof(IsMarkdownVisible));
        OnPropertyChanged(nameof(BubbleMaxWidth));
        OnPropertyChanged(nameof(BubbleColumnSpan));
    }

    partial void OnContentChanged(string value)
    {
        OnPropertyChanged(nameof(ThinkingContent));
        OnPropertyChanged(nameof(HasThinking));
        if (IsPending)
        {
            OnPropertyChanged(nameof(StreamingAnswer));
        }
        else
        {
            OnPropertyChanged(nameof(MarkdownContent));
        }
        OnPropertyChanged(nameof(IsMarkdownVisible));
    }

    partial void OnIsPendingChanged(bool value)
    {
        OnPropertyChanged(nameof(IsNotPending));
        OnPropertyChanged(nameof(IsMarkdownVisible));
        OnPropertyChanged(value ? nameof(StreamingAnswer) : nameof(MarkdownContent));
    }

    partial void OnIsThinkingExpandedChanged(bool value)
    {
        OnPropertyChanged(nameof(ThinkingMaxHeight));
        OnPropertyChanged(nameof(ThinkingToggleText));
    }

    public double ThinkingMaxHeight => IsThinkingExpanded ? 240 : 112;
    public string ThinkingToggleText => IsThinkingExpanded ? "收起思考" : "展开思考";

    private static string NormalizeThinkTags(string text) =>
        Regex.Replace(text.Replace("<|think|>", "<think>", StringComparison.Ordinal)
            .Replace("<|/think|>", "</think>", StringComparison.Ordinal),
            @"</?think[^>]*>", match => match.Value.StartsWith("</", StringComparison.Ordinal)
                ? "</think>"
                : "<think>", RegexOptions.CultureInvariant);

    private static string ExtractAnswer(string content)
    {
        var normalized = NormalizeThinkTags(content);
        var closingIndex = normalized.LastIndexOf("</think>", StringComparison.Ordinal);
        if (closingIndex >= 0)
        {
            normalized = normalized[(closingIndex + "</think>".Length)..];
        }
        else
        {
            var openingIndex = normalized.IndexOf("<think>", StringComparison.Ordinal);
            if (openingIndex >= 0)
            {
                normalized = normalized[..openingIndex];
            }
        }
        return Regex.Replace(normalized, @"<\|im_(?:start|end)\|>", string.Empty).Trim();
    }
}
