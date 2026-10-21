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
    public string DisplayRole => IsUser ? "用户" : "助手";

    public bool IsUser => Role == "user";

    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void ToggleThinking() => IsThinkingExpanded = !IsThinkingExpanded;

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsUser));
        OnPropertyChanged(nameof(IsAssistant));
        OnPropertyChanged(nameof(DisplayRole));
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
    }

    partial void OnIsPendingChanged(bool value)
    {
        OnPropertyChanged(nameof(IsNotPending));
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
