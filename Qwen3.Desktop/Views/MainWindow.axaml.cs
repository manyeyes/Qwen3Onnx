using Avalonia.Controls;
using Avalonia.Input.Platform;
using Avalonia.Platform.Storage;
using Qwen3.Desktop.ViewModels;

namespace Qwen3.Desktop.Views;

public sealed partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContextChanged += (_, _) =>
        {
            if (DataContext is MainWindowViewModel viewModel)
            {
                viewModel.Chat.SelectTextFilesAsync = SelectTextFilesAsync;
                viewModel.Chat.CopyTextAsync = CopyTextAsync;
            }
        };
    }

    private Task CopyTextAsync(string text)
    {
        var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;
        if (clipboard is null)
        {
            throw new InvalidOperationException("系统剪贴板在当前窗口不可用。");
        }
        return clipboard.SetTextAsync(text);
    }

    private async Task<IReadOnlyList<string>> SelectTextFilesAsync()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "选择文本附件",
            AllowMultiple = true,
            FileTypeFilter =
            [
                new FilePickerFileType("文本与源码")
                {
                    Patterns =
                    [
                        "*.txt", "*.md", "*.markdown", "*.json", "*.csv", "*.tsv", "*.xml",
                        "*.yaml", "*.yml", "*.log", "*.ini", "*.toml", "*.cs", "*.py", "*.js",
                        "*.ts", "*.html", "*.css", "*.sql"
                    ]
                }
            ]
        });
        return files.Select(file => file.TryGetLocalPath())
            .Where(path => path is not null)
            .Select(path => path!)
            .ToArray();
    }

    private void InitializeComponent() =>
        Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
