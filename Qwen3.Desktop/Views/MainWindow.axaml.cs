using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.VisualTree;
using Qwen3.Desktop.ViewModels;

namespace Qwen3.Desktop.Views;

public sealed partial class MainWindow : Window
{
    private Border? _bottomMenuBorder;
    private Button? _bottomMenuButton;

    public MainWindow()
    {
        InitializeComponent();

        // 手写 InitializeComponent 时不走名称生成器，需手动查找
        _bottomMenuBorder = this.FindControl<Border>("BottomMenuBorder");
        _bottomMenuButton = this.FindControl<Button>("BottomMenuButton");

        // 隧道阶段拦截，先于按钮 Click 执行
        AddHandler(PointerPressedEvent, OnWindowPointerPressed, RoutingStrategies.Tunnel);
        // 键盘 Tab 移出焦点时兜底
        AddHandler(GotFocusEvent, OnWindowGotFocus, RoutingStrategies.Bubble);
        // 窗口失活（切到其他程序、最小化等）时收起
        Deactivated += (_, _) => CloseBottomMenu();

        DataContextChanged += (_, _) =>
        {
            if (DataContext is MainWindowViewModel viewModel)
            {
                viewModel.Chat.SelectTextFilesAsync = SelectTextFilesAsync;
                viewModel.Chat.CopyTextAsync = CopyTextAsync;
            }
        };
    }

    private void OnWindowPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (!IsBottomMenuOpen() || e.Source is not Visual source)
        {
            return;
        }

        if (IsInsideBottomMenuArea(source))
        {
            return;
        }

        CloseBottomMenu();
    }

    private void OnWindowGotFocus(object? sender, FocusChangedEventArgs e)
    {
        if (!IsBottomMenuOpen() || e.Source is not Visual source)
        {
            return;
        }

        if (IsInsideBottomMenuArea(source))
        {
            return;
        }

        CloseBottomMenu();
    }

    private bool IsInsideBottomMenuArea(Visual source)
    {
        var insideMenu = _bottomMenuBorder is not null &&
                         (ReferenceEquals(source, _bottomMenuBorder) ||
                          _bottomMenuBorder.IsVisualAncestorOf(source));
        var insideToggle = _bottomMenuButton is not null &&
                           (ReferenceEquals(source, _bottomMenuButton) ||
                            _bottomMenuButton.IsVisualAncestorOf(source));
        return insideMenu || insideToggle;
    }

    private bool IsBottomMenuOpen() =>
        DataContext is MainWindowViewModel { IsBottomMenuOpen: true };

    private void CloseBottomMenu()
    {
        if (DataContext is MainWindowViewModel viewModel)
        {
            viewModel.IsBottomMenuOpen = false;
        }
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