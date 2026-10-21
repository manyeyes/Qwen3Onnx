using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Qwen3.Desktop.Services;
using Qwen3.Desktop.ViewModels;
using Qwen3.Desktop.Views;

namespace Qwen3.Desktop;

public sealed partial class App : Application
{
    private DesktopServices? _services;

    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            _services = new DesktopServices();
            var bootstrap = _services.Database.InitializeAsync().GetAwaiter().GetResult();
            desktop.MainWindow = new MainWindow
            {
                DataContext = new MainWindowViewModel(_services, bootstrap)
            };
            desktop.Exit += (_, _) => _services.Dispose();
        }

        base.OnFrameworkInitializationCompleted();
    }
}
