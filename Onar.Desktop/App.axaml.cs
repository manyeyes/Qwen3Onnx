using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Onar.Desktop.Services;
using Onar.Desktop.ViewModels;
using Onar.Desktop.Views;

namespace Onar.Desktop;

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
