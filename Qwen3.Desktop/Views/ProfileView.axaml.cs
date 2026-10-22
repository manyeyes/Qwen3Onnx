using Avalonia.Controls;

namespace Qwen3.Desktop.Views;

public sealed partial class ProfileView : UserControl
{
    public ProfileView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
