using Avalonia.Controls;

namespace Onar.Desktop.Views;

public sealed partial class ProfileView : UserControl
{
    public ProfileView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
