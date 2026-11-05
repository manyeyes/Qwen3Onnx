using Avalonia.Controls;

namespace Onar.Desktop.Views;

public sealed partial class UsersView : UserControl
{
    public UsersView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
