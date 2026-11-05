using Avalonia.Controls;
using Avalonia.Threading;
using Qwen3.Desktop.ViewModels;

namespace Qwen3.Desktop.Views;

public sealed partial class ChatView : UserControl
{
    // �Ƿ��Զ����浽�ײ����û��ֶ��Ϲ����� false�����صײ���㰴ť���� true��
    private bool _autoScroll = true;
    // ��ֹ���򻯹���ʱ�ݹ鴥�� ScrollChanged
    private bool _suppressScrollChanged;
    // ��Ϣ���϶��ģ�����"������Ϣ"ʱǿ�ƹ�����
    private ChatPageViewModel? _subscribedViewModel;

    public ChatView()
    {
        Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
        DataContextChanged += OnDataContextChanged;
        // ������ɺ��ȹ�һ�ε��ײ�
        AttachedToVisualTree += (_, _) => Dispatcher.UIThread.Post(ScrollToBottomIfAuto, DispatcherPriority.Loaded);
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.Messages.CollectionChanged -= OnMessagesCollectionChanged;
            _subscribedViewModel.PropertyChanged -= OnViewModelPropertyChanged;
        }

        _subscribedViewModel = DataContext as ChatPageViewModel;

        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.Messages.CollectionChanged += OnMessagesCollectionChanged;
            _subscribedViewModel.PropertyChanged += OnViewModelPropertyChanged;
        }
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ChatPageViewModel.IsBusy) &&
            _subscribedViewModel is { IsBusy: false })
        {
            Dispatcher.UIThread.Post(FocusInput, DispatcherPriority.Background);
        }

        // �Ự�л� / ���¼�����Ϣʱ��ǿ�Ƹ��浽��
        if (e.PropertyName == nameof(ChatPageViewModel.Messages))
        {
            if (_subscribedViewModel is not null)
            {
                _subscribedViewModel.Messages.CollectionChanged -= OnMessagesCollectionChanged;
                _subscribedViewModel.Messages.CollectionChanged += OnMessagesCollectionChanged;
            }
            _autoScroll = true;
            Dispatcher.UIThread.Post(ScrollToBottom, DispatcherPriority.Loaded);
        }
    }

    private void FocusInput()
    {
        if (!IsVisible || VisualRoot is null ||
            this.FindControl<TextBox>("InputTextBox") is not { IsEnabled: true } input)
        {
            return;
        }

        input.Focus();
    }

    private void OnMessagesCollectionChanged(
        object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
    {
        // ������Ϣ�����ͻ��յ��ظ���ʱ��ǿ�ƹ����ײ��ָ��Զ�����
        if (e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Add)
        {
            _autoScroll = true;
            Dispatcher.UIThread.Post(ScrollToBottom, DispatcherPriority.Loaded);
        }
    }

    private void MessageScrollViewer_OnScrollChanged(object? sender, ScrollChangedEventArgs e)
    {
        if (_suppressScrollChanged || sender is not ScrollViewer scrollViewer)
        {
            return;
        }

        UpdateScrollState(scrollViewer);

        // ��ʽ���ʱ����������Զ�����״̬������������
        if (_autoScroll)
        {
            ScrollToBottom();
        }
    }

    private void ScrollToBottomButton_OnClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        _autoScroll = true;
        ScrollToBottom();
    }

    private void ScrollToBottomIfAuto()
    {
        if (_autoScroll)
        {
            ScrollToBottom();
        }
    }

    private void ScrollToBottom()
    {
        if (this.FindControl<ScrollViewer>("MessageScrollViewer") is not { } scrollViewer)
        {
            return;
        }

        _suppressScrollChanged = true;
        try
        {
            scrollViewer.ScrollToEnd();
        }
        finally
        {
            // ���򻯹�����ɺ�����ˢ��һ��״̬������ť�ɼ��ԣ���
            // ��Ϊ ScrollChanged �ѱ����ƣ����������������¡�
            Dispatcher.UIThread.Post(
                () =>
                {
                    _suppressScrollChanged = false;
                    UpdateScrollState(scrollViewer);
                },
                DispatcherPriority.Background);
        }
    }

    private void UpdateScrollState(ScrollViewer scrollViewer)
    {
        var scrollableHeight = scrollViewer.Extent.Height - scrollViewer.Viewport.Height;
        var distanceFromBottom = scrollableHeight - scrollViewer.Offset.Y;

        // ��ײ�С�� 24px ��Ϊ"�ڵײ�"
        var atBottom = distanceFromBottom <= 24;

        if (atBottom)
        {
            // �û����صײ� �� �ָ��Զ�����
            _autoScroll = true;
        }
        else
        {
            // �û��ֶ��Ϲ� �� ֹͣ����
            _autoScroll = false;
        }

        //if (this.FindControl<Button>("ScrollToBottomButton") is { } button)
        //{
        //    button.IsVisible = !atBottom;
        //}
        // ��Ϊ���� Border �Ŀɼ��ԣ�
        if (this.FindControl<Border>("ScrollToBottomBorder") is { } border)
        {
            border.IsVisible = !atBottom;
        }
    }

    private void ThinkingScrollViewer_OnScrollChanged(object? sender, ScrollChangedEventArgs eventArgs)
    {
        if (sender is ScrollViewer { DataContext: MessageViewModel { IsPending: true } } scrollViewer)
        {
            scrollViewer.ScrollToEnd();
        }
    }
}