using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Qwen3.Desktop.Models;

namespace Qwen3.Desktop.Services;

public sealed class ModelDownloadService(HttpClient httpClient)
{
    private const string UserAgent = "Qwen3OnnxDesktop/1.0";

    public async Task<ModelDownloadResult> DownloadAsync(
        ModelDefinition model,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (Directory.Exists(model.LocalPath))
        {
            throw new IOException($"目标目录已存在；请先检查或删除它：{model.LocalPath}");
        }

        var parent = Path.GetDirectoryName(model.LocalPath)
            ?? throw new InvalidOperationException("Model destination has no parent directory.");
        Directory.CreateDirectory(parent);
        var stagingPath = Path.Combine(parent, $".{Path.GetFileName(model.LocalPath)}-{Guid.NewGuid():N}.partial");
        Directory.CreateDirectory(stagingPath);
        try
        {
            var cloneCompleted = await TryGitCloneAsync(
                model.RepositoryUrl, stagingPath, progress, cancellationToken);
            RepositoryManifest manifest;
            if (cloneCompleted)
            {
                manifest = InspectLocalManifest(stagingPath);
                var hasLfsPointers = manifest.Files.Any(file =>
                    IsGitLfsPointer(ResolveInsideDirectory(stagingPath, file.Path)));
                if (hasLfsPointers || !HasRequiredModelFiles(manifest))
                {
                    if (model.HttpFileLinks.Count == 0 &&
                        !TryGetModelScopeRepository(model.RepositoryUrl, out _))
                    {
                        throw new IOException(
                            "The Git clone is missing required model files or contains Git LFS pointers, and no HTTP fallback is configured.");
                    }
                    progress?.Report(new ModelDownloadProgress(
                        "Git 文件不完整或包含 LFS 指针，保留已克隆文件并通过 HTTP 补齐…", 0, 0, null));
                    manifest = await DownloadRepositoryOverHttpAsync(
                        model, stagingPath, progress, cancellationToken);
                }
            }
            else
            {
                if (model.HttpFileLinks.Count == 0 && string.IsNullOrWhiteSpace(model.RepositoryUrl))
                {
                    throw new NotSupportedException("No direct file links or Git repository URL were configured.");
                }
                manifest = await DownloadRepositoryOverHttpAsync(model, stagingPath, progress, cancellationToken);
            }

            progress?.Report(new ModelDownloadProgress("正在校验必需模型文件…", 0, 0, null));
            ValidateModelFiles(stagingPath, manifest);
            var completeManifest = manifest.Files.Select(file => file.Path).ToArray();
            var (embeddingFile, decoderFile) = FindGraphFiles(completeManifest);
            Directory.Move(stagingPath, model.LocalPath);
            var cleanupWarning = await TryDeleteGitMetadataAsync(model.LocalPath, progress);
            return new ModelDownloadResult(model with
            {
                IsInstalled = true,
                EmbeddingFileName = embeddingFile,
                DecoderFileName = decoderFile,
                RequiredFiles = completeManifest
            }, cleanupWarning);
        }
        catch (Exception exception)
        {
            progress?.Report(new ModelDownloadProgress(
                $"下载或校验失败，临时文件已保留以便排查：{stagingPath}（{exception.Message}）", 0, 0, null));
            if (exception is OperationCanceledException)
            {
                throw;
            }
            throw new IOException(
                $"Model download or validation failed; temporary files were retained at: {stagingPath}",
                exception);
        }
    }

    private static async Task<string?> TryDeleteGitMetadataAsync(
        string installedPath,
        IProgress<ModelDownloadProgress>? progress)
    {
        var gitDirectory = Path.Combine(installedPath, ".git");
        if (!Directory.Exists(gitDirectory))
        {
            return null;
        }

        Exception? lastError = null;
        for (var attempt = 0; attempt < 3; attempt++)
        {
            try
            {
                if (attempt > 0)
                {
                    await Task.Delay(TimeSpan.FromMilliseconds(250 * attempt));
                }
                Directory.Delete(gitDirectory, recursive: true);
                return null;
            }
            catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
            {
                lastError = exception;
            }
        }

        var warning = $"Git 元数据未能清理，已保留在 {gitDirectory}：{lastError?.Message}";
        progress?.Report(new ModelDownloadProgress(warning, 0, 0, 100));
        return warning;
    }

    public static bool IsInstalled(ModelDefinition model)
    {
        if (!Directory.Exists(model.LocalPath))
        {
            return false;
        }

        var requiredPaths = new[]
        {
            "config.json",
            "tokenizer.json",
            model.EmbeddingFileName,
            model.DecoderFileName
        };
        return requiredPaths.All(relativePath =>
        {
            var path = Path.Combine(model.LocalPath, relativePath);
            return File.Exists(path) && !IsGitLfsPointer(path);
        });
    }

    private async Task<bool> TryGitCloneAsync(
        string repositoryUrl,
        string destination,
        IProgress<ModelDownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var cloneUrl = GetGitCloneUrl(repositoryUrl);
        if (cloneUrl is null)
        {
            return false;
        }

        try
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "git",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                },
                EnableRaisingEvents = true
            };
            process.StartInfo.Environment["GIT_TERMINAL_PROMPT"] = "0";
            process.StartInfo.Environment["GCM_INTERACTIVE"] = "Never";
            process.StartInfo.ArgumentList.Add("clone");
            process.StartInfo.ArgumentList.Add("--progress");
            process.StartInfo.ArgumentList.Add("--depth");
            process.StartInfo.ArgumentList.Add("1");
            process.StartInfo.ArgumentList.Add(cloneUrl);
            process.StartInfo.ArgumentList.Add(destination);

            progress?.Report(new ModelDownloadProgress("正在使用 Git 克隆模型仓库…", 0, 0, null));
            if (!process.Start())
            {
                return false;
            }

            var errors = new List<string>();
            process.ErrorDataReceived += (_, eventArgs) =>
            {
                if (eventArgs.Data is not { Length: > 0 } line)
                {
                    return;
                }
                lock (errors)
                {
                    if (errors.Count < 20)
                    {
                        errors.Add(line);
                    }
                }
                var match = Regex.Match(line, @"(?<percent>\d{1,3})%");
                if (match.Success && double.TryParse(match.Groups["percent"].Value, out var percent))
                {
                    progress?.Report(new ModelDownloadProgress(
                        $"Git 下载中：{line.Trim()}", 0, 0, Math.Clamp(percent, 0, 100)));
                }
            };
            process.BeginErrorReadLine();
            process.BeginOutputReadLine();
            using var registration = cancellationToken.Register(() =>
            {
                try
                {
                    if (!process.HasExited)
                    {
                        process.Kill(entireProcessTree: true);
                    }
                }
                catch (InvalidOperationException)
                {
                }
            });
            await process.WaitForExitAsync(cancellationToken);
            if (process.ExitCode == 0)
            {
                progress?.Report(new ModelDownloadProgress("Git 克隆完成，正在校验模型文件…", 0, 0, 95));
                return true;
            }

            progress?.Report(new ModelDownloadProgress(
                $"Git 克隆失败，尝试 HTTP 后备下载：{string.Join(" | ", errors.TakeLast(3))}", 0, 0, null));
            return false;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            progress?.Report(new ModelDownloadProgress("未检测到 Git，改用 HTTP 下载…", 0, 0, null));
            return false;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (IOException)
        {
            progress?.Report(new ModelDownloadProgress(
                $"Git 操作失败，保留已下载文件并尝试 HTTP 补齐：{destination}", 0, 0, null));
            return false;
        }
    }

    private async Task<RepositoryManifest> DownloadRepositoryOverHttpAsync(
        ModelDefinition model,
        string destination,
        IProgress<ModelDownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        RepositoryIdentifier repository = default;
        List<RepositoryFile> files;
        if (model.HttpFileLinks.Count > 0)
        {
            files = model.HttpFileLinks.Select(link =>
                new RepositoryFile(link.Path.Replace('\\', '/'), link.Size, link.Sha256, link.Url)).ToList();
        }
        else
        {
            if (!TryGetModelScopeRepository(model.RepositoryUrl, out repository))
            {
                throw new NotSupportedException(
                    "HTTP fallback requires direct file links or a ModelScope repository manifest.");
            }

            using var manifestRequest = new HttpRequestMessage(
                HttpMethod.Get,
                $"https://www.modelscope.cn/api/v1/models/{Uri.EscapeDataString(repository.Owner)}/" +
                $"{Uri.EscapeDataString(repository.Name)}/repo/files?Revision=master");
            manifestRequest.Headers.UserAgent.ParseAdd(UserAgent);
            using var manifestResponse = await httpClient.SendAsync(
                manifestRequest, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            manifestResponse.EnsureSuccessStatusCode();
            await using var manifestStream = await manifestResponse.Content.ReadAsStreamAsync(cancellationToken);
            using var document = await JsonDocument.ParseAsync(manifestStream, cancellationToken: cancellationToken);
            var root = document.RootElement;
            if (root.TryGetProperty("Code", out var code) && code.ToString() is not ("200" or "0"))
            {
                throw new HttpRequestException($"ModelScope file-list API returned code {code}.");
            }
            if (!root.TryGetProperty("Data", out var data) || !data.TryGetProperty("Files", out var filesElement))
            {
                throw new InvalidDataException("ModelScope returned an invalid repository file manifest.");
            }

            files = [];
            foreach (var item in filesElement.EnumerateArray())
            {
                if (!item.TryGetProperty("Path", out var pathElement) ||
                    !item.TryGetProperty("Type", out var typeElement) ||
                    !string.Equals(typeElement.GetString(), "blob", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                var path = pathElement.GetString();
                if (string.IsNullOrWhiteSpace(path))
                {
                    continue;
                }
                ValidateRelativePath(path);
                var size = item.TryGetProperty("Size", out var sizeElement) &&
                           sizeElement.TryGetInt64(out var parsedSize) ? parsedSize : 0;
                var sha = item.TryGetProperty("Sha256", out var shaElement) ? shaElement.GetString() : null;
                files.Add(new RepositoryFile(path, size, sha));
            }
        }
        if (files.Count == 0)
        {
            throw new InvalidDataException("ModelScope repository contains no downloadable files.");
        }

        var totalBytes = files.Sum(file => Math.Max(0, file.Size));
        long received = 0;
        for (var index = 0; index < files.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var file = files[index];
            var destinationPath = ResolveInsideDirectory(destination, file.Path);
            if (File.Exists(destinationPath))
            {
                var existingInfo = new FileInfo(destinationPath);
                var isComplete = !IsGitLfsPointer(destinationPath) &&
                                 (file.Size <= 0 || existingInfo.Length == file.Size);
                if (isComplete)
                {
                    received += existingInfo.Length;
                    progress?.Report(new ModelDownloadProgress(
                        $"已复用克隆文件 {index + 1}/{files.Count}：{file.Path}",
                        received, totalBytes, totalBytes > 0 ? received * 100.0 / totalBytes : null));
                    continue;
                }
            }
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            var relativeFile = string.Join("/", file.Path.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                .Select(Uri.EscapeDataString));
            var url = file.Url ??
                      $"https://www.modelscope.cn/models/{Uri.EscapeDataString(repository.Owner)}/" +
                      $"{Uri.EscapeDataString(repository.Name)}/resolve/master/{relativeFile}";
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            request.Headers.UserAgent.ParseAdd(UserAgent);
            using var response = await httpClient.SendAsync(
                request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();
            var expectedSize = file.Size > 0
                ? file.Size
                : response.Content.Headers.ContentLength ?? 0;
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var output = new FileStream(
                destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024,
                FileOptions.Asynchronous | FileOptions.SequentialScan);
            using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            var buffer = new byte[1024 * 1024];
            long fileBytes = 0;
            while (true)
            {
                var read = await source.ReadAsync(buffer, cancellationToken);
                if (read == 0)
                {
                    break;
                }
                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                digest.AppendData(buffer, 0, read);
                fileBytes += read;
                var overall = received + fileBytes;
                var percentage = totalBytes > 0 ? overall * 100.0 / totalBytes : (double?)null;
                progress?.Report(new ModelDownloadProgress(
                    $"下载 {index + 1}/{files.Count}：{file.Path}", overall, totalBytes, percentage));
            }
            await output.FlushAsync(cancellationToken);
            if (expectedSize > 0 && fileBytes != expectedSize)
            {
                throw new InvalidDataException(
                    $"Downloaded size mismatch for {file.Path}: expected {expectedSize}, received {fileBytes}.");
            }
            if (!string.IsNullOrWhiteSpace(file.Sha256))
            {
                var actualHash = Convert.ToHexString(digest.GetHashAndReset());
                if (!CryptographicOperations.FixedTimeEquals(
                        Convert.FromHexString(actualHash), Convert.FromHexString(file.Sha256)))
                {
                    throw new InvalidDataException($"SHA-256 verification failed for {file.Path}.");
                }
            }
            received += fileBytes;
        }

        return new RepositoryManifest(files);
    }

    private static RepositoryManifest InspectLocalManifest(string directory)
    {
        var files = new List<RepositoryFile>();
        var rootNames = new[] { "config.json", "tokenizer.json" };
        foreach (var name in rootNames)
        {
            var path = Path.Combine(directory, name);
            if (File.Exists(path))
            {
                files.Add(new RepositoryFile(name, new FileInfo(path).Length, null));
            }
        }
        foreach (var path in Directory.EnumerateFiles(directory, "*.onnx", SearchOption.TopDirectoryOnly))
        {
            var name = Path.GetFileName(path);
            if (IsGraphFile(name, "embedding") || IsGraphFile(name, "decoder"))
            {
                files.Add(new RepositoryFile(name, new FileInfo(path).Length, null));
            }
        }
        return new RepositoryManifest(files);
    }

    private static void ValidateModelFiles(string directory, RepositoryManifest manifest)
    {
        var requiredPaths = new[]
        {
            "config.json",
            "tokenizer.json",
            manifest.Files.FirstOrDefault(file => IsGraphFile(file.Path, "embedding"))?.Path,
            manifest.Files.FirstOrDefault(file => IsGraphFile(file.Path, "decoder"))?.Path
        };
        if (requiredPaths.Any(string.IsNullOrWhiteSpace))
        {
            throw new InvalidDataException(
                "Downloaded model is incomplete. Required files: config.json, tokenizer.json, an embedding*.onnx and a decoder*.onnx file.");
        }

        foreach (var relativePath in requiredPaths)
        {
            var path = ResolveInsideDirectory(directory, relativePath!);
            if (!File.Exists(path))
            {
                throw new InvalidDataException($"Required model file is missing: {relativePath}");
            }
            var file = manifest.Files.FirstOrDefault(item =>
                string.Equals(item.Path, relativePath, StringComparison.OrdinalIgnoreCase));
            if (file is not null && file.Size > 0 && new FileInfo(path).Length != file.Size)
            {
                throw new InvalidDataException($"Required model file is incomplete: {relativePath}");
            }
            if (IsGitLfsPointer(path))
            {
                throw new InvalidDataException($"Required model file is still a Git LFS pointer: {relativePath}");
            }
        }
    }

    private static bool HasRequiredModelFiles(RepositoryManifest manifest) =>
        manifest.Files.Any(file => string.Equals(file.Path, "config.json", StringComparison.OrdinalIgnoreCase)) &&
        manifest.Files.Any(file => string.Equals(file.Path, "tokenizer.json", StringComparison.OrdinalIgnoreCase)) &&
        manifest.Files.Any(file => IsGraphFile(file.Path, "embedding")) &&
        manifest.Files.Any(file => IsGraphFile(file.Path, "decoder"));

    private static (string Embedding, string Decoder) FindGraphFiles(IEnumerable<string> paths)
    {
        var embedding = paths.FirstOrDefault(path => IsGraphFile(path, "embedding"));
        var decoder = paths.FirstOrDefault(path => IsGraphFile(path, "decoder"));
        if (embedding is null || decoder is null)
        {
            throw new InvalidDataException("Could not locate embedding and decoder ONNX graphs.");
        }
        return (Path.GetFileName(embedding), Path.GetFileName(decoder));
    }

    private static bool IsGraphFile(string path, string prefix) =>
        Path.GetFileName(path).StartsWith(prefix, StringComparison.OrdinalIgnoreCase) &&
        Path.GetFileName(path).EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);

    private static bool IsGitLfsPointer(string path)
    {
        if (!File.Exists(path) || new FileInfo(path).Length > 1024)
        {
            return false;
        }
        using var reader = new StreamReader(path);
        return reader.ReadLine()?.StartsWith("version https://git-lfs.github.com/spec/v1", StringComparison.Ordinal) == true;
    }

    private static string? GetGitCloneUrl(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
            uri.Scheme is not ("http" or "https"))
        {
            return null;
        }

        if (uri.Host.EndsWith("modelscope.cn", StringComparison.OrdinalIgnoreCase))
        {
            var segments = uri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
            var modelsIndex = Array.FindIndex(segments, segment => segment.Equals("models", StringComparison.OrdinalIgnoreCase));
            if (modelsIndex >= 0 && segments.Length > modelsIndex + 2)
            {
                return $"https://www.modelscope.cn/{segments[modelsIndex + 1]}/{segments[modelsIndex + 2]}.git";
            }
            if (segments.Length >= 2)
            {
                var repo = segments[^1].EndsWith(".git", StringComparison.OrdinalIgnoreCase)
                    ? segments[^1] : segments[^1] + ".git";
                return $"https://www.modelscope.cn/{segments[^2]}/{repo}";
            }
        }
        return url;
    }

    private static bool TryGetModelScopeRepository(string url, out RepositoryIdentifier repository)
    {
        repository = default;
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
            !uri.Host.EndsWith("modelscope.cn", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }
        var segments = uri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
        var modelsIndex = Array.FindIndex(segments, segment =>
            segment.Equals("models", StringComparison.OrdinalIgnoreCase));
        if (modelsIndex >= 0 && segments.Length > modelsIndex + 2)
        {
            repository = new RepositoryIdentifier(segments[modelsIndex + 1], segments[modelsIndex + 2]);
            return true;
        }
        if (segments.Length >= 2)
        {
            repository = new RepositoryIdentifier(segments[^2], segments[^1].Replace(".git", string.Empty, StringComparison.OrdinalIgnoreCase));
            return true;
        }
        return false;
    }

    private static string ResolveInsideDirectory(string root, string relativePath)
    {
        ValidateRelativePath(relativePath);
        var fullRoot = Path.GetFullPath(root) + Path.DirectorySeparatorChar;
        var fullPath = Path.GetFullPath(Path.Combine(root,
            relativePath.Replace('/', Path.DirectorySeparatorChar)));
        if (!fullPath.StartsWith(fullRoot,
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
        {
            throw new InvalidDataException("Repository file path escapes the model destination.");
        }
        return fullPath;
    }

    private static void ValidateRelativePath(string path)
    {
        if (Path.IsPathRooted(path) ||
            path.Split('/', '\\').Any(segment => segment is "." or "..") ||
            path.Contains('\0'))
        {
            throw new InvalidDataException($"Unsafe repository path: {path}");
        }
    }

    private readonly record struct RepositoryIdentifier(string Owner, string Name);
    private sealed record RepositoryFile(string Path, long Size, string? Sha256, string? Url = null);
    private sealed record RepositoryManifest(IReadOnlyList<RepositoryFile> Files);
}
