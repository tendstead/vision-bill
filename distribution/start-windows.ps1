#Requires -Version 5.1
[CmdletBinding()]
param(
    [ValidatePattern('\A[a-z0-9][a-z0-9_-]{0,62}\z')]
    [string]$ProjectName,
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [ValidatePattern('^[0-9a-fA-F]{64}$')]
    [string]$ExpectedManifestSha256,
    [switch]$VerifyOnly,
    [switch]$ShowInitialAdmin,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$pathPattern = '[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*'
$projectStateFile = 'windows-project.state'
$ownerLabel = 'io.vision-bill.launcher-owner'
$installationLabel = 'io.vision-bill.installation'

function Get-PlainItem([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Use regular files and directories for the bundle and its settings. Check $Path."
    }
    return $item
}

function Get-PlainFile([string]$Path) {
    $item = Get-PlainItem $Path
    if ($item.PSIsContainer) { throw "Expected a file at $Path." }
    return $item
}

function Get-Checksum([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-BundleDirectory([string]$Directory, [string]$Prefix) {
    foreach ($child in Get-ChildItem -LiteralPath $Directory -Force) {
        $relative = "$Prefix$($child.Name)"
        $null = Get-PlainItem $child.FullName
        if ($child.PSIsContainer) {
            if (-not $expectedDirectories.Contains($relative)) {
                throw "Unexpected directory in the bundle: $relative"
            }
            Test-BundleDirectory $child.FullName "$relative/"
        } elseif ($relative -cnotin @('SHA256SUMS', 'runtime.env', $projectStateFile)) {
            if (-not $expectedFiles.ContainsKey($relative)) {
                throw "Unexpected file in the bundle: $relative"
            }
            if ((Get-Checksum $child.FullName) -cne $expectedFiles[$relative]) {
                throw "Checksum mismatch: $relative"
            }
            $null = $verifiedFiles.Add($relative)
        }
    }
}

function Assert-Bundle {
    $null = Get-PlainItem $root
    $manifestFile = Get-PlainFile (Join-Path $root 'SHA256SUMS')
    if ($manifestFile.Length -gt 1MB) { throw 'The bundle checksum list is too large.' }
    if ($ExpectedManifestSha256 -and (Get-Checksum $manifestFile.FullName) -ne $ExpectedManifestSha256) {
        throw 'SHA256SUMS differs from the trusted manifest hash.'
    }
    $expectedFiles = [Collections.Generic.Dictionary[string, string]]::new([StringComparer]::Ordinal)
    $visitedPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in [IO.File]::ReadAllLines($manifestFile.FullName)) {
        if ($line -cnotmatch "^([0-9a-f]{64})  ($pathPattern)$") {
            throw 'The bundle checksum list contains a malformed line.'
        }
        $checksum = $Matches[1]
        $relative = $Matches[2]
        if (-not $visitedPaths.Add($relative)) { throw "Repeated bundle checksum path: $relative" }
        $expectedFiles.Add($relative, $checksum)
    }

    $launcher = 'start-windows.ps1'
    $requiredFiles = @('compose.yaml', 'release.json', 'runtime.env.example', 'check-compose.sh', $launcher, 'start-linux.sh')
    foreach ($relative in $requiredFiles) {
        if (-not $expectedFiles.ContainsKey($relative)) { throw "Missing bundle checksum: $relative" }
    }
    $releaseFile = Get-PlainFile (Join-Path $root 'release.json')
    if ($releaseFile.Length -gt 1MB -or (Get-Checksum $releaseFile.FullName) -cne $expectedFiles['release.json']) {
        throw 'Invalid checksum or size for release.json.'
    }
    $release = ConvertFrom-Json -InputObject ([IO.File]::ReadAllText($releaseFile.FullName))
    if ($release.schema_version -ne 2 -or $release.variant -cne 'app-ocr-cuda' -or $release.platform -cne 'linux/amd64') {
        throw 'Extract a verified app-ocr-cuda release to use this launcher.'
    }
    $expectedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $null = $expectedDirectories.Add('release')
    $assets = @($release.assets.PSObject.Properties)
    if ($assets.Count -eq 0 -or $expectedFiles.Count -ne $requiredFiles.Count + $assets.Count) {
        throw 'The checksum list does not match the release file inventory.'
    }
    foreach ($asset in $assets) {
        $relative = "release/$($asset.Name)"
        if ($asset.Name -cnotmatch "^$pathPattern$" -or -not $expectedFiles.ContainsKey($relative) -or
            $expectedFiles[$relative] -cne $asset.Value.sha256) {
            throw "The asset differs from the release file inventory: $relative"
        }
        while ($relative.Contains('/')) {
            $relative = $relative.Substring(0, $relative.LastIndexOf('/'))
            $null = $expectedDirectories.Add($relative)
        }
    }
    $verifiedFiles = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    Test-BundleDirectory $root ''
    if ($verifiedFiles.Count -ne $expectedFiles.Count) { throw 'The extracted bundle is incomplete.' }
}

function Assert-PrivateFile([string]$Path, [System.Security.Principal.SecurityIdentifier]$User) {
    $null = Get-PlainFile $Path
    $acl = Get-Acl -LiteralPath $Path
    $name = [IO.Path]::GetFileName($Path)
    $trustedSids = @($User.Value, 'S-1-5-18', 'S-1-5-32-544')
    foreach ($rule in $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -notin $trustedSids) {
            throw "Restrict access to $name to your Windows account, SYSTEM and Administrators before starting."
        }
    }
    if ($acl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -notin $trustedSids) {
        throw "Keep $name owned by your Windows account, SYSTEM or Administrators."
    }
}

function Write-PrivateLines([string]$Path, $User, [string[]]$Lines, [switch]$Replace) {
    $temporary = Join-Path $root ('.settings-' + [Guid]::NewGuid().ToString('N'))
    $stream = [IO.File]::Open($temporary, [IO.FileMode]::CreateNew)
    $stream.Dispose()
    try {
        $acl = Get-Acl -LiteralPath $temporary
        $acl.SetOwner($User)
        $acl.SetAccessRuleProtection($true, $false)
        $acl.SetAccessRule([System.Security.AccessControl.FileSystemAccessRule]::new($User, 'FullControl', 'Allow'))
        Set-Acl -LiteralPath $temporary -AclObject $acl
        Assert-PrivateFile $temporary $User
        [IO.File]::WriteAllLines($temporary, $Lines, [Text.UTF8Encoding]::new($false))
        if ($Replace) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) }
        else { [IO.File]::Move($temporary, $Path) }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}

function Invoke-Docker([string[]]$Command) {
    $result = & $docker.Source @desktop @Command
    if ($LASTEXITCODE -ne 0) { throw "Docker could not verify the installation project ($($Command[0]) $($Command[1]))." }
    return $result
}

function Assert-ProjectResources([string]$Name, [string]$Owner) {
    foreach ($kind in @('container', 'network', 'volume')) {
        $format = if ($kind -eq 'container') { '{{.ID}}' } else { '{{.Name}}' }
        $command = @($kind, 'ls', '--filter', "label=com.docker.compose.project=$Name", '--format', $format)
        if ($kind -eq 'container') { $command += '--all' }
        $resources = @(Invoke-Docker $command)
        if ($kind -in @('network', 'volume')) {
            # Compose can also find volumes or the default network by name alone.
            $keys = if ($kind -eq 'volume') { $projectVolumeNames } else { 'default' }
            $names = @(Invoke-Docker @($kind, 'ls', '--filter', "name=${Name}_", '--format', '{{.Name}}'))
            foreach ($key in $keys) {
                $generated = "${Name}_$key"
                if ($names -ccontains $generated -and $resources -cnotcontains $generated) {
                    throw "Docker project $Name has a conflicting $kind named $generated."
                }
            }
        }
        if ($resources.Count -eq 0) { continue }
        if (-not $Owner) { throw "Docker project $Name already contains Compose resources." }
        $labelFormat = if ($kind -eq 'container') { '{{json .Config.Labels}}' } else { '{{json .Labels}}' }
        $reports = @(Invoke-Docker (@($kind, 'inspect', '--format', $labelFormat) + $resources))
        if ($reports.Count -ne $resources.Count) { throw "Docker project $Name has unverified $kind resources." }
        foreach ($report in $reports) {
            $labels = ConvertFrom-Json -InputObject $report
            $installation = if ($null -ne $labels) { $labels.PSObject.Properties[$installationLabel] }
            if ($null -eq $installation -or $installation.Value -cne $Owner) {
                throw "Docker project $Name has a $kind belonging to another installation."
            }
        }
    }
}

function Assert-ProjectClaim($Project) {
    $json = Invoke-Docker @('volume', 'inspect', $Project.Claim)
    $claims = @((($json -join "`n") | ConvertFrom-Json))
    if ($claims.Count -ne 1 -or $claims[0].Name -cne $Project.Claim -or $claims[0].Driver -cne 'local') {
        throw "Docker project $($Project.Name) has an invalid launcher claim."
    }
    $labels = $claims[0].Labels
    $owner = if ($null -ne $labels) { $labels.PSObject.Properties[$ownerLabel] }
    if ($null -eq $owner -or $owner.Value -cne $Project.Owner) {
        throw "Docker project $($Project.Name) is owned by another installation."
    }
}

function Get-InstallProject([string]$Requested, [string]$Path, $User) {
    if ($Requested -and $Requested -cnotmatch '\A[a-z0-9][a-z0-9_-]{0,62}\z') {
        throw 'Choose a lowercase Docker project name of 1 to 63 letters, digits, hyphens or underscores.'
    }
    if ([IO.File]::Exists($Path)) {
        Assert-PrivateFile $Path $User
        if ((Get-PlainFile $Path).Length -gt 128 -or
            [IO.File]::ReadAllText($Path) -cnotmatch '\A(pending|owned) ([a-z0-9][a-z0-9_-]{0,62}) ([0-9a-f]{32})\r?\n\z') {
            throw 'The saved Windows launcher project identity is invalid.'
        }
        $phase, $name, $owner = $Matches[1], $Matches[2], $Matches[3]
        if ($Requested -and $Requested -cne $name) { throw "This extraction is already assigned to Docker project $name." }
        if ($phase -eq 'pending') { Assert-ProjectResources $name }
    } else {
        $phase = 'pending'
        $name = if ($Requested) { $Requested } else { 'vision-bill-' + [Guid]::NewGuid().ToString('N') }
        $owner = [Guid]::NewGuid().ToString('N')
        Assert-ProjectResources $name
        Write-PrivateLines $Path $User @("$phase $name $owner")
    }
    $project = @{ Name = $name; Owner = $owner; Claim = "$name-vision-bill-launcher-owner" }
    if ($phase -eq 'pending') {
        $null = Invoke-Docker @('volume', 'create', '--driver', 'local', '--label', "$ownerLabel=$owner", $project.Claim)
        Assert-ProjectClaim $project
        Assert-ProjectResources $name
        Write-PrivateLines $Path $User @("owned $name $owner") -Replace
    }
    Assert-ProjectClaim $project
    return $project
}

function New-Runtime([string]$Path, [System.Security.Principal.SecurityIdentifier]$User) {
    do {
        $mode = (Read-Host 'LLM connection type (ollama, llamacpp, openai) [ollama]').Trim().ToLowerInvariant()
        if (-not $mode) { $mode = 'ollama' }
    } until ($mode -in @('ollama', 'llamacpp', 'openai'))
    $provider = if ($mode -eq 'ollama') { 'ollama' } else { 'openai' }
    $defaultEndpoint = switch ($mode) {
        'ollama' { 'http://host.docker.internal:11434' }
        'llamacpp' { 'http://host.docker.internal:8083/v1' }
    }
    do {
        $hint = if ($defaultEndpoint) { " [$defaultEndpoint]" } else { '' }
        $endpoint = (Read-Host "Private LLM URL (host.docker.internal for this PC)$hint").Trim()
        if (-not $endpoint) { $endpoint = $defaultEndpoint }
        $uri = $null
        $validEndpoint = [Uri]::TryCreate($endpoint, [UriKind]::Absolute, [ref]$uri) -and
            $uri.Scheme -in @('http', 'https') -and $uri.Host -and -not $uri.UserInfo -and -not $uri.IsLoopback
        if (-not $validEndpoint) { Write-Host 'Enter a reachable HTTP or HTTPS endpoint without embedded credentials.' }
    } until ($validEndpoint)
    do {
        $model = Read-Host 'Exact installed model ID'
    } until (-not [string]::IsNullOrWhiteSpace($model))
    $lines = @("LLM__PROVIDER=$provider", "LLM__HOST=$endpoint", "LLM__MODEL_NAME=$model")
    if ($mode -eq 'llamacpp') { $lines += 'LLM__MODEL_LIFECYCLE=llamacpp' }
    if ($provider -eq 'openai') {
        $secret = Read-Host 'API key (press Enter for a credentialless endpoint)' -AsSecureString
        if ($secret.Length -gt 0) {
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
            try { $lines += 'LLM__API_KEY=' + [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
            finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
        }
        $secret.Dispose()
    }
    if ($lines -match '[\r\n\x00]') { throw 'Runtime settings must fit on individual lines.' }

    Write-PrivateLines $Path $User $lines
}

Assert-Bundle
if ($VerifyOnly) { Write-Host 'Bundle files verified.'; return }
if ($env:OS -ne 'Windows_NT') { throw 'Start this launcher in Windows PowerShell.' }
if ([IO.Path]::GetPathRoot($root) -cnotmatch '\A[A-Za-z]:\\\z') {
    throw 'Extract the release to a local Windows drive before starting Vision Bill.'
}
for ($directory = [IO.DirectoryInfo]::new($root); $null -ne $directory; $directory = $directory.Parent) {
    $null = Get-PlainItem $directory.FullName
}
$docker = (Get-Command docker.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1)
if (-not $docker) { throw 'Install Docker Desktop for Windows and open it before starting Vision Bill.' }
$desktop = @('--context', 'desktop-linux')
$dockerEndpoint = & $docker.Source @desktop context inspect desktop-linux --format '{{.Endpoints.docker.Host}}'
if ($LASTEXITCODE -ne 0 -or $dockerEndpoint -ne 'npipe:////./pipe/dockerDesktopLinuxEngine') {
    throw 'Open Docker Desktop, enable its WSL 2 engine and switch to Linux containers.'
}
$engine = & $docker.Source @desktop info --format '{{.OSType}}|{{.Architecture}}'
if ($LASTEXITCODE -ne 0 -or $engine -ne 'linux|x86_64') {
    throw 'Start Docker Desktop with its Linux x86_64 engine and run the launcher again.'
}
$composeVersion = & $docker.Source @desktop compose version --short
if ($LASTEXITCODE -ne 0 -or $composeVersion -notmatch '^v?(\d+)\.(\d+)\.(\d+)') {
    throw 'Update Docker Desktop to a version with Docker Compose 2.30 or later.'
}
if ([int]$Matches[1] -lt 2 -or ([int]$Matches[1] -eq 2 -and [int]$Matches[2] -lt 30)) {
    throw 'Update Docker Desktop to a version with Docker Compose 2.30 or later.'
}

$runtime = Join-Path $root 'runtime.env'
$userSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$composePath = Join-Path $root 'compose.yaml'
$compose = [IO.File]::ReadAllText($composePath) | ConvertFrom-Json
$projectVolumeNames = $compose.volumes.PSObject.Properties.Name
$project = Get-InstallProject $ProjectName (Join-Path $root $projectStateFile) $userSid
$createdRuntime = -not [IO.File]::Exists($runtime)
if ($createdRuntime) { New-Runtime $runtime $userSid }
Assert-PrivateFile $runtime $userSid

$composeArgs = @('compose', '--file', $composePath, '--project-name', $project.Name)
$previousProject = $env:COMPOSE_PROJECT_NAME
$previousInstallation = $env:VISION_BILL_INSTALLATION_ID
$previousBind = $env:APP_BIND_ADDRESS
$previousPort = $env:APP_PORT
try {
    $env:COMPOSE_PROJECT_NAME = $project.Name
    $env:VISION_BILL_INSTALLATION_ID = $project.Owner
    $env:APP_BIND_ADDRESS = '127.0.0.1'
    $env:APP_PORT = [string]$Port
    Assert-ProjectClaim $project
    Assert-ProjectResources $project.Name $project.Owner
    & $docker.Source @desktop @composeArgs up --detach --no-build --wait --wait-timeout 300
    if ($LASTEXITCODE -ne 0) { throw 'Vision Bill could not start. Check the Compose output and rerun this launcher.' }

    $url = "http://127.0.0.1:$Port/"
    $deadline = [Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    do {
        try {
            $response = Invoke-WebRequest -Uri "${url}api/v1/system/ui-config" -UseBasicParsing -TimeoutSec 2
            $ready = $response.StatusCode -eq 200
        } catch {
            Start-Sleep -Seconds 1
        }
    } until ($ready -or $deadline.Elapsed.TotalSeconds -ge 60)
    if (-not $ready) { throw "The app did not answer at $url. Check the app container in Docker Desktop." }

    $displayAdmin = $ShowInitialAdmin -or ($createdRuntime -and (Read-Host 'Show initial administrator credentials here? [y/N]') -in @('y', 'yes'))
    if ($displayAdmin) {
        Assert-ProjectClaim $project
        Assert-ProjectResources $project.Name $project.Owner
        Write-Host 'Initial administrator credentials for this installation'
        & $docker.Source @desktop @composeArgs exec -T app cat /app/config/initial-admin.yaml
        if ($LASTEXITCODE -ne 0) { Write-Host 'Use the administrator account for your existing installation.' }
    }
    Write-Host "Vision Bill project $($project.Name) is running at $url"
    Write-Host 'Sign in as an administrator. With Ollama or a dedicated llama.cpp router, open Settings > Dedicated OCR to start setup.'
    if (-not $NoBrowser) { Start-Process $url }
} finally {
    $env:COMPOSE_PROJECT_NAME = $previousProject
    $env:VISION_BILL_INSTALLATION_ID = $previousInstallation
    $env:APP_BIND_ADDRESS = $previousBind
    $env:APP_PORT = $previousPort
}
