import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

test("the Windows launcher reuses its claim and rejects a foreign project or claim", (t) => {
  const powershellCommands = process.platform === "win32" ? ["powershell.exe", "pwsh"] : ["pwsh"];
  const powershell = powershellCommands.find((command) =>
    spawnSync(command, ["-NoProfile", "-Command", "exit 0"]).status === 0,
  );
  if (!powershell) return t.skip("PowerShell is unavailable");

  const directory = mkdtempSync(join(tmpdir(), "vb-windows-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const script = join(directory, "project-claim.ps1");
  writeFileSync(script, String.raw`
param($Launcher, $Directory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Launcher, [ref]$tokens, [ref]$errors)
if ($errors) { throw ($errors | Out-String) }
foreach ($definition in $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    . ([scriptblock]::Create($definition.Extent.Text))
}
$ownerLabel = 'io.vision-bill.launcher-owner'
$installationLabel = 'io.vision-bill.installation'
$projectVolumeNames = @('app-config')
$mock = @{ Claims = @{}; ForeignProject = 'existing'; Creates = 0 }
function Assert-PrivateFile($Path, $User) { $null = Get-PlainFile $Path }
function Write-PrivateLines($Path, $User, [string[]]$Lines, [switch]$Replace) {
    if ([IO.File]::Exists($Path) -and -not $Replace) { throw 'Unexpected local-state overwrite' }
    [IO.File]::WriteAllLines($Path, $Lines)
}
function Invoke-Docker([string[]]$Command) {
    switch ("$($Command[0]) $($Command[1])") {
        'container ls' { if ($Command -contains "label=com.docker.compose.project=$($mock.ForeignProject)") { 'foreign-id' } }
        'container inspect' { '{"io.vision-bill.installation":"foreign-owner"}' }
        'network ls' { }
        'volume ls' { }
        'volume create' {
            $claim = $Command[-1]
            if (-not $mock.Claims.ContainsKey($claim)) { $mock.Claims[$claim] = ($Command[-2] -split '=', 2)[1] }
            $mock.Creates++
            $claim
        }
        'volume inspect' {
            $claim = $Command[-1]
            $labels = @{ 'io.vision-bill.launcher-owner' = $mock.Claims[$claim] }
            ConvertTo-Json -InputObject @(@{ Name = $claim; Driver = 'local'; Labels = $labels }) -Compress
        }
        default { throw 'Unexpected Docker call' }
    }
}
function Assert-Rejected([scriptblock]$Action, [string]$Pattern) {
    try { $null = & $Action }
    catch { if ($_.Exception.Message -like $Pattern) { return }; throw }
    throw 'Expected rejection'
}
$saved = Join-Path $Directory 'saved.state'
$project = Get-InstallProject 'first-install' $saved $null
$rerun = Get-InstallProject '' $saved $null
if ($rerun.Name -ne $project.Name -or $rerun.Owner -ne $project.Owner -or $mock.Creates -ne 1) { throw 'Rerun changed its recorded claim' }
Assert-Rejected { Get-InstallProject 'existing' (Join-Path $Directory 'existing.state') $null } '*already contains Compose resources*'
Assert-Rejected { Get-InstallProject $project.Name (Join-Path $Directory 'other.state') $null } '*owned by another installation*'
$mock.ForeignProject = $project.Name
Assert-Rejected { Assert-ProjectResources $project.Name $project.Owner } '*belonging to another installation*'
`);
  const launcher = fileURLToPath(new URL("../distribution/start-windows.ps1", import.meta.url));
  const result = spawnSync(powershell, [
    "-NoProfile", "-NonInteractive",
    ...(process.platform === "win32" ? ["-ExecutionPolicy", "RemoteSigned"] : []),
    "-File", script, launcher, directory,
  ], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
});
